#!/usr/bin/env node

import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath, pathToFileURL } from "node:url";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const bundleDir = path.resolve(
  repoRoot,
  "../encodec-worker/wasm/encodec_48khz_12kbps_1333ms",
);
const runtimeModulePath = path.join(
  repoRoot,
  "dist/wasm-fixed-bundles/encodec-ecdc-runtime.js",
);
const wasmModulePath = path.join(repoRoot, "pkg/encodec_rs.js");
const wasmBinaryPath = path.join(repoRoot, "pkg/encodec_rs_bg.wasm");
const ortModuleUrl = pathToFileURL(
  path.join(
    repoRoot,
    "browser-smoke/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs",
  ),
).href;
const ortWasmBaseUrl = pathToFileURL(
  path.join(repoRoot, "browser-smoke/node_modules/onnxruntime-web/dist") + path.sep,
).href;
const reportPath = path.join(repoRoot, "tmp/encodec-memory-benchmark.json");
const sampleIntervalMs = 15;
const settleDelayMs = 60;
const finalDelayMs = 250;
const segmentDurationMs = 1333;

const stageNames = [
  "Node process started",
  "Encoder module imported",
  "WASM runtime initialised",
  "ONNX model bytes loaded",
  "LM weights loaded",
  "Encoder instance created",
  "Production PCM segment allocated",
  "First segment encoded",
  "Five segments encoded",
  "Twenty segments encoded",
  "Encoder references released",
  "Garbage collection requested",
  "Final settled memory after a short delay",
];

let report = {
  nodeVersion: process.version,
  platform: process.platform,
  architecture: process.arch,
  benchmarkTimestamp: new Date().toISOString(),
  benchmarkScript: path.relative(repoRoot, fileURLToPath(import.meta.url)),
  bundleDir,
  checkpoints: [],
  peakSampled: null,
  timings: {},
  modelFileSizes: {},
  encoderConfiguration: {},
  error: null,
};

try {
  report = await runBenchmark();
  mkdirSync(path.dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  printReport(report);
} catch (error) {
  report.error = {
    message: error?.message ?? String(error),
    stack: error?.stack ?? null,
  };
  mkdirSync(path.dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.error(report.error.stack ?? report.error.message);
  process.exitCode = 1;
}

async function runBenchmark() {
  const sampler = createMemorySampler(sampleIntervalMs);
  const checkpoints = [];

  const bundleJsonPath = path.join(bundleDir, "bundle.json");
  const encodeModelPath = path.join(bundleDir, "encode_frame.onnx");
  const lmWeightsPath = path.join(bundleDir, "lm_weights_q8.bin");
  const bundleJson = readFileSync(bundleJsonPath, "utf8");
  const bundleMeta = JSON.parse(bundleJson);

  report.modelFileSizes = {
    encodeFrameOnnxBytes: statSync(encodeModelPath).size,
    lmWeightsQ8Bytes: statSync(lmWeightsPath).size,
    totalModelBytes:
      statSync(encodeModelPath).size + statSync(lmWeightsPath).size,
  };
  report.encoderConfiguration = {
    modelProfile: "encodec_48khz_12kbps_1333ms",
    sampleRate: bundleMeta.sample_rate,
    channels: bundleMeta.channels,
    requestedSegmentDurationMs: segmentDurationMs,
    modelSegmentSamples: bundleMeta.segment_samples,
    segmentStrideSamples: bundleMeta.segment_stride,
    frameLength: bundleMeta.frame_length,
    numCodebooks: bundleMeta.num_codebooks,
    pcmRepresentation: "Float32Array planar (channel-major)",
  };

  let runtimeModule = null;
  let wasmModule = null;
  let ort = null;
  let modelBytes = null;
  let lmWeights = null;
  let encodeSession = null;
  let pcmSegment = null;
  let firstChunk = null;
  let lastChunk = null;

  await checkpoint(checkpoints, sampler, 1, { gcBefore: true });

  runtimeModule = await importEsmFromJsFile(runtimeModulePath);
  if (typeof runtimeModule.createEncodecEcdcRuntime !== "function") {
    throw new Error("encodec runtime module is missing createEncodecEcdcRuntime()");
  }
  wasmModule = await importEsmFromJsFile(wasmModulePath);
  ort = await import(ortModuleUrl);
  await checkpoint(checkpoints, sampler, 2, { gcBefore: true });

  sampler.start();
  const initialisationStarted = performance.now();
  ort.env.wasm.wasmPaths = ortWasmBaseUrl;
  ort.env.wasm.numThreads = 1;
  const wasmBytes = readFileSync(wasmBinaryPath);
  wasmModule.initSync({ module: wasmBytes });
  wasmModule.initPanicHook?.();
  await checkpoint(checkpoints, sampler, 3, { gcBefore: true });

  modelBytes = toOwnedUint8Array(readFileSync(encodeModelPath));
  await checkpoint(checkpoints, sampler, 4, { gcBefore: true });

  lmWeights = toOwnedUint8Array(readFileSync(lmWeightsPath));
  await checkpoint(checkpoints, sampler, 5, { gcBefore: true });

  encodeSession = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  await checkpoint(checkpoints, sampler, 6, { gcBefore: true });

  pcmSegment = buildDeterministicPcmSegment(bundleMeta);
  await checkpoint(checkpoints, sampler, 7, { gcBefore: true });

  const firstEncodeStarted = performance.now();
  firstChunk = await encodeSegment({
    ort,
    wasmModule,
    encodeSession,
    bundleJson,
    bundleMeta,
    lmWeights,
    pcmSegment,
    segmentIndex: 0,
  });
  const firstEncodeMs = performance.now() - firstEncodeStarted;
  await checkpoint(checkpoints, sampler, 8, { gcBefore: false });

  const repeatedWarmStarted = performance.now();
  for (let segmentIndex = 1; segmentIndex < 5; segmentIndex += 1) {
    lastChunk = await encodeSegment({
      ort,
      wasmModule,
      encodeSession,
      bundleJson,
      bundleMeta,
      lmWeights,
      pcmSegment,
      segmentIndex,
    });
  }
  const segments2to5AverageMs =
    (performance.now() - repeatedWarmStarted) / 4;
  await checkpoint(checkpoints, sampler, 9, { gcBefore: false });

  const repeatedSteadyStarted = performance.now();
  for (let segmentIndex = 5; segmentIndex < 20; segmentIndex += 1) {
    lastChunk = await encodeSegment({
      ort,
      wasmModule,
      encodeSession,
      bundleJson,
      bundleMeta,
      lmWeights,
      pcmSegment,
      segmentIndex,
    });
  }
  const segments6to20AverageMs =
    (performance.now() - repeatedSteadyStarted) / 15;
  await checkpoint(checkpoints, sampler, 10, { gcBefore: false });
  sampler.stop();

  report.timings = {
    initialisationMs: roundMs(performance.now() - initialisationStarted),
    firstEncodeMs: roundMs(firstEncodeMs),
    segments2To5AverageMs: roundMs(segments2to5AverageMs),
    segments6To20AverageMs: roundMs(segments6to20AverageMs),
  };

  bestEffortReleaseSession(encodeSession);
  encodeSession = null;
  modelBytes = null;
  lmWeights = null;
  pcmSegment = null;
  firstChunk = null;
  lastChunk = null;
  ort = null;
  wasmModule = null;
  runtimeModule = null;
  await checkpoint(checkpoints, sampler, 11, { gcBefore: true });

  await requestGcAndSettle(settleDelayMs);
  await checkpoint(checkpoints, sampler, 12, { gcBefore: false });

  await delay(finalDelayMs);
  await checkpoint(checkpoints, sampler, 13, { gcBefore: false });

  report.checkpoints = checkpoints;
  report.peakSampled = finalizeMemorySnapshot(
    sampler.peak(),
    checkpoints[0].memory,
  );

  return report;
}

async function encodeSegment({
  ort,
  wasmModule,
  encodeSession,
  bundleJson,
  bundleMeta,
  lmWeights,
  pcmSegment,
  segmentIndex,
}) {
  let feeds = null;
  let outputs = null;

  try {
    feeds = {
      [encodeSession.inputNames[0]]: new ort.Tensor("float32", pcmSegment, [
        1,
        bundleMeta.channels,
        bundleMeta.segment_samples,
      ]),
    };
    outputs = await encodeSession.run(feeds);

    const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
    const expectedCodes = bundleMeta.num_codebooks * bundleMeta.frame_length;
    if (codesTensor.data.length !== expectedCodes) {
      throw new Error(
        `encoder returned ${codesTensor.data.length} codes, expected ${expectedCodes}`,
      );
    }

    const rawScale = Number(scaleTensor.data?.[0] ?? 1);
    const scale = Number.isFinite(rawScale) ? rawScale : 1;

    const chunk = wasmModule.lmEcdcChunkFromFrame(
      bundleJson,
      lmWeights,
      scale,
      toU16Codes(codesTensor.data, segmentIndex),
      bundleMeta.frame_length,
    );

    return chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
  } finally {
    disposeTensorMap(feeds);
    disposeTensorMap(outputs);
  }
}

function buildDeterministicPcmSegment(bundleMeta) {
  const frames = bundleMeta.segment_samples;
  const channels = bundleMeta.channels;
  const sampleRate = bundleMeta.sample_rate;
  const pcm = new Float32Array(channels * frames);

  for (let channel = 0; channel < channels; channel += 1) {
    const base = channel * frames;
    const phase = channel * Math.PI * 0.25;

    for (let frame = 0; frame < frames; frame += 1) {
      const t = frame / sampleRate;
      const sample =
        0.45 * Math.sin((2 * Math.PI * 220 * t) + phase) +
        0.24 * Math.sin((2 * Math.PI * 659.25 * t) + phase * 0.5) +
        0.12 * Math.sin((2 * Math.PI * 1234 * t) + phase * 1.5) +
        0.05 * Math.sin((2 * Math.PI * 37 * t));

      pcm[base + frame] = Math.max(-1, Math.min(1, sample));
    }
  }

  return pcm;
}

async function checkpoint(checkpoints, sampler, stageNumber, { gcBefore }) {
  if (gcBefore) {
    await requestGcAndSettle(settleDelayMs);
  } else {
    await delay(settleDelayMs);
  }

  sampler.sample();
  const baseline = checkpoints[0]?.memory ?? null;
  const memory = finalizeMemorySnapshot(process.memoryUsage(), baseline);
  checkpoints.push({
    stageNumber,
    stage: stageNames[stageNumber - 1],
    timestamp: new Date().toISOString(),
    memory,
  });
}

function createMemorySampler(intervalMs) {
  let intervalId = null;
  let peakSnapshot = process.memoryUsage();

  return {
    start() {
      if (intervalId !== null) {
        return;
      }

      peakSnapshot = process.memoryUsage();
      intervalId = setInterval(() => {
        peakSnapshot = maxMemorySnapshot(peakSnapshot, process.memoryUsage());
      }, intervalMs);
    },
    stop() {
      if (intervalId !== null) {
        clearInterval(intervalId);
        intervalId = null;
      }
      peakSnapshot = maxMemorySnapshot(peakSnapshot, process.memoryUsage());
    },
    sample() {
      peakSnapshot = maxMemorySnapshot(peakSnapshot, process.memoryUsage());
    },
    peak() {
      return peakSnapshot;
    },
  };
}

function finalizeMemorySnapshot(snapshot, baseline) {
  const values = {
    rss: Number(snapshot.rss),
    heapTotal: Number(snapshot.heapTotal),
    heapUsed: Number(snapshot.heapUsed),
    external: Number(snapshot.external),
    arrayBuffers: Number(snapshot.arrayBuffers),
  };

  return Object.fromEntries(
    Object.entries(values).map(([key, bytes]) => [
      key,
      {
        bytes,
        mib: roundMiB(bytes),
        deltaBytes: baseline ? bytes - baseline[key].bytes : 0,
        deltaMiB: roundMiB(baseline ? bytes - baseline[key].bytes : 0),
      },
    ]),
  );
}

function maxMemorySnapshot(left, right) {
  return {
    rss: Math.max(left.rss, right.rss),
    heapTotal: Math.max(left.heapTotal, right.heapTotal),
    heapUsed: Math.max(left.heapUsed, right.heapUsed),
    external: Math.max(left.external, right.external),
    arrayBuffers: Math.max(left.arrayBuffers, right.arrayBuffers),
  };
}

function findEncodeOutputs(outputs) {
  const tensors = Object.values(outputs || {});
  const codesTensor = tensors.find((tensor) => tensor.type === "int64");
  const scaleTensor = tensors.find(
    (tensor) => tensor.type === "float32" && tensor.dims.length === 2,
  );

  if (!codesTensor || !scaleTensor) {
    throw new Error(
      `unexpected encoder outputs: ${JSON.stringify(summarizeOutputs(outputs))}`,
    );
  }

  return { codesTensor, scaleTensor };
}

function summarizeOutputs(outputs) {
  return Object.fromEntries(
    Object.entries(outputs || {}).map(([name, tensor]) => [
      name,
      {
        type: tensor.type,
        dims: tensor.dims,
        length: tensor.data?.length ?? 0,
      },
    ]),
  );
}

function toU16Codes(codes, segmentIndex) {
  const frameCodes = new Uint16Array(codes.length);

  for (let index = 0; index < codes.length; index += 1) {
    const value = Number(codes[index]);
    if (!Number.isInteger(value) || value < 0 || value > 65535) {
      throw new Error(`invalid code at segment ${segmentIndex}, index ${index}`);
    }
    frameCodes[index] = value;
  }

  return frameCodes;
}

function disposeTensorMap(tensors) {
  for (const tensor of Object.values(tensors || {})) {
    try {
      tensor?.dispose?.();
    } catch (_error) {
      // Tensor disposal is best-effort.
    }
  }
}

function bestEffortReleaseSession(session) {
  try {
    session?.release?.();
  } catch (_error) {
    // Session release is best-effort.
  }
}

async function importEsmFromJsFile(filePath) {
  const source = readFileSync(filePath, "utf8");
  const url = `data:text/javascript;base64,${Buffer.from(source, "utf8").toString("base64")}`;
  return import(url);
}

function toOwnedUint8Array(bufferLike) {
  const source = bufferLike instanceof Uint8Array ? bufferLike : new Uint8Array(bufferLike);
  const owned = new Uint8Array(source.byteLength);
  owned.set(source);
  return owned;
}

async function requestGcAndSettle(delayMs) {
  if (typeof global.gc === "function") {
    global.gc();
  }
  await delay(delayMs);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function roundMiB(bytes) {
  return Number((bytes / (1024 * 1024)).toFixed(3));
}

function roundMs(ms) {
  return Number(ms.toFixed(3));
}

function formatBytes(bytes) {
  return `${roundMiB(bytes).toFixed(3)} MiB`;
}

function formatDelta(bytes) {
  const sign = bytes >= 0 ? "+" : "-";
  return `${sign}${roundMiB(Math.abs(bytes)).toFixed(3)} MiB`;
}

function printReport(result) {
  const checkpoints = result.checkpoints;
  const baseline = checkpoints[0].memory;
  const afterModelLoad = checkpoints[4].memory;
  const afterInit = checkpoints[5].memory;
  const afterFirstEncode = checkpoints[7].memory;
  const afterTwentyEncodes = checkpoints[9].memory;
  const finalMemory = checkpoints[12].memory;
  const peakSampled = result.peakSampled;

  console.log("EnCodec local memory benchmark");
  console.log("Model:");
  console.log(`  encode_frame.onnx: ${formatBytes(result.modelFileSizes.encodeFrameOnnxBytes)} (${result.modelFileSizes.encodeFrameOnnxBytes} bytes)`);
  console.log(`  lm_weights_q8.bin: ${formatBytes(result.modelFileSizes.lmWeightsQ8Bytes)} (${result.modelFileSizes.lmWeightsQ8Bytes} bytes)`);
  console.log(`  total model bytes: ${formatBytes(result.modelFileSizes.totalModelBytes)} (${result.modelFileSizes.totalModelBytes} bytes)`);
  console.log(`Baseline RSS: ${formatBytes(baseline.rss.bytes)}`);
  console.log(`RSS after model load: ${formatBytes(afterModelLoad.rss.bytes)}`);
  console.log(`RSS after encoder initialisation: ${formatBytes(afterInit.rss.bytes)}`);
  console.log(`RSS after first encode: ${formatBytes(afterFirstEncode.rss.bytes)}`);
  console.log(`RSS after 20 encodes: ${formatBytes(afterTwentyEncodes.rss.bytes)}`);
  console.log(`Peak sampled RSS: ${formatBytes(peakSampled.rss.bytes)}`);
  console.log(`Final RSS after release and GC: ${formatBytes(finalMemory.rss.bytes)}`);
  console.log(`Peak increase over baseline: ${formatBytes(peakSampled.rss.deltaBytes)}`);
  console.log(`Estimated model/init retained memory: ${formatBytes(afterInit.rss.deltaBytes)}`);
  console.log(`Observed growth across repeated encodes: ${formatBytes(afterTwentyEncodes.rss.bytes - afterFirstEncode.rss.bytes)}`);
  console.log(`Initialisation time: ${result.timings.initialisationMs.toFixed(3)} ms`);
  console.log(`First encode time: ${result.timings.firstEncodeMs.toFixed(3)} ms`);
  console.log(`Steady-state average encode time: ${result.timings.segments6To20AverageMs.toFixed(3)} ms`);
  console.log("");
  console.log("Checkpoints:");

  for (const checkpoint of checkpoints) {
    console.log(`${checkpoint.stageNumber}. ${checkpoint.stage}`);
    for (const key of ["rss", "heapTotal", "heapUsed", "external", "arrayBuffers"]) {
      const field = checkpoint.memory[key];
      console.log(
        `  ${key}: ${field.mib.toFixed(3)} MiB (${field.bytes} bytes, ${formatDelta(field.deltaBytes)} from baseline)`,
      );
    }
  }

  console.log("");
  console.log("Peak sampled values:");
  for (const key of ["rss", "heapTotal", "heapUsed", "external", "arrayBuffers"]) {
    const field = peakSampled[key];
    console.log(
      `  ${key}: ${field.mib.toFixed(3)} MiB (${field.bytes} bytes, ${formatDelta(field.deltaBytes)} from baseline)`,
    );
  }

  console.log("");
  console.log("Timings:");
  console.log(`  Initialisation: ${result.timings.initialisationMs.toFixed(3)} ms`);
  console.log(`  First encode: ${result.timings.firstEncodeMs.toFixed(3)} ms`);
  console.log(`  Average encode time for segments 2-5: ${result.timings.segments2To5AverageMs.toFixed(3)} ms`);
  console.log(`  Average encode time for segments 6-20: ${result.timings.segments6To20AverageMs.toFixed(3)} ms`);
  console.log("");
  console.log(`JSON report: ${path.relative(repoRoot, reportPath)}`);
}
