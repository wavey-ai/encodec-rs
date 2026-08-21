#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const options = parseArguments(process.argv.slice(2));
const bundleRoot = path.join(
  options.wasmRoot,
  "encodec-rs/bundles/encodec_48khz_12kbps_1333ms",
);

const report = {
  schema: "yl.vin.encodec-wasm-session-profile",
  schemaVersion: 1,
  startedAt: new Date().toISOString(),
  status: "starting",
  configuration: {
    trackA: options.trackA,
    trackB: options.trackB,
    wasmRoot: options.wasmRoot,
    bundleRoot,
    randomSeed: options.seed,
    onnxThreads: options.threads,
    normalUsageGateRealtimeFactor: 2,
  },
  runs: [],
};

try {
  await checkpoint();
  const [ort, encodec] = await Promise.all([
    import(pathToFileURL(path.join(options.wasmRoot, "onnxruntime-web/ort.wasm.min.mjs"))),
    import(pathToFileURL(path.join(options.wasmRoot, "encodec-rs/pkg/encodec_rs.js"))),
  ]);
  configureOrt(ort);
  encodec.initSync({
    module: await readFile(path.join(options.wasmRoot, "encodec-rs/pkg/encodec_rs_bg.wasm")),
  });
  encodec.initPanicHook?.();

  const bundleJson = await readFile(path.join(bundleRoot, "bundle.json"), "utf8");
  const metadata = JSON.parse(bundleJson);
  const lmWeights = new Uint8Array(
    await readFile(path.join(bundleRoot, metadata.lm_quant_weight_model)),
  );
  const modelBytes = await readFile(path.join(bundleRoot, metadata.encode_model));
  validateMetadata(metadata);

  log("Decoding the two real pipeline WAV files");
  const [trackA, trackB] = await Promise.all([
    loadWave(options.trackA, metadata),
    loadWave(options.trackB, metadata),
  ]);
  report.sources = [summarizeTrack("A", trackA), summarizeTrack("B", trackB)];
  await checkpoint("decoded");

  log(`A isolated clean session: ${trackA.chunkCount} chunks`);
  const isolatedA = await runIsolated({
    label: "isolated-a",
    track: trackA,
    ort,
    encodec,
    modelBytes,
    metadata,
    bundleJson,
    lmWeights,
  });
  report.runs.push(isolatedA.summary);
  await checkpoint("isolated-a-complete");

  log(`B isolated clean session: ${trackB.chunkCount} chunks`);
  const isolatedB = await runIsolated({
    label: "isolated-b",
    track: trackB,
    ort,
    encodec,
    modelBytes,
    metadata,
    bundleJson,
    lmWeights,
  });
  report.runs.push(isolatedB.summary);
  await checkpoint("isolated-b-complete");

  log(`A+B random interleave on one clean shared session: ${trackA.chunkCount + trackB.chunkCount} chunks`);
  const interleaved = await runInterleaved({
    tracks: [trackA, trackB],
    baselines: [isolatedA.hashes, isolatedB.hashes],
    ort,
    encodec,
    modelBytes,
    metadata,
    bundleJson,
    lmWeights,
  });
  report.runs.push(interleaved.summary);
  report.determinism = interleaved.determinism;
  report.status = interleaved.determinism.pass ? "passed" : "failed";
  report.finishedAt = new Date().toISOString();
  await checkpoint();
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!interleaved.determinism.pass) process.exitCode = 1;
} catch (error) {
  report.status = "error";
  report.finishedAt = new Date().toISOString();
  report.error = String(error?.stack || error);
  await checkpoint().catch(() => {});
  process.stderr.write(`${report.error}\n`);
  process.exitCode = 1;
}

function parseArguments(arguments_) {
  const positional = [];
  const parsed = {
    wasmRoot: path.resolve(repoRoot, "../vin.yl.vendor/wasm"),
    output: path.join(repoRoot, "target/performance/encodec-wasm-session-profile.json"),
    seed: 0x594c5649,
    threads: 1,
    progressEvery: 5,
  };
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === "--wasm-root") parsed.wasmRoot = path.resolve(arguments_[++index]);
    else if (argument === "--output") parsed.output = path.resolve(arguments_[++index]);
    else if (argument === "--seed") parsed.seed = Number(arguments_[++index]);
    else if (argument === "--threads") parsed.threads = Number(arguments_[++index]);
    else if (argument === "--progress-every") parsed.progressEvery = Number(arguments_[++index]);
    else if (argument === "--help" || argument === "-h") usage();
    else positional.push(path.resolve(argument));
  }
  if (positional.length !== 2) usage("Provide exactly two 48 kHz stereo WAV files.");
  parsed.trackA = positional[0];
  parsed.trackB = positional[1];
  if (!Number.isInteger(parsed.threads) || parsed.threads < 1) {
    usage("--threads must be a positive integer.");
  }
  if (!Number.isInteger(parsed.progressEvery) || parsed.progressEvery < 1) {
    usage("--progress-every must be a positive integer.");
  }
  return parsed;
}

function usage(error = "") {
  if (error) process.stderr.write(`${error}\n\n`);
  process.stderr.write([
    "Usage:",
    "  node scripts/profile-encodec-wasm-session.mjs [options] track-a.wav track-b.wav",
    "",
    "Options:",
    "  --wasm-root <dir>      Production WASM asset root",
    "  --output <file>         Incremental JSON report path",
    "  --seed <integer>        Interleave shuffle seed",
    "  --threads <count>       ONNX Runtime WASM thread count",
    "  --progress-every <n>    Progress interval in chunks",
  ].join("\n"));
  process.exit(error ? 1 : 0);
}

function configureOrt(ort) {
  ort.env.wasm.wasmPaths = `${pathToFileURL(path.join(options.wasmRoot, "onnxruntime-web"))}/`;
  ort.env.wasm.numThreads = options.threads;
}

function validateMetadata(metadata) {
  const expected = {
    sample_rate: 48_000,
    channels: 2,
    segment_samples: 64_960,
    segment_stride: 64_000,
    frame_length: 203,
  };
  for (const [key, value] of Object.entries(expected)) {
    if (metadata[key] !== value) {
      throw new Error(`Unexpected production bundle ${key}: ${metadata[key]} (expected ${value})`);
    }
  }
}

async function loadWave(filename, metadata) {
  const bytes = await readFile(filename);
  const wave = decodeWave(bytes);
  if (wave.sampleRate !== metadata.sample_rate || wave.channels !== metadata.channels) {
    throw new Error(
      `${filename} is ${wave.channels}ch ${wave.sampleRate} Hz; expected `
        + `${metadata.channels}ch ${metadata.sample_rate} Hz.`,
    );
  }
  return {
    id: filename === options.trackA ? "A" : "B",
    filename,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    ...wave,
    chunkCount: Math.ceil(wave.frames / metadata.segment_stride),
  };
}

function summarizeTrack(id, track) {
  return {
    id,
    filename: track.filename,
    sha256: track.sha256,
    sampleRate: track.sampleRate,
    channels: track.channels,
    bitsPerSample: track.bitsPerSample,
    frames: track.frames,
    seconds: round(track.frames / track.sampleRate, 6),
    chunks: track.chunkCount,
  };
}

async function createSession(ort, modelBytes) {
  const started = performance.now();
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return { session, sessionCreateMs: round(performance.now() - started) };
}

async function runIsolated(context) {
  const { session, sessionCreateMs } = await createSession(context.ort, context.modelBytes);
  const records = [];
  const hashes = [];
  const started = performance.now();
  try {
    for (let index = 0; index < context.track.chunkCount; index += 1) {
      const result = await encodeChunk({ ...context, session, index });
      hashes.push(result.sha256);
      records.push(performanceRecord({
        ...result,
        order: index,
        trackId: context.track.id,
        cumulativeMs: performance.now() - started,
      }));
      progress(context.label, index + 1, context.track.chunkCount, records);
    }
  } finally {
    await session.release?.();
  }
  return {
    hashes,
    summary: summarizeRun(context.label, sessionCreateMs, records),
  };
}

async function runInterleaved(context) {
  const { session, sessionCreateMs } = await createSession(context.ort, context.modelBytes);
  const work = [];
  for (let trackIndex = 0; trackIndex < context.tracks.length; trackIndex += 1) {
    for (let index = 0; index < context.tracks[trackIndex].chunkCount; index += 1) {
      work.push({ trackIndex, index });
    }
  }
  shuffle(work, options.seed);
  const records = [];
  const hashes = context.tracks.map((track) => Array(track.chunkCount));
  const started = performance.now();
  try {
    for (let order = 0; order < work.length; order += 1) {
      const item = work[order];
      const track = context.tracks[item.trackIndex];
      const result = await encodeChunk({ ...context, session, track, index: item.index });
      hashes[item.trackIndex][item.index] = result.sha256;
      records.push(performanceRecord({
        ...result,
        order,
        trackId: track.id,
        cumulativeMs: performance.now() - started,
      }));
      progress("interleaved", order + 1, work.length, records);
    }
  } finally {
    await session.release?.();
  }
  const mismatches = [];
  for (let trackIndex = 0; trackIndex < hashes.length; trackIndex += 1) {
    for (let index = 0; index < hashes[trackIndex].length; index += 1) {
      if (hashes[trackIndex][index] !== context.baselines[trackIndex][index]) {
        mismatches.push({
          trackId: context.tracks[trackIndex].id,
          chunkIndex: index,
          isolatedSha256: context.baselines[trackIndex][index],
          interleavedSha256: hashes[trackIndex][index],
        });
      }
    }
  }
  return {
    summary: summarizeRun("interleaved-a-b", sessionCreateMs, records),
    determinism: {
      pass: mismatches.length === 0,
      comparedChunks: work.length,
      mismatchedChunks: mismatches.length,
      mismatches,
      assertion: "Every ECDC chunk must match its isolated clean-session SHA-256 byte-for-byte.",
    },
  };
}

async function encodeChunk({
  track,
  index,
  session,
  ort,
  encodec,
  metadata,
  bundleJson,
  lmWeights,
}) {
  const startFrame = index * metadata.segment_stride;
  const ownedFrames = Math.min(metadata.segment_stride, track.frames - startFrame);
  const segment = buildSegment(track, startFrame, metadata);
  const input = new ort.Tensor("float32", segment, [
    1,
    metadata.channels,
    metadata.segment_samples,
  ]);
  let outputs;
  const chunkStarted = performance.now();
  const onnxStarted = performance.now();
  try {
    outputs = await session.run({ [session.inputNames[0]]: input });
    const onnxMs = performance.now() - onnxStarted;
    const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
    const codes = toUint16(codesTensor.data, index);
    const lmStarted = performance.now();
    const bytes = encodec.lmEcdcChunkFromFrame(
      bundleJson,
      lmWeights,
      Number(scaleTensor.data[0] ?? 1),
      codes,
      metadata.frame_length,
    );
    const lmWasmMs = performance.now() - lmStarted;
    return {
      chunkIndex: index,
      startFrame,
      ownedFrames,
      audioSeconds: ownedFrames / metadata.sample_rate,
      onnxMs,
      lmWasmMs,
      totalMs: performance.now() - chunkStarted,
      bytes: bytes.byteLength,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    };
  } finally {
    input.dispose?.();
    for (const tensor of Object.values(outputs || {})) tensor.dispose?.();
  }
}

function buildSegment(track, startFrame, metadata) {
  const contextFrames = (metadata.segment_samples - metadata.segment_stride) / 2;
  const output = new Float32Array(metadata.channels * metadata.segment_samples);
  const sourceStart = Math.max(0, startFrame - contextFrames);
  const targetStart = Math.max(0, contextFrames - startFrame);
  const copyFrames = Math.max(
    0,
    Math.min(track.frames - sourceStart, metadata.segment_samples - targetStart),
  );
  for (let channel = 0; channel < metadata.channels; channel += 1) {
    const sourceBase = channel * track.frames;
    const targetBase = channel * metadata.segment_samples;
    output.set(
      track.audio.subarray(sourceBase + sourceStart, sourceBase + sourceStart + copyFrames),
      targetBase + targetStart,
    );
  }
  return output;
}

function findEncodeOutputs(outputs) {
  const tensors = Object.values(outputs);
  const codesTensor = tensors.find((tensor) => tensor.type === "int64");
  const scaleTensor = tensors.find(
    (tensor) => tensor.type === "float32" && tensor.dims.length === 2,
  );
  if (!codesTensor || !scaleTensor) throw new Error("The encoder returned unexpected outputs.");
  return { codesTensor, scaleTensor };
}

function toUint16(values, chunkIndex) {
  const output = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const value = Number(values[index]);
    if (!Number.isInteger(value) || value < 0 || value > 65_535) {
      throw new Error(`Invalid code at chunk ${chunkIndex}, code ${index}: ${values[index]}`);
    }
    output[index] = value;
  }
  return output;
}

function performanceRecord(record) {
  const memory = process.memoryUsage();
  return {
    order: record.order,
    trackId: record.trackId,
    chunkIndex: record.chunkIndex,
    startFrame: record.startFrame,
    ownedFrames: record.ownedFrames,
    audioSeconds: round(record.audioSeconds, 6),
    onnxMs: round(record.onnxMs),
    lmWasmMs: round(record.lmWasmMs),
    totalMs: round(record.totalMs),
    cumulativeMs: round(record.cumulativeMs),
    realtimeFactor: round(record.audioSeconds / (record.totalMs / 1000)),
    bytes: record.bytes,
    sha256: record.sha256,
    rssMiB: round(memory.rss / 1_048_576),
    heapUsedMiB: round(memory.heapUsed / 1_048_576),
    externalMiB: round(memory.external / 1_048_576),
  };
}

function summarizeRun(label, sessionCreateMs, records) {
  const totalMs = sum(records, "totalMs");
  const audioSeconds = sum(records, "audioSeconds");
  const warmRecords = records.slice(1);
  const warmMs = sum(warmRecords, "totalMs");
  const warmAudioSeconds = sum(warmRecords, "audioSeconds");
  return {
    label,
    sessionCreateMs,
    chunks: records.length,
    audioSeconds: round(audioSeconds),
    timings: {
      firstChunkMs: records[0]?.totalMs ?? 0,
      onnxMs: round(sum(records, "onnxMs")),
      lmWasmMs: round(sum(records, "lmWasmMs")),
      totalEncodeMs: round(totalMs),
    },
    realtimeFactor: round(audioSeconds / (totalMs / 1000)),
    warmRealtimeFactor: round(warmAudioSeconds / (warmMs / 1000)),
    passesTwoTimesRealtimeWarmGate: warmMs > 0 && warmAudioSeconds / (warmMs / 1000) >= 2,
    chunksDetail: records,
  };
}

function sum(records, key) {
  return records.reduce((total, record) => total + Number(record[key] || 0), 0);
}

function progress(label, completed, total, records) {
  if (completed !== 1 && completed !== total && completed % options.progressEvery !== 0) return;
  const totalMs = sum(records, "totalMs");
  const audioSeconds = sum(records, "audioSeconds");
  log(
    `${label} ${completed}/${total}: ${round(totalMs / 1000, 1)}s wall, `
      + `${round(audioSeconds / (totalMs / 1000), 2)}x realtime, `
      + `${records.at(-1).rssMiB} MiB RSS`,
  );
}

function shuffle(values, seed) {
  let state = Number(seed) >>> 0;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
  for (let index = values.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [values[index], values[other]] = [values[other], values[index]];
  }
}

function decodeWave(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (ascii(view, 0, 4) !== "RIFF" || ascii(view, 8, 4) !== "WAVE") {
    throw new Error("Input is not a RIFF/WAVE file.");
  }
  let format;
  let dataOffset = 0;
  let dataBytes = 0;
  for (let offset = 12; offset + 8 <= view.byteLength;) {
    const id = ascii(view, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      const tag = view.getUint16(body, true);
      format = {
        tag,
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        bitsPerSample: view.getUint16(body + 14, true),
        subformat: tag === 0xfffe && size >= 40 ? view.getUint16(body + 24, true) : tag,
      };
    } else if (id === "data") {
      dataOffset = body;
      dataBytes = Math.min(size, view.byteLength - body);
    }
    offset = body + size + (size & 1);
  }
  if (!format || !dataOffset || !dataBytes) throw new Error("WAV has no usable fmt/data chunks.");
  const bytesPerSample = format.bitsPerSample / 8;
  const frames = Math.floor(dataBytes / (format.channels * bytesPerSample));
  const audio = new Float32Array(format.channels * frames);
  let cursor = dataOffset;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < format.channels; channel += 1) {
      audio[channel * frames + frame] = readSample(view, cursor, format);
      cursor += bytesPerSample;
    }
  }
  return { ...format, frames, audio };
}

function readSample(view, offset, format) {
  if (format.subformat === 3 && format.bitsPerSample === 32) {
    return view.getFloat32(offset, true);
  }
  if (format.subformat !== 1) {
    throw new Error(`Unsupported WAV format tag ${format.subformat}.`);
  }
  if (format.bitsPerSample === 16) return view.getInt16(offset, true) / 32_768;
  if (format.bitsPerSample === 24) {
    const unsigned = view.getUint8(offset)
      | (view.getUint8(offset + 1) << 8)
      | (view.getUint8(offset + 2) << 16);
    return ((unsigned << 8) >> 8) / 8_388_608;
  }
  if (format.bitsPerSample === 32) return view.getInt32(offset, true) / 2_147_483_648;
  throw new Error(`Unsupported PCM bit depth ${format.bitsPerSample}.`);
}

function ascii(view, offset, length) {
  let value = "";
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(view.getUint8(offset + index));
  }
  return value;
}

async function checkpoint(status = report.status) {
  report.status = status;
  report.updatedAt = new Date().toISOString();
  await mkdir(path.dirname(options.output), { recursive: true });
  await writeFile(options.output, `${JSON.stringify(report, null, 2)}\n`);
}

function log(message) {
  process.stderr.write(`[encodec-wasm-profile] ${message}\n`);
}

function round(value, digits = 3) {
  const scale = 10 ** digits;
  return Math.round(Number(value) * scale) / scale;
}
