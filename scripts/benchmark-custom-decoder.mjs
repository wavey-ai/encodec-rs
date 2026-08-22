#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

const repoRoot = path.resolve(import.meta.dirname, "..");
const stageRoot = path.resolve(
  process.argv[2] ?? path.join(repoRoot, "target/performance/custom-kernel/hybrid-decoder-1333"),
);
const runtimeRoot = path.resolve(
  process.argv[3] ?? path.join(repoRoot, "../vin.yl.vendor/wasm/onnxruntime-web"),
);
const kernelModulePath = path.resolve(
  process.argv[4] ??
    path.join(repoRoot, "target/performance/custom-kernel/convtranspose-first/encodec-convtranspose.mjs"),
);
const iterations = Number(process.argv[5] ?? 7);
const decoderInputsRoot = process.argv[6] ? path.resolve(process.argv[6]) : null;

if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error(`iterations must be a positive integer, got ${iterations}`);
}

const metadata = JSON.parse(fs.readFileSync(path.join(stageRoot, "metadata.json"), "utf8"));
const ort = await import(pathToFileURL(path.join(runtimeRoot, "ort.wasm.min.mjs")));
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = `${pathToFileURL(runtimeRoot).href}/`;

const sessionOptions = {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
  enableCpuMemArena: false,
  enableMemPattern: false,
};
const setupStarted = performance.now();
const fullSession = await ort.InferenceSession.create(
  fs.readFileSync(metadata.sourceModel),
  sessionOptions,
);
const stageSessions = [];
for (let stage = 0; stage < metadata.stages; ++stage) {
  stageSessions.push(
    await ort.InferenceSession.create(
      fs.readFileSync(path.join(stageRoot, `stage-${stage}.onnx`)),
      sessionOptions,
    ),
  );
}

const createKernelModule = (await import(pathToFileURL(kernelModulePath))).default;
const kernel = await createKernelModule({
  locateFile: (file) => path.join(path.dirname(kernelModulePath), file),
});
const kernelState = initializeKernel(kernel, stageRoot, metadata.layers);
const setupMs = performance.now() - setupStarted;

const decoderInputs = decoderInputsRoot
  ? loadDecoderInputs(decoderInputsRoot, metadata, ort)
  : {
      audioLength: metadata.segmentSamples,
      sampleRate: 48_000,
      frames: [
        {
          feeds: {
            codes: new ort.Tensor(
              "int64",
              deterministicCodes(metadata.numCodebooks * metadata.frameLength),
              [1, metadata.numCodebooks, metadata.frameLength],
            ),
            scale: new ort.Tensor("float32", new Float32Array([0.2]), [1, 1]),
          },
        },
      ],
    };
const windowLength = metadata.channels * metadata.segmentSamples;

const runFull = async () => {
  const windows = new Float32Array(decoderInputs.frames.length * windowLength);
  for (let frameIndex = 0; frameIndex < decoderInputs.frames.length; ++frameIndex) {
    const outputs = await fullSession.run(decoderInputs.frames[frameIndex].feeds);
    windows.set(outputs[fullSession.outputNames[0]].data, frameIndex * windowLength);
  }
  return windows;
};

const runHybrid = async () => {
  const stageMs = Array.from({ length: metadata.stages }, () => 0);
  const kernelMs = Array.from({ length: metadata.layers.length }, () => 0);
  const transferMs = Array.from({ length: metadata.layers.length }, () => 0);
  const windows = new Float32Array(decoderInputs.frames.length * windowLength);

  for (let frameIndex = 0; frameIndex < decoderInputs.frames.length; ++frameIndex) {
    const frameFeeds = decoderInputs.frames[frameIndex].feeds;
    let started = performance.now();
    let outputs = await stageSessions[0].run(frameFeeds);
    stageMs[0] += performance.now() - started;
    let activation = outputs[stageSessions[0].outputNames[0]].data;

    for (let layerIndex = 0; layerIndex < metadata.layers.length; ++layerIndex) {
      const layer = metadata.layers[layerIndex];
      const layerState = kernelState.layers[layerIndex];

      started = performance.now();
      kernel.HEAPF32.set(
        activation,
        kernelState.input / Float32Array.BYTES_PER_ELEMENT,
      );
      transferMs[layerIndex] += performance.now() - started;

      started = performance.now();
      const ok = kernel._conv_transpose1d_phase_simd_8x8(
        kernelState.input,
        layerState.packed,
        layerState.bias,
        kernelState.output,
        layer.inputTime,
        layer.inputChannels,
        layer.outputChannels,
        layer.stride,
      );
      if (ok !== 1) throw new Error(`custom kernel rejected layer ${layerIndex}`);
      kernelMs[layerIndex] += performance.now() - started;

      started = performance.now();
      const outputLength = layer.outputChannels * layer.rawOutputTime;
      const outputOffset = kernelState.output / Float32Array.BYTES_PER_ELEMENT;
      const customOutput = kernel.HEAPF32.slice(outputOffset, outputOffset + outputLength);
      const activationTensor = new ort.Tensor("float32", customOutput, [
        1,
        layer.outputChannels,
        layer.rawOutputTime,
      ]);
      transferMs[layerIndex] += performance.now() - started;

      const stage = stageSessions[layerIndex + 1];
      const feeds = {};
      for (const inputName of stage.inputNames) {
        feeds[inputName] = inputName === "scale" ? frameFeeds.scale : activationTensor;
      }
      started = performance.now();
      outputs = await stage.run(feeds);
      stageMs[layerIndex + 1] += performance.now() - started;
      activation = outputs[stage.outputNames[0]].data;
    }
    windows.set(activation, frameIndex * windowLength);
  }

  return { audio: windows, stageMs, kernelMs, transferMs };
};

let reference = await runFull();
let hybrid = await runHybrid();
for (let index = 0; index < 2; ++index) {
  if (index % 2 === 0) {
    reference = await runFull();
    hybrid = await runHybrid();
  } else {
    hybrid = await runHybrid();
    reference = await runFull();
  }
}

const fullSamples = [];
const hybridSamples = [];
const hybridBreakdowns = [];
for (let index = 0; index < iterations; ++index) {
  if (index % 2 === 0) {
    let started = performance.now();
    reference = await runFull();
    fullSamples.push(performance.now() - started);
    started = performance.now();
    hybrid = await runHybrid();
    hybridSamples.push(performance.now() - started);
  } else {
    let started = performance.now();
    hybrid = await runHybrid();
    hybridSamples.push(performance.now() - started);
    started = performance.now();
    reference = await runFull();
    fullSamples.push(performance.now() - started);
  }
  hybridBreakdowns.push(hybrid);
}

const fullMedian = median(fullSamples);
const hybridMedian = median(hybridSamples);
console.log(
  JSON.stringify(
    {
      environment: {
        runtime: "Node-hosted WebAssembly",
        onnxRuntime: "onnxruntime-web wasm",
        threads: 1,
        node: process.version,
      },
      geometry: {
        frameLength: metadata.frameLength,
        codebooks: metadata.numCodebooks,
        frames: decoderInputs.frames.length,
        sourceSamples: decoderInputs.audioLength,
        decodedWindowSamples: metadata.segmentSamples,
      },
      setupMs: round(setupMs),
      baseline: {
        medianMs: round(fullMedian),
        samplesMs: fullSamples.map(round),
        realtimeFactor: round(
          (decoderInputs.audioLength / decoderInputs.sampleRate) / (fullMedian / 1000),
        ),
        outputSha256: sha256(reference),
      },
      hybrid: {
        medianMs: round(hybridMedian),
        samplesMs: hybridSamples.map(round),
        speedup: round(fullMedian / hybridMedian),
        realtimeFactor: round(
          (decoderInputs.audioLength / decoderInputs.sampleRate) / (hybridMedian / 1000),
        ),
        medianStageMs: medianColumns(hybridBreakdowns.map((value) => value.stageMs)).map(round),
        medianKernelMs: medianColumns(hybridBreakdowns.map((value) => value.kernelMs)).map(round),
        medianTransferMs: medianColumns(hybridBreakdowns.map((value) => value.transferMs)).map(round),
        outputSha256: sha256(hybrid.audio),
      },
      parity: compareFloat32(reference, hybrid.audio),
    },
    null,
    2,
  ),
);

releaseKernel(kernel, kernelState);

function initializeKernel(module, root, layers) {
  const allocate = (length) => {
    const pointer = module._malloc(length * Float32Array.BYTES_PER_ELEMENT);
    if (pointer === 0) throw new Error(`failed to allocate ${length} float32 values`);
    return pointer;
  };
  const maxInput = Math.max(...layers.map((layer) => layer.inputChannels * layer.inputTime));
  const maxOutput = Math.max(
    ...layers.map((layer) => layer.outputChannels * layer.rawOutputTime),
  );
  const state = {
    input: allocate(maxInput),
    output: allocate(maxOutput),
    layers: [],
  };

  for (const layer of layers) {
    const weights = readFloat32(path.join(root, `layer-${layer.layer}-weight.f32le`));
    const bias = readFloat32(path.join(root, `layer-${layer.layer}-bias.f32le`));
    const rawWeights = allocate(weights.length);
    const packed = allocate(weights.length);
    const biasPointer = allocate(bias.length);
    module.HEAPF32.set(weights, rawWeights / Float32Array.BYTES_PER_ELEMENT);
    module.HEAPF32.set(bias, biasPointer / Float32Array.BYTES_PER_ELEMENT);
    const ok = module._pack_conv_transpose1d_weights(
      rawWeights,
      packed,
      layer.inputChannels,
      layer.outputChannels,
      layer.stride,
    );
    module._free(rawWeights);
    if (ok !== 1) throw new Error(`failed to pack layer ${layer.layer}`);
    state.layers.push({ packed, bias: biasPointer });
  }
  return state;
}

function releaseKernel(module, state) {
  module._free(state.input);
  module._free(state.output);
  for (const layer of state.layers) {
    module._free(layer.packed);
    module._free(layer.bias);
  }
}

function loadDecoderInputs(root, modelMetadata, runtime) {
  const fixture = JSON.parse(
    fs.readFileSync(path.join(root, "metadata.json"), "utf8"),
  );
  if (fixture.frameLength !== modelMetadata.frameLength) {
    throw new Error(
      `frame length mismatch: ${fixture.frameLength} != ${modelMetadata.frameLength}`,
    );
  }
  if (fixture.numCodebooks !== modelMetadata.numCodebooks) {
    throw new Error(
      `codebook count mismatch: ${fixture.numCodebooks} != ${modelMetadata.numCodebooks}`,
    );
  }
  if (fixture.channels !== modelMetadata.channels) {
    throw new Error(
      `channel count mismatch: ${fixture.channels} != ${modelMetadata.channels}`,
    );
  }
  if (!Array.isArray(fixture.frames) || fixture.frames.length === 0) {
    throw new Error("decoder input fixture contains no frames");
  }

  const codeCount = modelMetadata.numCodebooks * modelMetadata.frameLength;
  return {
    audioLength: fixture.audioLength,
    sampleRate: fixture.sampleRate,
    frames: fixture.frames.map((frame, frameIndex) => {
      const bytes = fs.readFileSync(path.join(root, frame.file));
      if (bytes.byteLength !== codeCount * Uint16Array.BYTES_PER_ELEMENT) {
        throw new Error(
          `frame ${frameIndex} has ${bytes.byteLength} code bytes; expected ${
            codeCount * Uint16Array.BYTES_PER_ELEMENT
          }`,
        );
      }
      const copy = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      );
      const sourceCodes = new Uint16Array(copy);
      const codes = new BigInt64Array(sourceCodes.length);
      for (let index = 0; index < sourceCodes.length; ++index) {
        codes[index] = BigInt(sourceCodes[index]);
      }
      return {
        ...frame,
        feeds: {
          codes: new runtime.Tensor("int64", codes, [
            1,
            modelMetadata.numCodebooks,
            modelMetadata.frameLength,
          ]),
          scale: new runtime.Tensor(
            "float32",
            new Float32Array([frame.scale]),
            [1, 1],
          ),
        },
      };
    }),
  };
}

function deterministicCodes(length) {
  const values = new BigInt64Array(length);
  let state = 0x243f6a88;
  for (let index = 0; index < length; ++index) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values[index] = BigInt(state & 1023);
  }
  return values;
}

function readFloat32(file) {
  const bytes = fs.readFileSync(file);
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(copy);
}

function compareFloat32(reference, candidate) {
  if (reference.length !== candidate.length) {
    throw new Error(`output length mismatch: ${reference.length} != ${candidate.length}`);
  }
  const referenceBits = new Uint32Array(reference.buffer, reference.byteOffset, reference.length);
  const candidateBits = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let exactMismatches = 0;
  let maxAbsError = 0;
  let squaredError = 0;
  let squaredSignal = 0;
  for (let index = 0; index < reference.length; ++index) {
    if (referenceBits[index] !== candidateBits[index]) exactMismatches += 1;
    const error = reference[index] - candidate[index];
    maxAbsError = Math.max(maxAbsError, Math.abs(error));
    squaredError += error * error;
    squaredSignal += reference[index] * reference[index];
  }
  return {
    exact: exactMismatches === 0,
    exactMismatches,
    maxAbsError,
    maxErrorDbfs: maxAbsError === 0 ? "-Infinity" : 20 * Math.log10(maxAbsError),
    rmse: Math.sqrt(squaredError / reference.length),
    snrDb: squaredError === 0 ? "Infinity" : 10 * Math.log10(squaredSignal / squaredError),
  };
}

function medianColumns(rows) {
  if (rows.length === 0) return [];
  return rows[0].map((_, column) => median(rows.map((row) => row[column])));
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function sha256(values) {
  return crypto
    .createHash("sha256")
    .update(Buffer.from(values.buffer, values.byteOffset, values.byteLength))
    .digest("hex");
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}
