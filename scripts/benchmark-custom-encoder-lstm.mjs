#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

const repoRoot = path.resolve(import.meta.dirname, "..");
const root = path.resolve(
  process.argv[2] ?? path.join(repoRoot, "target/performance/custom-encoder/conv-kernel-v1"),
);
const runtimeRoot = path.resolve(
  process.argv[3] ?? path.join(repoRoot, "../vin.yl.vendor/wasm/onnxruntime-web"),
);
const kernelModulePath = path.resolve(
  process.argv[4] ?? path.join(root, "encodec-encoder-relaxed.mjs"),
);
const iterations = Number(process.argv[5] ?? 7);
if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error(`iterations must be a positive integer, got ${iterations}`);
}

const metadata = JSON.parse(fs.readFileSync(path.join(root, "metadata.json"), "utf8"));
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

const audio = deterministicAudio(metadata.channels, metadata.segmentSamples);
const audioTensor = new ort.Tensor(
  "float32",
  audio,
  [1, metadata.channels, metadata.segmentSamples],
);
const frontSession = await ort.InferenceSession.create(
  fs.readFileSync(path.join(root, "stage-0.onnx")),
  sessionOptions,
);
const frontOutputs = await frontSession.run({ audio: audioTensor });
const front = frontOutputs[metadata.stages[0].outputs[0]];
const frontData = front.data.slice();
await frontSession.release();

const recurrentSession = await ort.InferenceSession.create(
  fs.readFileSync(path.join(root, "stage-1.onnx")),
  sessionOptions,
);
const frontTensor = new ort.Tensor("float32", frontData, front.dims);
const runReference = async () => {
  const started = performance.now();
  const outputs = await recurrentSession.run({
    [metadata.stages[1].inputs[0]]: frontTensor,
  });
  return {
    elapsedMs: performance.now() - started,
    output: outputs[metadata.stages[1].outputs[0]],
  };
};

const createKernel = (await import(pathToFileURL(kernelModulePath))).default;
const kernel = await createKernel({
  locateFile: (file) => path.join(path.dirname(kernelModulePath), file),
});
const custom = prepareCustomLstm(frontData);
const runCustom = () => {
  const started = performance.now();
  const output = custom.run();
  return { elapsedMs: performance.now() - started, output };
};

let reference = await runReference();
let candidate = runCustom();
reference = await runReference();
candidate = runCustom();
const referenceSamples = [];
const candidateSamples = [];
for (let index = 0; index < iterations; ++index) {
  if (index % 2 === 0) {
    reference = await runReference();
    candidate = runCustom();
  } else {
    candidate = runCustom();
    reference = await runReference();
  }
  referenceSamples.push(reference.elapsedMs);
  candidateSamples.push(candidate.elapsedMs);
}

const hiddenSize = metadata.lstmLayers[0].hiddenSize;
const sequenceLength = metadata.frameLength;
const candidateNchw = transposeTimeMajorToPlanar(
  candidate.output,
  sequenceLength,
  hiddenSize,
);
const activationParity = compareFloat32(reference.output.data, candidateNchw);
const referenceActivation = reference.output.data.slice();
await recurrentSession.release();

const projectionSession = await ort.InferenceSession.create(
  fs.readFileSync(path.join(root, "stage-2.onnx")),
  sessionOptions,
);
const quantizerSession = await ort.InferenceSession.create(
  fs.readFileSync(path.join(root, "stage-3.onnx")),
  sessionOptions,
);
const runTail = async (activation) => {
  const tensor = new ort.Tensor("float32", activation, [1, hiddenSize, sequenceLength]);
  const projection = await projectionSession.run({
    [metadata.stages[2].inputs[0]]: tensor,
  });
  const quantized = await quantizerSession.run({
    [metadata.stages[3].inputs[0]]: projection[metadata.stages[2].outputs[0]],
  });
  return quantized.codes.data;
};
const referenceCodes = await runTail(referenceActivation);
const candidateCodes = await runTail(candidateNchw);

const referenceMedian = median(referenceSamples);
const candidateMedian = median(candidateSamples);
const report = {
  environment: {
    runtime: "Node-hosted WebAssembly",
    onnxRuntime: "onnxruntime-web wasm",
    threads: 1,
    node: process.version,
    kernelModule: kernelModulePath,
  },
  geometry: {
    sequenceLength,
    hiddenSize,
    layers: metadata.lstmLayers.length,
  },
  iterations,
  reference: {
    medianMs: round(referenceMedian),
    samplesMs: referenceSamples.map(round),
  },
  candidate: {
    medianMs: round(candidateMedian),
    samplesMs: candidateSamples.map(round),
    speedup: round(referenceMedian / candidateMedian),
  },
  parity: {
    activation: activationParity,
    codes: compareIntegers(referenceCodes, candidateCodes),
  },
};
console.log(JSON.stringify(report, null, 2));

custom.release();
await projectionSession.release();
await quantizerSession.release();

function prepareCustomLstm(input) {
  const hiddenSize = metadata.lstmLayers[0].hiddenSize;
  const sequenceLength = metadata.frameLength;
  const gateSize = 4 * hiddenSize;
  const pointers = {
    input: allocate(input.length),
    firstOutput: allocate(input.length),
    secondOutput: allocate(input.length),
    hidden: allocate(hiddenSize),
    cell: allocate(hiddenSize),
    inputProjection: allocate(sequenceLength * gateSize),
  };
  kernel.HEAPF32.set(input, pointers.input / 4);
  const layers = metadata.lstmLayers.map((layer) => {
    const inputWeights = readFloat32(
      path.join(root, `lstm-${layer.layer}-input-weight.f32le`),
    );
    const recurrentWeights = readFloat32(
      path.join(root, `lstm-${layer.layer}-recurrent-weight.f32le`),
    );
    const bias = readFloat32(path.join(root, `lstm-${layer.layer}-bias.f32le`));
    const unpackedInput = allocate(inputWeights.length);
    const unpackedRecurrent = allocate(recurrentWeights.length);
    const packedInput = allocate(inputWeights.length);
    const packedRecurrent = allocate(recurrentWeights.length);
    const biasPointer = allocate(bias.length);
    kernel.HEAPF32.set(inputWeights, unpackedInput / 4);
    kernel.HEAPF32.set(recurrentWeights, unpackedRecurrent / 4);
    kernel.HEAPF32.set(bias, biasPointer / 4);
    const inputOk = kernel._pack_linear_weights_8(
      unpackedInput,
      packedInput,
      hiddenSize,
      gateSize,
    );
    const recurrentOk = kernel._pack_linear_weights_8(
      unpackedRecurrent,
      packedRecurrent,
      hiddenSize,
      gateSize,
    );
    kernel._free(unpackedInput);
    kernel._free(unpackedRecurrent);
    if (inputOk !== 1 || recurrentOk !== 1) {
      throw new Error(`failed to pack LSTM layer ${layer.layer}`);
    }
    return { packedInput, packedRecurrent, bias: biasPointer };
  });

  const runLayer = (inputPointer, layer, outputPointer) => {
    const ok = kernel._lstm_layer_simd_64(
      inputPointer,
      layer.packedInput,
      layer.packedRecurrent,
      layer.bias,
      outputPointer,
      pointers.hidden,
      pointers.cell,
      pointers.inputProjection,
      sequenceLength,
      hiddenSize,
    );
    if (ok !== 1) throw new Error("custom LSTM rejected its geometry");
  };
  return {
    run() {
      runLayer(pointers.input, layers[0], pointers.firstOutput);
      runLayer(pointers.firstOutput, layers[1], pointers.secondOutput);
      const added = kernel._add_nhwc_in_place(
        pointers.secondOutput,
        pointers.input,
        input.length,
      );
      if (added !== 1) throw new Error("LSTM residual add failed");
      return kernel.HEAPF32.slice(
        pointers.secondOutput / 4,
        pointers.secondOutput / 4 + input.length,
      );
    },
    release() {
      for (const pointer of Object.values(pointers)) kernel._free(pointer);
      for (const layer of layers) {
        kernel._free(layer.packedInput);
        kernel._free(layer.packedRecurrent);
        kernel._free(layer.bias);
      }
    },
  };
}

function allocate(length) {
  const pointer = kernel._malloc(length * Float32Array.BYTES_PER_ELEMENT);
  if (pointer === 0) throw new Error(`failed to allocate ${length} float32 values`);
  return pointer;
}

function readFloat32(file) {
  const bytes = fs.readFileSync(file);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function deterministicAudio(channels, samples) {
  const output = new Float32Array(channels * samples);
  let state = 0x9e3779b9;
  for (let channel = 0; channel < channels; ++channel) {
    for (let index = 0; index < samples; ++index) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const noise = (state / 0xffffffff - 0.5) * 0.01;
      output[channel * samples + index] =
        Math.sin(index * (0.011 + channel * 0.004)) * 0.2 + noise;
    }
  }
  return output;
}

function transposeTimeMajorToPlanar(input, time, channels) {
  const output = new Float32Array(input.length);
  for (let channel = 0; channel < channels; ++channel) {
    for (let index = 0; index < time; ++index) {
      output[channel * time + index] = input[index * channels + channel];
    }
  }
  return output;
}

function compareIntegers(reference, candidate) {
  let differences = 0;
  let firstDifference = null;
  for (let index = 0; index < reference.length; ++index) {
    if (reference[index] !== candidate[index]) {
      differences += 1;
      firstDifference ??= {
        index,
        reference: Number(reference[index]),
        candidate: Number(candidate[index]),
      };
    }
  }
  return {
    exact: differences === 0 && reference.length === candidate.length,
    differences,
    total: reference.length,
    firstDifference,
  };
}

function compareFloat32(reference, candidate) {
  let signal = 0;
  let noise = 0;
  let maxAbsError = 0;
  for (let index = 0; index < reference.length; ++index) {
    const delta = candidate[index] - reference[index];
    signal += reference[index] * reference[index];
    noise += delta * delta;
    maxAbsError = Math.max(maxAbsError, Math.abs(delta));
  }
  return {
    snrDb: noise === 0 ? null : round(10 * Math.log10(signal / noise)),
    maxAbsError,
    maxAbsErrorDbfs: maxAbsError === 0 ? null : round(20 * Math.log10(maxAbsError)),
  };
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function round(value) {
  return Number(value.toFixed(3));
}
