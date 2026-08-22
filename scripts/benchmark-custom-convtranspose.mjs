#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { performance } from "node:perf_hooks";

const repoRoot = path.resolve(import.meta.dirname, "..");
const fixtureRoot = path.resolve(
  process.argv[2] ?? path.join(repoRoot, "target/performance/custom-kernel/convtranspose-first"),
);
const runtimeRoot = path.resolve(
  process.argv[3] ?? path.join(repoRoot, "../vin.yl.vendor/wasm/onnxruntime-web"),
);
const kernelModulePath = path.resolve(
  process.argv[4] ?? path.join(fixtureRoot, "encodec-convtranspose.mjs"),
);
const iterations = Number(process.argv[5] ?? 9);
const selectedKernel = process.argv[6] ?? "all";

if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error(`iterations must be a positive integer, got ${iterations}`);
}

const metadata = JSON.parse(fs.readFileSync(path.join(fixtureRoot, "metadata.json"), "utf8"));
const inputLength = metadata.time * metadata.inputChannels;
const outputLength = metadata.outputTime * metadata.outputChannels;
const weightLength = metadata.inputChannels * metadata.outputChannels * metadata.kernel;
const input = deterministicInput(inputLength);
const weights = readFloat32(path.join(fixtureRoot, "weight.f32le"));
const bias = readFloat32(path.join(fixtureRoot, "bias.f32le"));

if (weights.length !== weightLength || bias.length !== metadata.outputChannels) {
  throw new Error("fixture weights do not match metadata");
}

const ort = await import(pathToFileURL(path.join(runtimeRoot, "ort.wasm.min.mjs")));
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = `${pathToFileURL(runtimeRoot).href}/`;

const session = await ort.InferenceSession.create(
  fs.readFileSync(path.join(fixtureRoot, "first-convtranspose.onnx")),
  {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    enableCpuMemArena: false,
    enableMemPattern: false,
  },
);
const feeds = {
  [session.inputNames[0]]: new ort.Tensor("float32", input, [
    1,
    metadata.inputChannels,
    metadata.time,
  ]),
};

const runOrt = async () => {
  const outputs = await session.run(feeds);
  return outputs[session.outputNames[0]].data;
};

const createKernelModule = (await import(pathToFileURL(kernelModulePath))).default;
const kernel = await createKernelModule({
  locateFile: (file) => path.join(path.dirname(kernelModulePath), file),
});
const pointers = allocateKernelBuffers(kernel, {
  inputLength,
  outputLength,
  weightLength,
  biasLength: bias.length,
});
kernel.HEAPF32.set(input, pointers.input / Float32Array.BYTES_PER_ELEMENT);
kernel.HEAPF32.set(weights, pointers.weights / Float32Array.BYTES_PER_ELEMENT);
kernel.HEAPF32.set(bias, pointers.bias / Float32Array.BYTES_PER_ELEMENT);

const packed = kernel._pack_conv_transpose1d_weights(
  pointers.weights,
  pointers.packed,
  metadata.inputChannels,
  metadata.outputChannels,
  metadata.stride,
);
if (packed !== 1) throw new Error("weight packing failed");

let candidateDefinitions = [
  ["phase-scalar", kernel._conv_transpose1d_phase_scalar],
  ["phase-simd", kernel._conv_transpose1d_phase_simd],
  ["phase-simd-8x4", kernel._conv_transpose1d_phase_simd_8x4],
  ["phase-simd-8x8", kernel._conv_transpose1d_phase_simd_8x8],
];
if (selectedKernel !== "all") {
  candidateDefinitions = candidateDefinitions.filter(([name]) => name === selectedKernel);
  if (candidateDefinitions.length === 0) {
    throw new Error(`unknown kernel ${selectedKernel}`);
  }
}

const candidateRuns = candidateDefinitions.map(([name, entrypoint]) => {
  const run = () => {
    const ok = entrypoint(
      pointers.input,
      pointers.packed,
      pointers.bias,
      pointers.output,
      metadata.time,
      metadata.inputChannels,
      metadata.outputChannels,
      metadata.stride,
    );
    if (ok !== 1) throw new Error(`${name} kernel rejected the fixture`);
  };
  return { name, run };
});

let reference = await runOrt();
for (let index = 0; index < 3; ++index) {
  if (index % 2 === 0) {
    reference = await runOrt();
    for (const candidate of candidateRuns) candidate.run();
  } else {
    for (const candidate of [...candidateRuns].reverse()) candidate.run();
    reference = await runOrt();
  }
}

const samplesByName = new Map([
  ["onnx-runtime", []],
  ...candidateRuns.map(({ name }) => [name, []]),
]);
const timedEntries = [
  {
    name: "onnx-runtime",
    run: async () => {
      reference = await runOrt();
    },
  },
  ...candidateRuns,
];
for (let index = 0; index < iterations; ++index) {
  const order = index % 2 === 0 ? timedEntries : [...timedEntries].reverse();
  for (const entry of order) {
    const started = performance.now();
    await entry.run();
    samplesByName.get(entry.name).push(performance.now() - started);
  }
}

const ortSamples = samplesByName.get("onnx-runtime");
const candidates = [];
for (const candidate of candidateRuns) {
  candidate.run();
  const samples = samplesByName.get(candidate.name);
  const outputOffset = pointers.output / Float32Array.BYTES_PER_ELEMENT;
  const output = kernel.HEAPF32.slice(outputOffset, outputOffset + outputLength);
  candidates.push({
    name: candidate.name,
    medianMs: round(median(samples)),
    samplesMs: samples.map(round),
    speedupVsOrt: round(median(ortSamples) / median(samples)),
    parity: compareFloat32(reference, output),
  });
}

for (const pointer of Object.values(pointers)) kernel._free(pointer);

console.log(
  JSON.stringify(
    {
      environment: {
        runtime: "onnxruntime-web wasm",
        threads: 1,
        node: process.version,
      },
      geometry: metadata,
      onnxRuntime: {
        medianMs: round(median(ortSamples)),
        samplesMs: ortSamples.map(round),
        outputSha256: sha256(reference),
      },
      candidates,
    },
    null,
    2,
  ),
);

function allocateKernelBuffers(module, lengths) {
  const allocate = (length) => {
    const pointer = module._malloc(length * Float32Array.BYTES_PER_ELEMENT);
    if (pointer === 0) throw new Error(`failed to allocate ${length} float32 values`);
    return pointer;
  };
  return {
    input: allocate(lengths.inputLength),
    weights: allocate(lengths.weightLength),
    packed: allocate(lengths.weightLength),
    bias: allocate(lengths.biasLength),
    output: allocate(lengths.outputLength),
  };
}

function deterministicInput(length) {
  const values = new Float32Array(length);
  let state = 0x9e3779b9;
  for (let index = 0; index < length; ++index) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values[index] = ((state >>> 8) / 0x1000000) * 2 - 1;
  }
  return values;
}

function readFloat32(file) {
  const bytes = fs.readFileSync(file);
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(copy);
}

function compareFloat32(referenceValues, candidateValues) {
  if (referenceValues.length !== candidateValues.length) {
    throw new Error(`output length mismatch: ${referenceValues.length} != ${candidateValues.length}`);
  }
  const referenceBits = new Uint32Array(
    referenceValues.buffer,
    referenceValues.byteOffset,
    referenceValues.length,
  );
  const candidateBits = new Uint32Array(
    candidateValues.buffer,
    candidateValues.byteOffset,
    candidateValues.length,
  );
  let exactMismatches = 0;
  let maxAbsError = 0;
  let squaredError = 0;
  let squaredSignal = 0;
  for (let index = 0; index < referenceValues.length; ++index) {
    if (referenceBits[index] !== candidateBits[index]) exactMismatches += 1;
    const error = referenceValues[index] - candidateValues[index];
    maxAbsError = Math.max(maxAbsError, Math.abs(error));
    squaredError += error * error;
    squaredSignal += referenceValues[index] * referenceValues[index];
  }
  return {
    exact: exactMismatches === 0,
    exactMismatches,
    maxAbsError,
    rmse: Math.sqrt(squaredError / referenceValues.length),
    snrDb: squaredError === 0 ? "Infinity" : 10 * Math.log10(squaredSignal / squaredError),
    outputSha256: sha256(candidateValues),
  };
}

function sha256(values) {
  return crypto
    .createHash("sha256")
    .update(Buffer.from(values.buffer, values.byteOffset, values.byteLength))
    .digest("hex");
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function round(value) {
  return Math.round(value * 1000) / 1000;
}
