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
const iterations = Number(process.argv[5] ?? 9);
if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error(`iterations must be a positive integer, got ${iterations}`);
}

const metadata = JSON.parse(fs.readFileSync(path.join(root, "metadata.json"), "utf8"));
const { codebooks, entries, dimension } = metadata.rvq;
const time = metadata.frameLength;
const planarInput = deterministicInput(time, dimension);
const timeMajorInput = transposePlanarToTimeMajor(planarInput, time, dimension);

const ort = await import(pathToFileURL(path.join(runtimeRoot, "ort.wasm.min.mjs")));
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = `${pathToFileURL(runtimeRoot).href}/`;
const session = await ort.InferenceSession.create(
  fs.readFileSync(path.join(root, "stage-3.onnx")),
  {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    enableCpuMemArena: false,
    enableMemPattern: false,
  },
);
const inputTensor = new ort.Tensor("float32", planarInput, [1, dimension, time]);
const runReference = async () => {
  const started = performance.now();
  const outputs = await session.run({
    [metadata.stages[3].inputs[0]]: inputTensor,
  });
  return { elapsedMs: performance.now() - started, codes: outputs.codes.data };
};

const createKernel = (await import(pathToFileURL(kernelModulePath))).default;
const kernel = await createKernel({
  locateFile: (file) => path.join(path.dirname(kernelModulePath), file),
});
const embeddings = readFloat32(path.join(root, "rvq-embeddings.f32le"));
const norms = readFloat32(path.join(root, "rvq-norms.f32le"));
const pointers = {
  input: allocate(timeMajorInput.length),
  residual: allocate(timeMajorInput.length),
  embeddings: allocate(embeddings.length),
  norms: allocate(norms.length),
  codes: allocate(codebooks * time),
};
kernel.HEAPF32.set(timeMajorInput, pointers.input / 4);
kernel.HEAPF32.set(embeddings, pointers.embeddings / 4);
kernel.HEAPF32.set(norms, pointers.norms / 4);
const runCandidate = () => {
  const started = performance.now();
  const ok = kernel._rvq_encode_simd_8(
    pointers.input,
    pointers.residual,
    pointers.embeddings,
    pointers.norms,
    pointers.codes,
    time,
    dimension,
    entries,
    codebooks,
  );
  const elapsedMs = performance.now() - started;
  if (ok !== 1) throw new Error("custom RVQ rejected its geometry");
  return {
    elapsedMs,
    codes: new Int32Array(kernel.HEAPF32.buffer).slice(
      pointers.codes / 4,
      pointers.codes / 4 + codebooks * time,
    ),
  };
};

let reference = await runReference();
let candidate = runCandidate();
reference = await runReference();
candidate = runCandidate();
const referenceSamples = [];
const candidateSamples = [];
for (let index = 0; index < iterations; ++index) {
  if (index % 2 === 0) {
    reference = await runReference();
    candidate = runCandidate();
  } else {
    candidate = runCandidate();
    reference = await runReference();
  }
  referenceSamples.push(reference.elapsedMs);
  candidateSamples.push(candidate.elapsedMs);
}

const referenceMedian = median(referenceSamples);
const candidateMedian = median(candidateSamples);
console.log(
  JSON.stringify(
    {
      environment: {
        runtime: "Node-hosted WebAssembly",
        onnxRuntime: "onnxruntime-web wasm",
        threads: 1,
        node: process.version,
        kernelModule: kernelModulePath,
      },
      geometry: { time, dimension, entries, codebooks },
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
      parity: compareIntegers(reference.codes, candidate.codes),
    },
    null,
    2,
  ),
);

for (const pointer of Object.values(pointers)) kernel._free(pointer);
await session.release();

function allocate(length) {
  const pointer = kernel._malloc(length * Float32Array.BYTES_PER_ELEMENT);
  if (pointer === 0) throw new Error(`failed to allocate ${length} values`);
  return pointer;
}

function readFloat32(file) {
  const bytes = fs.readFileSync(file);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function deterministicInput(time, channels) {
  const output = new Float32Array(time * channels);
  let state = 0x7f4a7c15;
  for (let channel = 0; channel < channels; ++channel) {
    for (let index = 0; index < time; ++index) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      output[channel * time + index] =
        Math.sin(index * 0.071 + channel * 0.019) * 0.7 +
        (state / 0xffffffff - 0.5) * 0.03;
    }
  }
  return output;
}

function transposePlanarToTimeMajor(input, time, channels) {
  const output = new Float32Array(input.length);
  for (let index = 0; index < time; ++index) {
    for (let channel = 0; channel < channels; ++channel) {
      output[index * channels + channel] = input[channel * time + index];
    }
  }
  return output;
}

function compareIntegers(reference, candidate) {
  let differences = 0;
  let firstDifference = null;
  for (let index = 0; index < reference.length; ++index) {
    if (Number(reference[index]) !== candidate[index]) {
      differences += 1;
      firstDifference ??= {
        index,
        reference: Number(reference[index]),
        candidate: candidate[index],
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
