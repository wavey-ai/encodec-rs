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
  process.argv[4] ?? path.join(root, "encodec-encoder.mjs"),
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
const createKernel = (await import(pathToFileURL(kernelModulePath))).default;
const kernel = await createKernel({
  locateFile: (file) => path.join(path.dirname(kernelModulePath), file),
});
const sessionOptions = {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
  enableCpuMemArena: false,
  enableMemPattern: false,
};

const layers = [];
for (const layer of metadata.convLayers) {
  layers.push(await benchmarkLayer(layer));
  process.stderr.write(`benchmarked ${layer.layer + 1}/${metadata.convLayers.length}\n`);
}

const report = {
  environment: {
    runtime: "Node-hosted WebAssembly",
    onnxRuntime: "onnxruntime-web wasm",
    threads: 1,
    node: process.version,
    kernelModule: kernelModulePath,
  },
  iterations,
  totals: {
    onnxMedianMs: round(layers.reduce((sum, layer) => sum + layer.onnxMedianMs, 0)),
    customMedianMs: round(layers.reduce((sum, layer) => sum + layer.customMedianMs, 0)),
    speedup: round(
      layers.reduce((sum, layer) => sum + layer.onnxMedianMs, 0) /
        layers.reduce((sum, layer) => sum + layer.customMedianMs, 0),
    ),
  },
  layers,
};
console.log(JSON.stringify(report, null, 2));

async function benchmarkLayer(layer) {
  const session = await ort.InferenceSession.create(
    fs.readFileSync(path.join(root, `conv-${layer.layer}.onnx`)),
    sessionOptions,
  );
  const inputLength = layer.paddedInputTime * layer.inputChannels;
  const outputLength = layer.outputTime * layer.outputChannels;
  const weights = readFloat32(path.join(root, `conv-${layer.layer}-weight.f32le`));
  const bias = readFloat32(path.join(root, `conv-${layer.layer}-bias.f32le`));
  const inputNhwc = deterministicInput(inputLength, layer.layer);
  const inputNchw = transposeNhwcToNchw(
    inputNhwc,
    layer.paddedInputTime,
    layer.inputChannels,
  );
  const inputTensor = new ort.Tensor(
    "float32",
    inputNchw,
    [1, layer.inputChannels, layer.paddedInputTime],
  );
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const pointers = {
    input: allocate(inputLength),
    weights: allocate(weights.length),
    packed: allocate(weights.length),
    bias: allocate(bias.length),
    output: allocate(outputLength),
  };
  kernel.HEAPF32.set(inputNhwc, pointers.input / 4);
  kernel.HEAPF32.set(weights, pointers.weights / 4);
  kernel.HEAPF32.set(bias, pointers.bias / 4);
  const packed = kernel._pack_conv1d_nhwc_weights_8(
    pointers.weights,
    pointers.packed,
    layer.inputChannels,
    layer.outputChannels,
    layer.kernel,
  );
  if (packed !== 1) throw new Error(`weight packing failed for layer ${layer.layer}`);

  const runOnnx = async () => {
    const started = performance.now();
    const outputs = await session.run({ [inputName]: inputTensor });
    return { elapsedMs: performance.now() - started, output: outputs[outputName].data };
  };
  const runCustom = () => {
    const started = performance.now();
    const ok = kernel._conv1d_nhwc_simd_8x8(
      pointers.input,
      pointers.packed,
      pointers.bias,
      pointers.output,
      layer.paddedInputTime,
      layer.inputChannels,
      layer.outputChannels,
      layer.kernel,
      layer.stride,
    );
    const elapsedMs = performance.now() - started;
    if (ok !== 1) throw new Error(`custom kernel rejected layer ${layer.layer}`);
    return elapsedMs;
  };

  let onnx = await runOnnx();
  runCustom();
  onnx = await runOnnx();
  runCustom();
  const onnxSamples = [];
  const customSamples = [];
  for (let index = 0; index < iterations; ++index) {
    if (index % 2 === 0) {
      onnx = await runOnnx();
      onnxSamples.push(onnx.elapsedMs);
      customSamples.push(runCustom());
    } else {
      customSamples.push(runCustom());
      onnx = await runOnnx();
      onnxSamples.push(onnx.elapsedMs);
    }
  }

  const outputOffset = pointers.output / 4;
  const customNhwc = kernel.HEAPF32.slice(outputOffset, outputOffset + outputLength);
  const customNchw = transposeNhwcToNchw(
    customNhwc,
    layer.outputTime,
    layer.outputChannels,
  );
  const parity = compareFloat32(onnx.output, customNchw);
  const onnxMedianMs = median(onnxSamples);
  const customMedianMs = median(customSamples);

  for (const pointer of Object.values(pointers)) kernel._free(pointer);
  await session.release();
  return {
    ...layer,
    onnxMedianMs: round(onnxMedianMs),
    customMedianMs: round(customMedianMs),
    speedup: round(onnxMedianMs / customMedianMs),
    onnxSamplesMs: onnxSamples.map(round),
    customSamplesMs: customSamples.map(round),
    parity,
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

function deterministicInput(length, salt) {
  const output = new Float32Array(length);
  let state = (0x9e3779b9 ^ salt) >>> 0;
  for (let index = 0; index < length; ++index) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    output[index] = (state / 0xffffffff - 0.5) * 0.5;
  }
  return output;
}

function transposeNhwcToNchw(input, time, channels) {
  const output = new Float32Array(input.length);
  for (let channel = 0; channel < channels; ++channel) {
    for (let index = 0; index < time; ++index) {
      output[channel * time + index] = input[index * channels + channel];
    }
  }
  return output;
}

function compareFloat32(reference, candidate) {
  let signal = 0;
  let noise = 0;
  let maxAbsError = 0;
  for (let index = 0; index < reference.length; ++index) {
    const expected = reference[index];
    const delta = candidate[index] - expected;
    signal += expected * expected;
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
