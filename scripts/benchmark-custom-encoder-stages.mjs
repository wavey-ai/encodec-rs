#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

const repoRoot = path.resolve(import.meta.dirname, "..");
const stageRoot = path.resolve(
  process.argv[2] ?? path.join(repoRoot, "target/performance/custom-encoder/stage-profile-1333"),
);
const runtimeRoot = path.resolve(
  process.argv[3] ?? path.join(repoRoot, "../vin.yl.vendor/wasm/onnxruntime-web"),
);
const iterations = Number(process.argv[4] ?? 9);
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
for (let index = 0; index < metadata.stages.length; ++index) {
  stageSessions.push(
    await ort.InferenceSession.create(
      fs.readFileSync(path.join(stageRoot, `stage-${index}.onnx`)),
      sessionOptions,
    ),
  );
}
const frontStageSessions = [];
for (let index = 0; index < metadata.frontStages.length; ++index) {
  frontStageSessions.push(
    await ort.InferenceSession.create(
      fs.readFileSync(path.join(stageRoot, `front-stage-${index}.onnx`)),
      sessionOptions,
    ),
  );
}
const setupMs = performance.now() - setupStarted;
const audio = deterministicAudio(metadata.channels, metadata.segmentSamples);
const audioTensor = new ort.Tensor(
  "float32",
  audio,
  [1, metadata.channels, metadata.segmentSamples],
);

const runFull = async () => {
  const started = performance.now();
  const outputs = await fullSession.run({ audio: audioTensor });
  return {
    elapsedMs: performance.now() - started,
    codes: outputs.codes,
    scale: outputs.scale,
  };
};

const runStages = async () => {
  const elapsedMs = [];
  let started = performance.now();
  const front = await stageSessions[0].run({ audio: audioTensor });
  elapsedMs.push(performance.now() - started);

  started = performance.now();
  const recurrent = await stageSessions[1].run({
    [metadata.stages[1].inputs[0]]: front[metadata.stages[0].outputs[0]],
  });
  elapsedMs.push(performance.now() - started);

  started = performance.now();
  const projection = await stageSessions[2].run({
    [metadata.stages[2].inputs[0]]: recurrent[metadata.stages[1].outputs[0]],
  });
  elapsedMs.push(performance.now() - started);

  started = performance.now();
  const quantized = await stageSessions[3].run({
    [metadata.stages[3].inputs[0]]: projection[metadata.stages[2].outputs[0]],
  });
  elapsedMs.push(performance.now() - started);
  return {
    elapsedMs,
    totalMs: elapsedMs.reduce((sum, value) => sum + value, 0),
    codes: quantized.codes,
    scale: front.scale,
    frontActivation: front[metadata.stages[0].outputs[0]],
  };
};

const runFrontStages = async () => {
  const elapsedMs = [];
  let activation = audioTensor;
  let scale = null;
  for (let index = 0; index < metadata.frontStages.length; ++index) {
    const stage = metadata.frontStages[index];
    const started = performance.now();
    const outputs = await frontStageSessions[index].run({
      [stage.inputs[0]]: activation,
    });
    elapsedMs.push(performance.now() - started);
    activation = outputs[stage.outputs[0]];
    if (index === 0) scale = outputs.scale;
  }
  return {
    elapsedMs,
    totalMs: elapsedMs.reduce((sum, value) => sum + value, 0),
    activation,
    scale,
  };
};

let reference = await runFull();
let staged = await runStages();
let frontStaged = await runFrontStages();
for (let index = 0; index < 2; ++index) {
  if (index % 2 === 0) {
    reference = await runFull();
    staged = await runStages();
    frontStaged = await runFrontStages();
  } else {
    frontStaged = await runFrontStages();
    staged = await runStages();
    reference = await runFull();
  }
}

const fullSamples = [];
const stagedSamples = [];
const stageSamples = [];
const frontStagedSamples = [];
const frontStageSamples = [];
for (let index = 0; index < iterations; ++index) {
  if (index % 3 === 0) {
    reference = await runFull();
    staged = await runStages();
    frontStaged = await runFrontStages();
  } else if (index % 3 === 1) {
    staged = await runStages();
    frontStaged = await runFrontStages();
    reference = await runFull();
  } else {
    frontStaged = await runFrontStages();
    reference = await runFull();
    staged = await runStages();
  }
  fullSamples.push(reference.elapsedMs);
  stagedSamples.push(staged.totalMs);
  stageSamples.push(staged.elapsedMs);
  frontStagedSamples.push(frontStaged.totalMs);
  frontStageSamples.push(frontStaged.elapsedMs);
}

assertExact(reference.codes.data, staged.codes.data, "encoder codes");
assertExact(reference.scale.data, staged.scale.data, "encoder scale");
assertExact(staged.frontActivation.data, frontStaged.activation.data, "front activation");
assertExact(staged.scale.data, frontStaged.scale.data, "front scale");
const fullMedian = median(fullSamples);
const stagedMedian = median(stagedSamples);
const stageMedians = medianColumns(stageSamples);
const frontStageMedians = medianColumns(frontStageSamples);
const report = {
  environment: {
    runtime: "Node-hosted WebAssembly",
    onnxRuntime: "onnxruntime-web wasm",
    threads: 1,
    node: process.version,
  },
  geometry: {
    channels: metadata.channels,
    segmentSamples: metadata.segmentSamples,
    frameLength: metadata.frameLength,
    codebooks: metadata.numCodebooks,
  },
  setupMs: round(setupMs),
  full: {
    medianMs: round(fullMedian),
    samplesMs: fullSamples.map(round),
    modelRealtimeFactor: round(
      (metadata.segmentSamples / metadata.sampleRate) / (fullMedian / 1000),
    ),
  },
  staged: {
    medianMs: round(stagedMedian),
    samplesMs: stagedSamples.map(round),
    medianStageMs: Object.fromEntries(
      metadata.stages.map((stage, index) => [stage.name, round(stageMedians[index])]),
    ),
  },
  frontStaged: {
    medianMs: round(median(frontStagedSamples)),
    samplesMs: frontStagedSamples.map(round),
    medianStageMs: Object.fromEntries(
      metadata.frontStages.map((stage, index) => [stage.name, round(frontStageMedians[index])]),
    ),
  },
  exact: {
    codes: true,
    scale: true,
    codesSha256: sha256(reference.codes.data),
    scaleSha256: sha256(reference.scale.data),
  },
};
console.log(JSON.stringify(report, null, 2));

for (const session of [...frontStageSessions, ...stageSessions, fullSession]) {
  await session.release();
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

function assertExact(left, right, label) {
  if (left.length !== right.length) {
    throw new Error(`${label} length differs: ${left.length} != ${right.length}`);
  }
  for (let index = 0; index < left.length; ++index) {
    if (left[index] !== right[index]) {
      throw new Error(`${label} differs at ${index}: ${left[index]} != ${right[index]}`);
    }
  }
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

function medianColumns(rows) {
  return rows[0].map((_, column) => median(rows.map((row) => row[column])));
}

function round(value) {
  return Number(value.toFixed(3));
}
