#!/usr/bin/env node

import crypto from "node:crypto";
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
const inputWavPath = process.argv[6] ? path.resolve(process.argv[6]) : null;
const inputChunkIndex = Number(process.argv[7] ?? 0);
if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error(`iterations must be a positive integer, got ${iterations}`);
}
if (!Number.isInteger(inputChunkIndex) || inputChunkIndex < 0) {
  throw new Error(`chunk index must be a nonnegative integer, got ${inputChunkIndex}`);
}

const metadata = JSON.parse(fs.readFileSync(path.join(root, "metadata.json"), "utf8"));
const frontLayers = metadata.convLayers.slice(0, 17);
const ort = await import(pathToFileURL(path.join(runtimeRoot, "ort.wasm.min.mjs")));
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = `${pathToFileURL(runtimeRoot).href}/`;
const createKernel = (await import(pathToFileURL(kernelModulePath))).default;
let kernel;
const sessionOptions = {
  executionProviders: ["wasm"],
  graphOptimizationLevel: "all",
  enableCpuMemArena: false,
  enableMemPattern: false,
};

const audio = inputWavPath
  ? loadWaveSegment(inputWavPath, inputChunkIndex)
  : deterministicAudio(metadata.channels, metadata.segmentSamples);
const audioTensor = new ort.Tensor(
  "float32",
  audio,
  [1, metadata.channels, metadata.segmentSamples],
);

let fullSession = await ort.InferenceSession.create(
  fs.readFileSync(metadata.sourceModel),
  sessionOptions,
);
process.stderr.write("loaded full encoder control\n");
const runFull = async () => {
  const started = performance.now();
  const outputs = await fullSession.run({ audio: audioTensor });
  return {
    elapsedMs: performance.now() - started,
    codes: outputs.codes,
    scale: outputs.scale,
  };
};

let control;
for (let index = 0; index < 2; ++index) control = await runFull();
const controlSamples = [];
for (let index = 0; index < iterations; ++index) {
  control = await runFull();
  controlSamples.push(control.elapsedMs);
}
const controlCodes = control.codes.data.slice();
const controlScale = control.scale.data.slice();
process.stderr.write("measured full encoder control\n");

let frontSession = await ort.InferenceSession.create(
  fs.readFileSync(path.join(root, "stage-0.onnx")),
  sessionOptions,
);
const runOnnxFront = async () => {
  const started = performance.now();
  const outputs = await frontSession.run({ audio: audioTensor });
  return {
    elapsedMs: performance.now() - started,
    activation: outputs[metadata.stages[0].outputs[0]],
    scale: outputs.scale,
  };
};

let referenceFront;
for (let index = 0; index < 2; ++index) referenceFront = await runOnnxFront();
const onnxFrontSamples = [];
for (let index = 0; index < iterations; ++index) {
  referenceFront = await runOnnxFront();
  onnxFrontSamples.push(referenceFront.elapsedMs);
}
const referenceFrontActivation = referenceFront.activation.data.slice();
const referenceFrontScale = referenceFront.scale.data.slice();
await frontSession.release();
frontSession = null;
process.stderr.write("measured and released ONNX front reference\n");

const referenceStageActivations = [];
let fineStageInput = audioTensor;
for (let index = 0; index < metadata.frontStages.length - 1; ++index) {
  const stage = metadata.frontStages[index];
  const session = await ort.InferenceSession.create(
    fs.readFileSync(path.join(root, `front-stage-${index}.onnx`)),
    sessionOptions,
  );
  const outputs = await session.run({ [stage.inputs[0]]: fineStageInput });
  const output = outputs[stage.outputs[0]];
  const copied = output.data.slice();
  referenceStageActivations.push(copied);
  fineStageInput = new ort.Tensor("float32", copied, output.dims);
  await session.release();
}

kernel = await createKernel({
  locateFile: (file) => path.join(path.dirname(kernelModulePath), file),
});
process.stderr.write("loaded custom kernel\n");
const encoder = prepareCustomEncoder(audio);
process.stderr.write("prepared custom encoder weights\n");
const customDebug = encoder.run(true);
const customOperatorProfile = encoder.run(false, true).operatorMs;
let complete;
for (let index = 0; index < 3; ++index) {
  if (index % 2 === 0) {
    control = await runFull();
    complete = encoder.runComplete();
  } else {
    complete = encoder.runComplete();
    control = await runFull();
  }
}
const pairedControlSamples = [];
const completeSamples = [];
const completeStageSamples = [];
for (let index = 0; index < iterations; ++index) {
  if (index % 2 === 0) {
    control = await runFull();
    complete = encoder.runComplete();
  } else {
    complete = encoder.runComplete();
    control = await runFull();
  }
  pairedControlSamples.push(control.elapsedMs);
  completeSamples.push(complete.elapsedMs);
  completeStageSamples.push(complete.stagesMs);
}
await fullSession.release();
fullSession = null;
process.stderr.write("measured paired complete encoder and released full control\n");
const stageSessions = [null];
for (let index = 1; index < metadata.stages.length; ++index) {
  stageSessions[index] = await ort.InferenceSession.create(
    fs.readFileSync(path.join(root, `stage-${index}.onnx`)),
    sessionOptions,
  );
  process.stderr.write(`loaded ONNX stage ${index + 1}/${metadata.stages.length}\n`);
}

const runHybrid = async () => {
  let started = performance.now();
  const front = encoder.run();
  const frontMs = performance.now() - started;

  const frontTensor = new ort.Tensor(
    "float32",
    front.activation,
    [front.time, 1, front.channels],
  );
  started = performance.now();
  const recurrent = await stageSessions[1].run({
    [metadata.stages[1].inputs[0]]: frontTensor,
  });
  const recurrentMs = performance.now() - started;

  started = performance.now();
  const projection = await stageSessions[2].run({
    [metadata.stages[2].inputs[0]]: recurrent[metadata.stages[1].outputs[0]],
  });
  const projectionMs = performance.now() - started;

  started = performance.now();
  const quantized = await stageSessions[3].run({
    [metadata.stages[3].inputs[0]]: projection[metadata.stages[2].outputs[0]],
  });
  const quantizerMs = performance.now() - started;
  return {
    elapsedMs: frontMs + recurrentMs + projectionMs + quantizerMs,
    stagesMs: [frontMs, recurrentMs, projectionMs, quantizerMs],
    codes: quantized.codes,
    scale: front.scale,
    frontActivation: front.activation,
  };
};

let hybrid;
for (let index = 0; index < 3; ++index) {
  process.stderr.write(`warm-up ${index + 1}/3\n`);
  hybrid = await runHybrid();
}

const hybridSamples = [];
const hybridStageSamples = [];
for (let index = 0; index < iterations; ++index) {
  hybrid = await runHybrid();
  hybridSamples.push(hybrid.elapsedMs);
  hybridStageSamples.push(hybrid.stagesMs);
}

const codeParity = compareIntegers(controlCodes, hybrid.codes.data);
const scaleParity = compareFloat32(referenceFrontScale, hybrid.scale);
const activationParity = compareFloat32(
  referenceFrontActivation,
  hybrid.frontActivation,
);
const controlMedian = median(pairedControlSamples);
const hybridMedian = median(hybridSamples);
const completeMedian = median(completeSamples);
const report = {
  environment: {
    runtime: "Node-hosted WebAssembly",
    onnxRuntime: "onnxruntime-web wasm",
    threads: 1,
    node: process.version,
    kernelModule: kernelModulePath,
    measurement: "interleaved full control/custom-complete; sequential hybrid phase",
  },
  geometry: {
    input: inputWavPath,
    chunkIndex: inputWavPath ? inputChunkIndex : null,
    channels: metadata.channels,
    segmentSamples: metadata.segmentSamples,
    frameLength: metadata.frameLength,
    codebooks: metadata.numCodebooks,
  },
  iterations,
  control: {
    medianMs: round(controlMedian),
    samplesMs: pairedControlSamples.map(round),
    modelRealtimeFactor: round(
      (metadata.segmentSamples / metadata.sampleRate) / (controlMedian / 1000),
    ),
  },
  onnxFront: {
    medianMs: round(median(onnxFrontSamples)),
    samplesMs: onnxFrontSamples.map(round),
  },
  complete: {
    medianMs: round(completeMedian),
    samplesMs: completeSamples.map(round),
    speedup: round(controlMedian / completeMedian),
    modelRealtimeFactor: round(
      (metadata.segmentSamples / metadata.sampleRate) / (completeMedian / 1000),
    ),
    medianStageMs: Object.fromEntries(
      metadata.stages.map((stage, index) => [
        stage.name,
        round(medianColumns(completeStageSamples)[index]),
      ]),
    ),
    parity: {
      codes: compareIntegers(controlCodes, complete.codes),
      scale: compareFloat32(controlScale, complete.scale),
    },
  },
  hybrid: {
    medianMs: round(hybridMedian),
    samplesMs: hybridSamples.map(round),
    speedup: round(controlMedian / hybridMedian),
    modelRealtimeFactor: round(
      (metadata.segmentSamples / metadata.sampleRate) / (hybridMedian / 1000),
    ),
    medianStageMs: Object.fromEntries(
      metadata.stages.map((stage, index) => [
        stage.name,
        round(medianColumns(hybridStageSamples)[index]),
      ]),
    ),
    customOperatorProfileMs: customOperatorProfile,
  },
  parity: {
    codes: codeParity,
    scale: scaleParity,
    frontActivation: activationParity,
    frontStages: metadata.frontStages.slice(0, -1).map((stage, index) => {
      const channels = [32, 32, 64, 64, 128, 128, 256, 256, 512][index];
      const time = referenceStageActivations[index].length / channels;
      return {
        name: stage.name,
        ...compareFloat32(
          referenceStageActivations[index],
          transposeNhwcToNchw(customDebug.stageActivations[index], time, channels),
        ),
      };
    }),
    referenceScaleMatchesFullControl: compareFloat32(controlScale, referenceFrontScale),
    referenceCodesSha256: sha256(controlCodes),
    hybridCodesSha256: sha256(hybrid.codes.data),
  },
};
console.log(JSON.stringify(report, null, 2));

encoder.release();
for (const session of stageSessions.slice(1)) await session.release();

function prepareCustomEncoder(sourceAudio) {
  const maxActivationLength = Math.max(
    ...metadata.convLayers.map((layer) => layer.outputTime * layer.outputChannels),
  );
  const maxPaddedLength = Math.max(
    ...metadata.convLayers.map((layer) => layer.paddedInputTime * layer.inputChannels),
  );
  const hiddenSize = metadata.lstmLayers[0].hiddenSize;
  const gateSize = 4 * hiddenSize;
  const embeddings = readFloat32(path.join(root, "rvq-embeddings.f32le"));
  const norms = readFloat32(path.join(root, "rvq-norms.f32le"));
  const pointers = {
    audio: allocate(sourceAudio.length),
    normalizedAudio: allocate(sourceAudio.length),
    padded: allocate(maxPaddedLength),
    activation0: allocate(maxActivationLength),
    activation1: allocate(maxActivationLength),
    activation2: allocate(maxActivationLength),
    lstmHidden: allocate(hiddenSize),
    lstmCell: allocate(hiddenSize),
    lstmInputProjection: allocate(metadata.frameLength * gateSize),
    rvqEmbeddings: allocate(embeddings.length),
    rvqNorms: allocate(norms.length),
    rvqResidual: allocate(metadata.frameLength * metadata.rvq.dimension),
    rvqCodes: allocate(metadata.rvq.codebooks * metadata.frameLength),
  };
  kernel.HEAPF32.set(sourceAudio, pointers.audio / 4);
  kernel.HEAPF32.set(embeddings, pointers.rvqEmbeddings / 4);
  kernel.HEAPF32.set(norms, pointers.rvqNorms / 4);

  const layers = metadata.convLayers.map((layer) => {
    const weights = readFloat32(path.join(root, `conv-${layer.layer}-weight.f32le`));
    const bias = readFloat32(path.join(root, `conv-${layer.layer}-bias.f32le`));
    const normScale = readFloat32(
      path.join(root, `conv-${layer.layer}-norm-scale.f32le`),
    );
    const normBias = readFloat32(
      path.join(root, `conv-${layer.layer}-norm-bias.f32le`),
    );
    const unpackedWeights = allocate(weights.length);
    const layerPointers = {
      packed: allocate(weights.length),
      bias: allocate(bias.length),
      normScale: allocate(normScale.length),
      normBias: allocate(normBias.length),
    };
    kernel.HEAPF32.set(weights, unpackedWeights / 4);
    kernel.HEAPF32.set(bias, layerPointers.bias / 4);
    kernel.HEAPF32.set(normScale, layerPointers.normScale / 4);
    kernel.HEAPF32.set(normBias, layerPointers.normBias / 4);
    const packed = kernel._pack_conv1d_nhwc_weights_8(
      unpackedWeights,
      layerPointers.packed,
      layer.inputChannels,
      layer.outputChannels,
      layer.kernel,
    );
    if (packed !== 1) throw new Error(`weight packing failed for layer ${layer.layer}`);
    kernel._free(unpackedWeights);
    return { ...layer, pointers: layerPointers };
  });

  const lstmLayers = metadata.lstmLayers.map((layer) => {
    const inputWeights = readFloat32(
      path.join(root, `lstm-${layer.layer}-input-weight.f32le`),
    );
    const recurrentWeights = readFloat32(
      path.join(root, `lstm-${layer.layer}-recurrent-weight.f32le`),
    );
    const bias = readFloat32(path.join(root, `lstm-${layer.layer}-bias.f32le`));
    const unpackedInput = allocate(inputWeights.length);
    const unpackedRecurrent = allocate(recurrentWeights.length);
    const layerPointers = {
      packedInput: allocate(inputWeights.length),
      packedRecurrent: allocate(recurrentWeights.length),
      bias: allocate(bias.length),
    };
    kernel.HEAPF32.set(inputWeights, unpackedInput / 4);
    kernel.HEAPF32.set(recurrentWeights, unpackedRecurrent / 4);
    kernel.HEAPF32.set(bias, layerPointers.bias / 4);
    const inputOk = kernel._pack_linear_weights_8(
      unpackedInput,
      layerPointers.packedInput,
      hiddenSize,
      gateSize,
    );
    const recurrentOk = kernel._pack_linear_weights_8(
      unpackedRecurrent,
      layerPointers.packedRecurrent,
      hiddenSize,
      gateSize,
    );
    kernel._free(unpackedInput);
    kernel._free(unpackedRecurrent);
    if (inputOk !== 1 || recurrentOk !== 1) {
      throw new Error(`weight packing failed for LSTM layer ${layer.layer}`);
    }
    return { ...layer, pointers: layerPointers };
  });

  const activationPointers = [
    pointers.activation0,
    pointers.activation1,
    pointers.activation2,
  ];
  let activeProfile = null;
  const timed = (name, operation) => {
    if (activeProfile === null) return operation();
    const started = performance.now();
    const result = operation();
    activeProfile[name] += performance.now() - started;
    return result;
  };
  const unused = (...used) => activationPointers.find((pointer) => !used.includes(pointer));
  const runLayer = (layerIndex, input, output, applyElu = false) => {
    const layer = layers[layerIndex];
    let convolutionInput = input;
    if (layer.paddingLeft !== 0 || layer.paddingRight !== 0) {
      const padded = timed("padding", () =>
        (applyElu ? kernel._reflect_pad_elu_nhwc : kernel._reflect_pad_nhwc)(
          input,
          pointers.padded,
          layer.inputTime,
          layer.inputChannels,
          layer.paddingLeft,
          layer.paddingRight,
        ),
      );
      if (padded !== 1) throw new Error(`padding failed for layer ${layerIndex}`);
      convolutionInput = pointers.padded;
    } else if (applyElu) {
      const length = layer.inputTime * layer.inputChannels;
      if (timed("elu", () => kernel._elu_nhwc_in_place(input, length)) !== 1) {
        throw new Error(`input ELU failed for layer ${layerIndex}`);
      }
    }
    const convolved = timed("convolution", () =>
      kernel._conv1d_nhwc_simd_8x8(
        convolutionInput,
        layer.pointers.packed,
        layer.pointers.bias,
        output,
        layer.paddedInputTime,
        layer.inputChannels,
        layer.outputChannels,
        layer.kernel,
        layer.stride,
      ),
    );
    if (convolved !== 1) throw new Error(`convolution failed for layer ${layerIndex}`);
    const normalized = timed("normalization", () =>
      kernel._group_norm_nhwc_in_place(
        output,
        layer.pointers.normScale,
        layer.pointers.normBias,
        layer.outputTime,
        layer.outputChannels,
      ),
    );
    if (normalized !== 1) throw new Error(`normalization failed for layer ${layerIndex}`);
  };

  const run = (captureStages = false, profile = false, copyOutput = true) => {
    const totalStarted = profile ? performance.now() : 0;
    activeProfile = profile
      ? {
          audioNormalization: 0,
          padding: 0,
          convolution: 0,
          normalization: 0,
          elu: 0,
          residualAdd: 0,
          outputCopy: 0,
          unaccounted: 0,
          total: 0,
        }
      : null;
    const trace = !run.traced;
    run.traced = true;
    const scale = timed("audioNormalization", () =>
      kernel._normalize_audio_planar_to_nhwc(
        pointers.audio,
        pointers.normalizedAudio,
        metadata.segmentSamples,
        metadata.channels,
      ),
    );
    if (trace) process.stderr.write("    normalized audio\n");
    let current = pointers.activation0;
    runLayer(0, pointers.normalizedAudio, current);
    if (trace) process.stderr.write("    initial convolution\n");
    const stageActivations = [];
    if (captureStages) {
      const layer = layers[0];
      stageActivations.push(
        kernel.HEAPF32.slice(
          current / 4,
          current / 4 + layer.outputTime * layer.outputChannels,
        ),
      );
    }
    for (const [shortcutLayer, firstMainLayer, secondMainLayer, downsampleLayer] of [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ]) {
      const shortcut = unused(current);
      const main = unused(current, shortcut);
      runLayer(shortcutLayer, current, shortcut);
      if (trace) process.stderr.write(`    shortcut ${shortcutLayer}\n`);
      const inputLength =
        layers[shortcutLayer].inputTime * layers[shortcutLayer].inputChannels;
      runLayer(firstMainLayer, current, main, true);
      if (trace) process.stderr.write(`    residual convolution ${firstMainLayer}\n`);
      const mainLength =
        layers[firstMainLayer].outputTime * layers[firstMainLayer].outputChannels;
      if (timed("elu", () => kernel._elu_nhwc_in_place(main, mainLength)) !== 1) {
        throw new Error(`main ELU failed for residual layer ${shortcutLayer}`);
      }
      runLayer(secondMainLayer, main, current);
      if (trace) process.stderr.write(`    residual convolution ${secondMainLayer}\n`);
      const residualLength =
        layers[secondMainLayer].outputTime * layers[secondMainLayer].outputChannels;
      if (
        timed("residualAdd", () =>
          kernel._add_nhwc_in_place(current, shortcut, residualLength),
        ) !== 1
      ) {
        throw new Error(`residual add failed for layer ${shortcutLayer}`);
      }
      if (captureStages) {
        stageActivations.push(
          kernel.HEAPF32.slice(current / 4, current / 4 + residualLength),
        );
      }
      runLayer(downsampleLayer, current, shortcut, true);
      if (trace) process.stderr.write(`    downsample ${downsampleLayer}\n`);
      current = shortcut;
      if (captureStages) {
        const layer = layers[downsampleLayer];
        stageActivations.push(
          kernel.HEAPF32.slice(
            current / 4,
            current / 4 + layer.outputTime * layer.outputChannels,
          ),
        );
      }
    }
    const finalLayer = layers[16];
    const activationLength = finalLayer.outputTime * finalLayer.outputChannels;
    const activation = copyOutput
      ? timed("outputCopy", () =>
          kernel.HEAPF32.slice(current / 4, current / 4 + activationLength),
        )
      : null;
    if (activeProfile !== null) {
      activeProfile.total = performance.now() - totalStarted;
      activeProfile.unaccounted =
        activeProfile.total -
        Object.entries(activeProfile)
          .filter(([name]) => name !== "total" && name !== "unaccounted")
          .reduce((sum, [, value]) => sum + value, 0);
      for (const name of Object.keys(activeProfile)) {
        activeProfile[name] = round(activeProfile[name]);
      }
    }
    const operatorMs = activeProfile;
    activeProfile = null;
    return {
      activation,
      pointer: current,
      time: finalLayer.outputTime,
      channels: finalLayer.outputChannels,
      scale: new Float32Array([scale]),
      stageActivations,
      operatorMs,
    };
  };

  const runComplete = () => {
    const totalStarted = performance.now();
    let started = performance.now();
    const front = run(false, false, false);
    const frontMs = performance.now() - started;
    const firstOutput = unused(front.pointer);
    const secondOutput = unused(front.pointer, firstOutput);

    const runLstmLayer = (input, layer, output) => {
      const ok = kernel._lstm_layer_simd_64(
        input,
        layer.pointers.packedInput,
        layer.pointers.packedRecurrent,
        layer.pointers.bias,
        output,
        pointers.lstmHidden,
        pointers.lstmCell,
        pointers.lstmInputProjection,
        metadata.frameLength,
        hiddenSize,
      );
      if (ok !== 1) throw new Error(`LSTM layer ${layer.layer} failed`);
    };
    started = performance.now();
    runLstmLayer(front.pointer, lstmLayers[0], firstOutput);
    runLstmLayer(firstOutput, lstmLayers[1], secondOutput);
    if (
      kernel._add_nhwc_in_place(
        secondOutput,
        front.pointer,
        metadata.frameLength * hiddenSize,
      ) !== 1
    ) {
      throw new Error("LSTM residual add failed");
    }
    const recurrentMs = performance.now() - started;

    started = performance.now();
    runLayer(17, secondOutput, firstOutput, true);
    const projectionMs = performance.now() - started;

    started = performance.now();
    const quantized = kernel._rvq_encode_simd_8(
      firstOutput,
      pointers.rvqResidual,
      pointers.rvqEmbeddings,
      pointers.rvqNorms,
      pointers.rvqCodes,
      metadata.frameLength,
      metadata.rvq.dimension,
      metadata.rvq.entries,
      metadata.rvq.codebooks,
    );
    if (quantized !== 1) throw new Error("RVQ failed");
    const quantizerMs = performance.now() - started;
    const codes = new Int32Array(kernel.HEAPF32.buffer).slice(
      pointers.rvqCodes / 4,
      pointers.rvqCodes / 4 + metadata.rvq.codebooks * metadata.frameLength,
    );
    return {
      elapsedMs: performance.now() - totalStarted,
      stagesMs: [frontMs, recurrentMs, projectionMs, quantizerMs],
      codes,
      scale: front.scale,
    };
  };

  return {
    run,
    runComplete,
    release() {
      for (const pointer of Object.values(pointers)) kernel._free(pointer);
      for (const layer of layers) {
        for (const pointer of Object.values(layer.pointers)) kernel._free(pointer);
      }
      for (const layer of lstmLayers) {
        for (const pointer of Object.values(layer.pointers)) kernel._free(pointer);
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

function loadWaveSegment(filename, chunkIndex) {
  const wave = decodeWave(fs.readFileSync(filename));
  if (wave.sampleRate !== metadata.sampleRate) {
    throw new Error(
      `WAV sample rate ${wave.sampleRate} does not match ${metadata.sampleRate}`,
    );
  }
  if (wave.channels !== metadata.channels) {
    throw new Error(
      `WAV channels ${wave.channels} does not match ${metadata.channels}`,
    );
  }
  const offset = chunkIndex * metadata.segmentStride;
  if (offset >= wave.frames) {
    throw new Error(`chunk ${chunkIndex} starts after the end of the WAV file`);
  }
  const context = (metadata.segmentSamples - metadata.segmentStride) / 2;
  if (!Number.isInteger(context) || context < 0) {
    throw new Error("custom encoder metadata has invalid fixed context");
  }
  const segment = new Float32Array(metadata.channels * metadata.segmentSamples);
  for (let channel = 0; channel < metadata.channels; ++channel) {
    const sourceBase = channel * wave.frames;
    const targetBase = channel * metadata.segmentSamples;
    for (let modelIndex = 0; modelIndex < metadata.segmentSamples; ++modelIndex) {
      const sourceIndex = offset - context + modelIndex;
      if (sourceIndex >= 0 && sourceIndex < wave.frames) {
        segment[targetBase + modelIndex] = wave.audio[sourceBase + sourceIndex];
      }
    }
  }
  return segment;
}

function decodeWave(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (ascii(view, 0, 4) !== "RIFF" || ascii(view, 8, 4) !== "WAVE") {
    throw new Error("Input is not a RIFF/WAVE file");
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
        subformat:
          tag === 0xfffe && size >= 40 ? view.getUint16(body + 24, true) : tag,
      };
    } else if (id === "data") {
      dataOffset = body;
      dataBytes = Math.min(size, view.byteLength - body);
    }
    offset = body + size + (size & 1);
  }
  if (!format || !dataOffset || !dataBytes) {
    throw new Error("WAV has no usable fmt/data chunks");
  }
  const bytesPerSample = format.bitsPerSample / 8;
  const frames = Math.floor(
    dataBytes / (format.channels * bytesPerSample),
  );
  const audio = new Float32Array(format.channels * frames);
  let cursor = dataOffset;
  for (let frame = 0; frame < frames; ++frame) {
    for (let channel = 0; channel < format.channels; ++channel) {
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
    throw new Error(`unsupported WAV format tag ${format.subformat}`);
  }
  if (format.bitsPerSample === 16) {
    return view.getInt16(offset, true) / 32_768;
  }
  if (format.bitsPerSample === 24) {
    const unsigned =
      view.getUint8(offset) |
      (view.getUint8(offset + 1) << 8) |
      (view.getUint8(offset + 2) << 16);
    return ((unsigned << 8) >> 8) / 8_388_608;
  }
  if (format.bitsPerSample === 32) {
    return view.getInt32(offset, true) / 2_147_483_648;
  }
  throw new Error(`unsupported PCM bit depth ${format.bitsPerSample}`);
}

function ascii(view, offset, length) {
  let value = "";
  for (let index = 0; index < length; ++index) {
    value += String.fromCharCode(view.getUint8(offset + index));
  }
  return value;
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

function transposeNhwcToNchw(input, time, channels) {
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
    if (Number(reference[index]) !== Number(candidate[index])) {
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
  if (reference.length !== candidate.length) {
    return { exact: false, referenceLength: reference.length, candidateLength: candidate.length };
  }
  let signal = 0;
  let noise = 0;
  let maxAbsError = 0;
  let exact = true;
  for (let index = 0; index < reference.length; ++index) {
    const expected = reference[index];
    const delta = candidate[index] - expected;
    exact &&= delta === 0;
    signal += expected * expected;
    noise += delta * delta;
    maxAbsError = Math.max(maxAbsError, Math.abs(delta));
  }
  return {
    exact,
    snrDb: noise === 0 ? null : round(10 * Math.log10(signal / noise)),
    maxAbsError,
    maxAbsErrorDbfs: maxAbsError === 0 ? null : round(20 * Math.log10(maxAbsError)),
  };
}

function sha256(values) {
  return crypto
    .createHash("sha256")
    .update(Buffer.from(values.buffer, values.byteOffset, values.byteLength))
    .digest("hex");
}

function medianColumns(rows) {
  return rows[0].map((_, column) => median(rows.map((row) => row[column])));
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
