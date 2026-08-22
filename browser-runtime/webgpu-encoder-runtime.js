import {
  WebGpuResources,
  adapterDescription,
  createAddOperation,
  createConv1dOperation,
  createGroupNormOperation,
  createLstmLayerOperation,
  createNormalizeAudioOperation,
  createRvqEncodeOperation,
  createWebGpuKernels,
  fetchJson,
  loadTensorViews,
  normalizeAssetRoot,
  readFloat32Buffer,
  readUint32Buffer,
  recordWebGpuOperations,
  requestEncodecWebGpuDevice,
} from "./webgpu-kernel-runtime.js";

function encoderWeightNames(metadata) {
  const names = new Set(["rvq-embeddings.f32le", "rvq-norms.f32le"]);
  for (const layer of metadata.convLayers) {
    names.add(`conv-${layer.layer}-weight.f32le`);
    names.add(`conv-${layer.layer}-bias.f32le`);
    names.add(`conv-${layer.layer}-norm-scale.f32le`);
    names.add(`conv-${layer.layer}-norm-bias.f32le`);
  }
  for (const layer of metadata.lstmLayers) {
    names.add(`lstm-${layer.layer}-input-weight.f32le`);
    names.add(`lstm-${layer.layer}-recurrent-weight.f32le`);
    names.add(`lstm-${layer.layer}-bias.f32le`);
  }
  return [...names].sort();
}

function validateMetadata(metadata, bundle) {
  const checks = [
    ["sample rate", metadata.sampleRate, bundle.sample_rate],
    ["channels", metadata.channels, bundle.channels],
    ["segment samples", metadata.segmentSamples, bundle.segment_samples],
    ["segment stride", metadata.segmentStride, bundle.segment_stride],
    ["frame length", metadata.frameLength, bundle.frame_length],
    ["codebook count", metadata.numCodebooks, bundle.num_codebooks],
  ];
  for (const [name, actual, expected] of checks) {
    if (actual !== expected) {
      throw new Error(`WebGPU encoder ${name} mismatch: ${actual} != ${expected}`);
    }
  }
  if (
    metadata.convLayers?.length !== 18 ||
    metadata.lstmLayers?.length !== 2 ||
    metadata.rvq?.codebooks !== metadata.numCodebooks
  ) {
    throw new Error("WebGPU encoder metadata has an invalid model layout");
  }
}

export async function createWebGpuEncoder({
  assetRoot,
  bundleMetadata,
  fetchImpl = globalThis.fetch.bind(globalThis),
  gpuContext = null,
}) {
  const setupStarted = performance.now();
  const root = normalizeAssetRoot(assetRoot);
  const metadata = await fetchJson(new URL("metadata.json", root), fetchImpl);
  validateMetadata(metadata, bundleMetadata);
  const context = gpuContext || await requestEncodecWebGpuDevice();
  const ownsDevice = !gpuContext;
  const { adapter, device } = context;
  const resources = new WebGpuResources(device);
  let released = false;
  try {
    const [weights, kernels] = await Promise.all([
      loadTensorViews(root, encoderWeightNames(metadata), fetchImpl),
      createWebGpuKernels(device, resources),
    ]);
    const load = (name) => {
      const value = weights.get(name);
      if (!value) throw new Error(`WebGPU encoder weight is missing: ${name}`);
      return value;
    };

    const audioLength = metadata.channels * metadata.segmentSamples;
    const maxActivationLength = Math.max(
      ...metadata.convLayers.map((layer) =>
        layer.outputTime * Math.ceil(layer.outputChannels / 4) * 4),
    );
    const audio = resources.storage(audioLength, "encoder audio");
    const normalized = resources.storage(audioLength, "encoder normalized audio");
    const scale = resources.storage(1, "encoder scale");
    const activations = [0, 1, 2].map((index) =>
      resources.storage(maxActivationLength, `encoder activation ${index}`));
    const hiddenSize = metadata.lstmLayers[0].hiddenSize;
    const gateSize = 4 * hiddenSize;
    const gates = resources.storage(
      metadata.frameLength * gateSize,
      "encoder LSTM gates",
    );
    const hidden = resources.storage(hiddenSize, "encoder LSTM hidden state");
    const cell = resources.storage(hiddenSize, "encoder LSTM cell state");
    const residual = resources.storage(
      metadata.frameLength * metadata.rvq.dimension,
      "encoder RVQ residual",
    );
    const codes = resources.storageU32(
      metadata.frameLength * metadata.rvq.codebooks,
      "encoder RVQ codes",
    );
    const scaleReadback = resources.readback(4, "encoder scale readback");
    const codeBytes = metadata.frameLength * metadata.rvq.codebooks * 4;
    const codesReadback = resources.readback(codeBytes, "encoder codes readback");

    const operations = [];
    operations.push(createNormalizeAudioOperation(kernels, {
      input: audio,
      output: normalized,
      scale,
      time: metadata.segmentSamples,
      channels: metadata.channels,
    }));

    const createLayer = (layerIndex, input, output, applyElu = false) => {
      const layer = metadata.convLayers[layerIndex];
      const label = `encoder convolution ${layerIndex}`;
      const convolution = createConv1dOperation(kernels, {
        input,
        output,
        weights: load(`conv-${layerIndex}-weight.f32le`),
        bias: load(`conv-${layerIndex}-bias.f32le`),
        inputTime: layer.inputTime,
        inputChannels: layer.inputChannels,
        outputChannels: layer.outputChannels,
        kernel: layer.kernel,
        stride: layer.stride,
        paddingLeft: layer.paddingLeft,
        paddingRight: layer.paddingRight,
        applyElu,
        label,
      });
      const normalize = createGroupNormOperation(kernels, {
        values: output,
        scale: load(`conv-${layerIndex}-norm-scale.f32le`),
        bias: load(`conv-${layerIndex}-norm-bias.f32le`),
        time: layer.outputTime,
        channels: layer.outputChannels,
        storageChannels: convolution.storageChannels,
        label: `${label} normalization`,
      });
      operations.push((pass) => convolution.record(pass), normalize);
    };

    let current = activations[0];
    createLayer(0, normalized, current);
    const unused = (...used) => activations.find((buffer) => !used.includes(buffer));
    for (const [shortcutLayer, reduceLayer, expandLayer, downsampleLayer] of [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ]) {
      const shortcut = unused(current);
      const main = unused(current, shortcut);
      createLayer(shortcutLayer, current, shortcut);
      createLayer(reduceLayer, current, main, true);
      createLayer(expandLayer, main, current, true);
      const expanded = metadata.convLayers[expandLayer];
      operations.push(createAddOperation(kernels, {
        destination: current,
        source: shortcut,
        length: expanded.outputTime * expanded.outputChannels,
        label: `encoder residual ${expandLayer}`,
      }));
      createLayer(downsampleLayer, current, shortcut, true);
      current = shortcut;
    }

    const firstLstmOutput = unused(current);
    const secondLstmOutput = unused(current, firstLstmOutput);
    const lstm0 = metadata.lstmLayers[0];
    const lstm1 = metadata.lstmLayers[1];
    operations.push(createLstmLayerOperation(kernels, {
      input: current,
      output: firstLstmOutput,
      gates,
      hidden,
      cell,
      inputWeights: load("lstm-0-input-weight.f32le"),
      recurrentWeights: load("lstm-0-recurrent-weight.f32le"),
      bias: load("lstm-0-bias.f32le"),
      sequenceLength: metadata.frameLength,
      hiddenSize: lstm0.hiddenSize,
      label: "encoder LSTM 0",
    }));
    operations.push(createLstmLayerOperation(kernels, {
      input: firstLstmOutput,
      output: secondLstmOutput,
      gates,
      hidden,
      cell,
      inputWeights: load("lstm-1-input-weight.f32le"),
      recurrentWeights: load("lstm-1-recurrent-weight.f32le"),
      bias: load("lstm-1-bias.f32le"),
      sequenceLength: metadata.frameLength,
      hiddenSize: lstm1.hiddenSize,
      label: "encoder LSTM 1",
    }));
    operations.push(createAddOperation(kernels, {
      destination: secondLstmOutput,
      source: current,
      length: metadata.frameLength * hiddenSize,
      label: "encoder LSTM residual",
    }));
    createLayer(17, secondLstmOutput, firstLstmOutput, true);
    operations.push(createRvqEncodeOperation(kernels, {
      input: firstLstmOutput,
      residual,
      embeddings: load("rvq-embeddings.f32le"),
      norms: load("rvq-norms.f32le"),
      codes,
      time: metadata.frameLength,
      dimension: metadata.rvq.dimension,
      entries: metadata.rvq.entries,
      codebooks: metadata.rvq.codebooks,
    }));
    weights.clear();

    const setupMs = performance.now() - setupStarted;
    return {
      metadata,
      adapter: adapterDescription(adapter),
      setupMs,
      async encode(inputAudio) {
        if (released) throw new Error("WebGPU encoder is released");
        if (!(inputAudio instanceof Float32Array) || inputAudio.length !== audioLength) {
          throw new Error(
            `WebGPU encoder input must contain ${audioLength} float32 values`,
          );
        }
        device.queue.writeBuffer(audio, 0, inputAudio);
        const encoder = device.createCommandEncoder({ label: "EnCodec encode frame" });
        recordWebGpuOperations(encoder, operations, "EnCodec encoder");
        encoder.copyBufferToBuffer(scale, 0, scaleReadback, 0, 4);
        encoder.copyBufferToBuffer(codes, 0, codesReadback, 0, codeBytes);
        const started = performance.now();
        device.queue.submit([encoder.finish()]);
        const [scaleValues, codeValues] = await Promise.all([
          readFloat32Buffer(scaleReadback, 4),
          readUint32Buffer(codesReadback, codeBytes),
        ]);
        return {
          scale: scaleValues[0],
          codes: new Int32Array(codeValues),
          elapsedMs: performance.now() - started,
        };
      },
      release() {
        if (released) return;
        released = true;
        resources.destroy();
        if (ownsDevice) device.destroy();
      },
    };
  } catch (error) {
    resources.destroy();
    if (ownsDevice) device.destroy();
    throw error;
  }
}
