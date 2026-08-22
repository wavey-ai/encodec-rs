import {
  WebGpuResources,
  adapterDescription,
  createAddOperation,
  createConv1dOperation,
  createConvTransposeOperation,
  createCropOperation,
  createGroupNormOperation,
  createLstmLayerOperation,
  createRvqDecodeOperation,
  createScaleToPlanarOperation,
  createWebGpuKernels,
  fetchJson,
  loadTensorViews,
  normalizeAssetRoot,
  readFloat32Buffer,
  recordWebGpuOperations,
  requestEncodecWebGpuDevice,
} from "./webgpu-kernel-runtime.js";

function decoderWeightNames(metadata) {
  const names = new Set([
    "front-rvq-embeddings.f32le",
    "front-conv-weight.f32le",
    "front-conv-bias.f32le",
    "front-conv-norm-scale.f32le",
    "front-conv-norm-bias.f32le",
    "final-conv-weight.f32le",
    "final-conv-bias.f32le",
    "final-conv-norm-scale.f32le",
    "final-conv-norm-bias.f32le",
  ]);
  for (const layer of metadata.front.lstmLayers) {
    names.add(`front-lstm-${layer.layer}-input-weight.f32le`);
    names.add(`front-lstm-${layer.layer}-recurrent-weight.f32le`);
    names.add(`front-lstm-${layer.layer}-bias.f32le`);
  }
  for (const layer of metadata.layers) {
    names.add(`layer-${layer.layer}-weight.f32le`);
    names.add(`layer-${layer.layer}-bias.f32le`);
    names.add(`layer-${layer.layer}-norm-scale.f32le`);
    names.add(`layer-${layer.layer}-norm-bias.f32le`);
  }
  for (const layer of metadata.post.convLayers) {
    names.add(`post-conv-${layer.layer}-weight.f32le`);
    names.add(`post-conv-${layer.layer}-bias.f32le`);
    names.add(`post-conv-${layer.layer}-norm-scale.f32le`);
    names.add(`post-conv-${layer.layer}-norm-bias.f32le`);
  }
  return [...names].sort();
}

function validateMetadata(metadata, bundle) {
  const checks = [
    ["frame length", metadata.frameLength, bundle.frame_length],
    ["codebook count", metadata.numCodebooks, bundle.num_codebooks],
    ["channel count", metadata.channels, bundle.channels],
    ["segment samples", metadata.segmentSamples, bundle.segment_samples],
  ];
  for (const [name, actual, expected] of checks) {
    if (actual !== expected) {
      throw new Error(`WebGPU decoder ${name} mismatch: ${actual} != ${expected}`);
    }
  }
  if (
    metadata.onnxFree !== true ||
    metadata.front?.lstmLayers?.length !== 2 ||
    metadata.layers?.length !== 4 ||
    metadata.post?.convLayers?.length !== 12
  ) {
    throw new Error("WebGPU decoder metadata has an invalid model layout");
  }
}

export async function createWebGpuDecoder({
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
      loadTensorViews(root, decoderWeightNames(metadata), fetchImpl),
      createWebGpuKernels(device, resources),
    ]);
    const load = (name) => {
      const value = weights.get(name);
      if (!value) throw new Error(`WebGPU decoder weight is missing: ${name}`);
      return value;
    };

    const front = metadata.front;
    const hiddenSize = front.lstmLayers[0].hiddenSize;
    const frontActivationLength = metadata.frameLength * hiddenSize;
    const frontActivations = [0, 1, 2].map((index) =>
      resources.storage(frontActivationLength, `decoder front activation ${index}`));
    const codes = resources.storageU32(
      metadata.frameLength * metadata.numCodebooks,
      "decoder codes",
    );
    const gates = resources.storage(
      metadata.frameLength * 4 * hiddenSize,
      "decoder LSTM gates",
    );
    const hidden = resources.storage(hiddenSize, "decoder LSTM hidden state");
    const cell = resources.storage(hiddenSize, "decoder LSTM cell state");

    const maxPostLength = Math.max(
      ...metadata.layers.map((layer) =>
        layer.rawOutputTime * Math.ceil(layer.outputChannels / 4) * 4),
      ...metadata.layers.map((layer) =>
        layer.croppedOutputTime * Math.ceil(layer.outputChannels / 4) * 4),
      ...metadata.post.convLayers.map((layer) =>
        layer.outputTime * Math.ceil(layer.outputChannels / 4) * 4),
    );
    const maxReducedLength = Math.max(
      ...metadata.post.convLayers.map((layer) =>
        layer.outputTime * Math.ceil(layer.outputChannels / 4) * 4),
    );
    const transposed = resources.storage(maxPostLength, "decoder transposed activation");
    const cropped = resources.storage(maxPostLength, "decoder cropped activation");
    const shortcut = resources.storage(maxPostLength, "decoder shortcut activation");
    const expanded = resources.storage(maxPostLength, "decoder expanded activation");
    const reduced = resources.storage(maxReducedLength, "decoder reduced activation");
    const final = metadata.post.finalConv;
    const finalStorageChannels = Math.ceil(final.outputChannels / 4) * 4;
    const finalActivation = resources.storage(
      final.outputTime * finalStorageChannels,
      "decoder final activation",
    );
    const planarLength = metadata.channels * metadata.segmentSamples;
    const planar = resources.storage(planarLength, "decoder planar output");
    const planarBytes = planarLength * Float32Array.BYTES_PER_ELEMENT;
    const readback = resources.readback(planarBytes, "decoder output readback");

    const operations = [];
    operations.push(createRvqDecodeOperation(kernels, {
      codes,
      embeddings: load("front-rvq-embeddings.f32le"),
      output: frontActivations[0],
      time: metadata.frameLength,
      dimension: front.rvq.dimension,
      entries: front.rvq.entries,
      codebooks: front.rvq.codebooks,
    }));

    const frontConv = createConv1dOperation(kernels, {
      input: frontActivations[0],
      output: frontActivations[1],
      weights: load("front-conv-weight.f32le"),
      bias: load("front-conv-bias.f32le"),
      inputTime: front.conv.inputTime,
      inputChannels: front.conv.inputChannels,
      outputChannels: front.conv.outputChannels,
      kernel: front.conv.kernel,
      stride: front.conv.stride,
      paddingLeft: front.conv.paddingLeft,
      paddingRight: front.conv.paddingRight,
      label: "decoder front convolution",
    });
    operations.push(
      (pass) => frontConv.record(pass),
      createGroupNormOperation(kernels, {
        values: frontActivations[1],
        scale: load("front-conv-norm-scale.f32le"),
        bias: load("front-conv-norm-bias.f32le"),
        time: front.conv.outputTime,
        channels: front.conv.outputChannels,
        label: "decoder front normalization",
      }),
    );
    operations.push(createLstmLayerOperation(kernels, {
      input: frontActivations[1],
      output: frontActivations[0],
      gates,
      hidden,
      cell,
      inputWeights: load("front-lstm-0-input-weight.f32le"),
      recurrentWeights: load("front-lstm-0-recurrent-weight.f32le"),
      bias: load("front-lstm-0-bias.f32le"),
      sequenceLength: metadata.frameLength,
      hiddenSize,
      label: "decoder LSTM 0",
    }));
    operations.push(createLstmLayerOperation(kernels, {
      input: frontActivations[0],
      output: frontActivations[2],
      gates,
      hidden,
      cell,
      inputWeights: load("front-lstm-1-input-weight.f32le"),
      recurrentWeights: load("front-lstm-1-recurrent-weight.f32le"),
      bias: load("front-lstm-1-bias.f32le"),
      sequenceLength: metadata.frameLength,
      hiddenSize,
      label: "decoder LSTM 1",
    }));
    operations.push(createAddOperation(kernels, {
      destination: frontActivations[2],
      source: frontActivations[1],
      length: metadata.frameLength * hiddenSize,
      applyElu: true,
      label: "decoder front residual",
    }));

    const postLayers = metadata.post.convLayers;
    const createPostConv = (layerIndex, input, output, applyElu) => {
      const layer = postLayers[layerIndex];
      const label = `decoder residual convolution ${layerIndex}`;
      const convolution = createConv1dOperation(kernels, {
        input,
        output,
        weights: load(`post-conv-${layerIndex}-weight.f32le`),
        bias: load(`post-conv-${layerIndex}-bias.f32le`),
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
      operations.push(
        (pass) => convolution.record(pass),
        createGroupNormOperation(kernels, {
          values: output,
          scale: load(`post-conv-${layerIndex}-norm-scale.f32le`),
          bias: load(`post-conv-${layerIndex}-norm-bias.f32le`),
          time: layer.outputTime,
          channels: layer.outputChannels,
          storageChannels: convolution.storageChannels,
          label: `${label} normalization`,
        }),
      );
    };

    let current = frontActivations[2];
    for (let layerIndex = 0; layerIndex < metadata.layers.length; layerIndex += 1) {
      const layer = metadata.layers[layerIndex];
      const label = `decoder transpose convolution ${layerIndex}`;
      const transpose = createConvTransposeOperation(kernels, {
        input: current,
        output: transposed,
        weights: load(`layer-${layerIndex}-weight.f32le`),
        bias: load(`layer-${layerIndex}-bias.f32le`),
        inputTime: layer.inputTime,
        inputChannels: layer.inputChannels,
        outputChannels: layer.outputChannels,
        stride: layer.stride,
        label,
      });
      operations.push(
        (pass) => transpose.record(pass),
        createGroupNormOperation(kernels, {
          values: transposed,
          scale: load(`layer-${layerIndex}-norm-scale.f32le`),
          bias: load(`layer-${layerIndex}-norm-bias.f32le`),
          time: layer.rawOutputTime,
          channels: layer.outputChannels,
          label: `${label} normalization`,
        }),
        createCropOperation(kernels, {
          input: transposed,
          output: cropped,
          rawTime: layer.rawOutputTime,
          channels: layer.outputChannels,
          cropLeft: layer.cropLeft,
          outputTime: layer.croppedOutputTime,
          label: `${label} crop`,
        }),
      );
      const [shortcutIndex, reduceIndex, expandIndex] = metadata.post.blocks[layerIndex];
      createPostConv(shortcutIndex, cropped, shortcut, false);
      createPostConv(reduceIndex, cropped, reduced, true);
      createPostConv(expandIndex, reduced, expanded, true);
      operations.push(createAddOperation(kernels, {
        destination: expanded,
        source: shortcut,
        length: layer.croppedOutputTime * layer.outputChannels,
        applyElu: true,
        label: `decoder residual ${layerIndex}`,
      }));
      current = expanded;
    }

    const finalConv = createConv1dOperation(kernels, {
      input: current,
      output: finalActivation,
      weights: load("final-conv-weight.f32le"),
      bias: load("final-conv-bias.f32le"),
      inputTime: final.inputTime,
      inputChannels: final.inputChannels,
      outputChannels: final.outputChannels,
      kernel: final.kernel,
      stride: final.stride,
      paddingLeft: final.paddingLeft,
      paddingRight: final.paddingRight,
      label: "decoder final convolution",
    });
    operations.push(
      (pass) => finalConv.record(pass),
      createGroupNormOperation(kernels, {
        values: finalActivation,
        scale: load("final-conv-norm-scale.f32le"),
        bias: load("final-conv-norm-bias.f32le"),
        time: final.outputTime,
        channels: final.outputChannels,
        storageChannels: finalConv.storageChannels,
        label: "decoder final normalization",
      }),
    );
    const outputOperation = createScaleToPlanarOperation(kernels, {
      input: finalActivation,
      output: planar,
      time: final.outputTime,
      channels: final.outputChannels,
      storageChannels: finalConv.storageChannels,
    });
    operations.push((pass) => outputOperation.record(pass));
    weights.clear();

    const codeUpload = new Uint32Array(metadata.frameLength * metadata.numCodebooks);
    const setupMs = performance.now() - setupStarted;

    async function decodeFrame(frame) {
      if (released) throw new Error("WebGPU decoder is released");
      if (frame.codes.length !== codeUpload.length) {
        throw new Error(
          `decoder frame has ${frame.codes.length} codes, expected ${codeUpload.length}`,
        );
      }
      for (let index = 0; index < codeUpload.length; index += 1) {
        codeUpload[index] = frame.codes[index];
      }
      device.queue.writeBuffer(codes, 0, codeUpload);
      outputOperation.setScale(Number(frame.scale ?? 1));
      const encoder = device.createCommandEncoder({ label: "EnCodec decode frame" });
      recordWebGpuOperations(encoder, operations, "EnCodec decoder");
      encoder.copyBufferToBuffer(planar, 0, readback, 0, planarBytes);
      const started = performance.now();
      device.queue.submit([encoder.finish()]);
      const audio = await readFloat32Buffer(readback, planarBytes);
      return {
        audio,
        elapsedMs: performance.now() - started,
        shape: [metadata.channels, metadata.segmentSamples],
      };
    }

    return {
      metadata,
      adapter: adapterDescription(adapter),
      setupMs,
      decodeFrame,
      async decode(frames, { onProgress = null } = {}) {
        if (released) throw new Error("WebGPU decoder is released");
        const audio = new Float32Array(frames.length * planarLength);
        const frameTimesMs = [];
        const startedAll = performance.now();
        for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
          const decoded = await decodeFrame(frames[frameIndex]);
          frameTimesMs.push(decoded.elapsedMs);
          audio.set(decoded.audio, frameIndex * planarLength);
          onProgress?.({
            completed: frameIndex + 1,
            total: frames.length,
            elapsedMs: decoded.elapsedMs,
          });
        }
        return {
          audio,
          elapsedMs: performance.now() - startedAll,
          frameTimesMs,
          shape: [frames.length, metadata.channels, metadata.segmentSamples],
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
