function requireInteger(value, name) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function normalizeUrl(value) {
  const url = value instanceof URL ? value : new URL(String(value), globalThis.location?.href);
  return new URL(url.href.replace(/\/?$/, "/"));
}

async function fetchJson(url, fetchImpl) {
  const response = await fetchImpl(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`fetch ${url} failed: ${response.status}`);
  }
  return response.json();
}

async function fetchFloat32(url, fetchImpl) {
  const response = await fetchImpl(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`fetch ${url} failed: ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${url} is not a float32 asset`);
  }
  return new Float32Array(bytes);
}

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

async function loadDecoderWeights(root, metadata, fetchImpl) {
  const packedManifestResponse = await fetchImpl(new URL("weights.json", root), {
    cache: "force-cache",
  });
  if (packedManifestResponse.ok) {
    const manifest = await packedManifestResponse.json();
    const response = await fetchImpl(new URL(manifest.file, root), {
      cache: "force-cache",
    });
    if (!response.ok) {
      throw new Error(`fetch packed decoder weights failed: ${response.status}`);
    }
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== manifest.byteLength) {
      throw new Error("packed decoder weight length does not match its manifest");
    }
    return new Map(
      decoderWeightNames(metadata).map((name) => {
        const tensor = manifest.tensors?.[name];
        if (!tensor) throw new Error(`packed decoder weight is missing: ${name}`);
        return [
          name,
          new Float32Array(bytes, tensor.offsetBytes, tensor.length),
        ];
      }),
    );
  }
  if (packedManifestResponse.status !== 404) {
    throw new Error(
      `fetch decoder weight manifest failed: ${packedManifestResponse.status}`,
    );
  }
  const entries = await Promise.all(
    decoderWeightNames(metadata).map(async (name) => [
      name,
      await fetchFloat32(new URL(name, root), fetchImpl),
    ]),
  );
  return new Map(entries);
}

function validateMetadata(metadata, bundle) {
  if (metadata.onnxFree !== true) {
    throw new Error("custom decoder metadata is not ONNX-free");
  }
  const checks = [
    ["frame length", metadata.frameLength, bundle.frame_length],
    ["codebook count", metadata.numCodebooks, bundle.num_codebooks],
    ["channel count", metadata.channels, bundle.channels],
    ["segment samples", metadata.segmentSamples, bundle.segment_samples],
  ];
  for (const [name, actual, expected] of checks) {
    if (actual !== expected) {
      throw new Error(`custom decoder ${name} mismatch: ${actual} != ${expected}`);
    }
  }
  if (
    !metadata.front ||
    !metadata.post ||
    metadata.front.lstmLayers?.length !== 2 ||
    metadata.layers?.length !== 4 ||
    metadata.post.convLayers?.length !== 12
  ) {
    throw new Error("custom decoder metadata has an invalid model layout");
  }
}

function initializeKernel(module, metadata, weights) {
  const allocated = [];
  const allocate = (length) => {
    requireInteger(length, "allocation length");
    const pointer = module._malloc(length * Float32Array.BYTES_PER_ELEMENT);
    if (pointer === 0) {
      throw new Error(`failed to allocate ${length} decoder float32 values`);
    }
    allocated.push(pointer);
    return pointer;
  };
  allocate.free = (pointer) => {
    module._free(pointer);
    const index = allocated.indexOf(pointer);
    if (index >= 0) allocated.splice(index, 1);
  };
  const load = (name) => {
    const value = weights.get(name);
    if (!value) throw new Error(`custom decoder weight is missing: ${name}`);
    return value;
  };
  const copy = (pointer, values) => module.HEAPF32.set(values, pointer / 4);

  const maxInput = Math.max(
    ...metadata.layers.map((layer) => layer.inputChannels * layer.inputTime),
  );
  const maxOutput = Math.max(
    ...metadata.layers.map((layer) => layer.outputChannels * layer.rawOutputTime),
  );
  const activationLength = Math.max(maxInput, maxOutput);
  const state = {
    allocated,
    input: allocate(activationLength),
    output: allocate(activationLength),
    layers: [],
  };

  for (const layer of metadata.layers) {
    const rawWeights = load(`layer-${layer.layer}-weight.f32le`);
    const bias = load(`layer-${layer.layer}-bias.f32le`);
    const normScale = load(`layer-${layer.layer}-norm-scale.f32le`);
    const normBias = load(`layer-${layer.layer}-norm-bias.f32le`);
    const unpacked = allocate(rawWeights.length);
    const packed = allocate(rawWeights.length);
    const biasPointer = allocate(bias.length);
    const normScalePointer = allocate(normScale.length);
    const normBiasPointer = allocate(normBias.length);
    copy(unpacked, rawWeights);
    copy(biasPointer, bias);
    copy(normScalePointer, normScale);
    copy(normBiasPointer, normBias);
    if (
      module._pack_conv_transpose1d_weights(
        unpacked,
        packed,
        layer.inputChannels,
        layer.outputChannels,
        layer.stride,
      ) !== 1
    ) {
      throw new Error(`failed to pack decoder transpose layer ${layer.layer}`);
    }
    allocate.free(unpacked);
    state.layers.push({
      packed,
      bias: biasPointer,
      normScale: normScalePointer,
      normBias: normBiasPointer,
    });
  }

  state.front = initializeFront(module, metadata.front, load, allocate, copy);
  state.post = initializePost(module, metadata, load, allocate, copy);
  return state;
}

function initializeFront(module, metadata, load, allocate, copy) {
  const frameLength = metadata.conv.inputTime;
  const hiddenSize = metadata.conv.outputChannels;
  const activationLength = frameLength * hiddenSize;
  const embeddings = load("front-rvq-embeddings.f32le");
  const pointers = {
    codes: allocate(metadata.rvq.codebooks * frameLength),
    embeddings: allocate(embeddings.length),
    padded: allocate(metadata.conv.paddedInputTime * metadata.conv.inputChannels),
    activation0: allocate(activationLength),
    activation1: allocate(activationLength),
    activation2: allocate(activationLength),
    convPacked: allocate(
      metadata.conv.outputChannels * metadata.conv.inputChannels * metadata.conv.kernel,
    ),
    convBias: allocate(metadata.conv.outputChannels),
    convNormScale: allocate(metadata.conv.outputChannels),
    convNormBias: allocate(metadata.conv.outputChannels),
    lstmHidden: allocate(hiddenSize),
    lstmCell: allocate(hiddenSize),
    lstmInputProjection: allocate(frameLength * 4 * hiddenSize),
  };
  copy(pointers.embeddings, embeddings);

  const convWeights = load("front-conv-weight.f32le");
  const unpackedConv = allocate(convWeights.length);
  copy(unpackedConv, convWeights);
  copy(pointers.convBias, load("front-conv-bias.f32le"));
  copy(pointers.convNormScale, load("front-conv-norm-scale.f32le"));
  copy(pointers.convNormBias, load("front-conv-norm-bias.f32le"));
  if (
    module._pack_conv1d_nhwc_weights_8(
      unpackedConv,
      pointers.convPacked,
      metadata.conv.inputChannels,
      metadata.conv.outputChannels,
      metadata.conv.kernel,
    ) !== 1
  ) {
    throw new Error("failed to pack decoder front convolution");
  }
  allocate.free(unpackedConv);

  const lstmLayers = metadata.lstmLayers.map((layer) => {
    const inputWeights = load(`front-lstm-${layer.layer}-input-weight.f32le`);
    const recurrentWeights = load(`front-lstm-${layer.layer}-recurrent-weight.f32le`);
    const bias = load(`front-lstm-${layer.layer}-bias.f32le`);
    const unpackedInput = allocate(inputWeights.length);
    const unpackedRecurrent = allocate(recurrentWeights.length);
    const layerPointers = {
      packedInput: allocate(inputWeights.length),
      packedRecurrent: allocate(recurrentWeights.length),
      bias: allocate(bias.length),
    };
    copy(unpackedInput, inputWeights);
    copy(unpackedRecurrent, recurrentWeights);
    copy(layerPointers.bias, bias);
    const inputPacked = module._pack_linear_weights_8(
      unpackedInput,
      layerPointers.packedInput,
      hiddenSize,
      4 * hiddenSize,
    );
    const recurrentPacked = module._pack_linear_weights_8(
      unpackedRecurrent,
      layerPointers.packedRecurrent,
      hiddenSize,
      4 * hiddenSize,
    );
    allocate.free(unpackedInput);
    allocate.free(unpackedRecurrent);
    if (inputPacked !== 1 || recurrentPacked !== 1) {
      throw new Error(`failed to pack decoder front LSTM ${layer.layer}`);
    }
    return { ...layer, pointers: layerPointers };
  });
  return { pointers, lstmLayers };
}

function initializePost(module, metadata, load, allocate, copy) {
  const convLayers = metadata.post.convLayers.map((layer) => {
    const rawWeights = load(`post-conv-${layer.layer}-weight.f32le`);
    const bias = load(`post-conv-${layer.layer}-bias.f32le`);
    const normScale = load(`post-conv-${layer.layer}-norm-scale.f32le`);
    const normBias = load(`post-conv-${layer.layer}-norm-bias.f32le`);
    const unpacked = allocate(rawWeights.length);
    const pointers = {
      packed: allocate(rawWeights.length),
      bias: allocate(bias.length),
      normScale: allocate(normScale.length),
      normBias: allocate(normBias.length),
    };
    copy(unpacked, rawWeights);
    copy(pointers.bias, bias);
    copy(pointers.normScale, normScale);
    copy(pointers.normBias, normBias);
    if (
      module._pack_conv1d_nhwc_weights_8(
        unpacked,
        pointers.packed,
        layer.inputChannels,
        layer.outputChannels,
        layer.kernel,
      ) !== 1
    ) {
      throw new Error(`failed to pack decoder post convolution ${layer.layer}`);
    }
    allocate.free(unpacked);
    return { ...layer, pointers };
  });

  const final = metadata.post.finalConv;
  const finalWeights = load("final-conv-weight.f32le");
  const finalBias = load("final-conv-bias.f32le");
  const finalNormScale = load("final-conv-norm-scale.f32le");
  const finalNormBias = load("final-conv-norm-bias.f32le");
  const paddedWeights = new Float32Array(
    final.kernelOutputChannels * final.inputChannels * final.kernel,
  );
  const paddedBias = new Float32Array(final.kernelOutputChannels);
  paddedWeights.set(finalWeights);
  paddedBias.set(finalBias);
  const unpackedFinal = allocate(paddedWeights.length);
  const finalPointers = {
    packed: allocate(paddedWeights.length),
    bias: allocate(paddedBias.length),
    normScale: allocate(finalNormScale.length),
    normBias: allocate(finalNormBias.length),
  };
  copy(unpackedFinal, paddedWeights);
  copy(finalPointers.bias, paddedBias);
  copy(finalPointers.normScale, finalNormScale);
  copy(finalPointers.normBias, finalNormBias);
  if (
    module._pack_conv1d_nhwc_weights_8(
      unpackedFinal,
      finalPointers.packed,
      final.inputChannels,
      final.kernelOutputChannels,
      final.kernel,
    ) !== 1
  ) {
    throw new Error("failed to pack decoder final convolution");
  }
  allocate.free(unpackedFinal);

  const maxActivationLength = Math.max(
    ...metadata.layers.map((layer) => layer.croppedOutputTime * layer.outputChannels),
  );
  const maxReducedLength = Math.max(
    ...convLayers.map((layer) => layer.outputTime * layer.outputChannels),
    final.outputTime * final.outputChannels,
  );
  const maxPaddedLength = Math.max(
    ...convLayers.map((layer) => layer.paddedInputTime * layer.inputChannels),
    final.paddedInputTime * final.inputChannels,
  );
  return {
    convLayers,
    blocks: metadata.post.blocks,
    final: { ...final, pointers: finalPointers },
    pointers: {
      shortcut: allocate(maxActivationLength),
      reduced: allocate(maxReducedLength),
      padded: allocate(maxPaddedLength),
      planarOutput: allocate(final.outputTime * final.outputChannels),
    },
  };
}

function runFront(module, state, codes, metadata, timings) {
  const front = state.front;
  const pointers = front.pointers;
  const frameLength = metadata.conv.inputTime;
  const hiddenSize = metadata.conv.outputChannels;
  let started = performance.now();
  module.HEAPU16.set(codes, pointers.codes / Uint16Array.BYTES_PER_ELEMENT);
  if (
    module._rvq_decode_codes_nhwc(
      pointers.codes,
      pointers.embeddings,
      pointers.activation0,
      frameLength,
      metadata.rvq.dimension,
      metadata.rvq.entries,
      metadata.rvq.codebooks,
    ) !== 1
  ) {
    throw new Error("decoder codebook reconstruction failed");
  }
  timings.rvq += performance.now() - started;
  started = performance.now();
  if (
    module._reflect_pad_nhwc(
      pointers.activation0,
      pointers.padded,
      frameLength,
      metadata.conv.inputChannels,
      metadata.conv.paddingLeft,
      metadata.conv.paddingRight,
    ) !== 1 ||
    module._conv1d_nhwc_simd_8x8(
      pointers.padded,
      pointers.convPacked,
      pointers.convBias,
      pointers.activation1,
      metadata.conv.paddedInputTime,
      metadata.conv.inputChannels,
      hiddenSize,
      metadata.conv.kernel,
      metadata.conv.stride,
    ) !== 1 ||
    module._group_norm_nhwc_in_place(
      pointers.activation1,
      pointers.convNormScale,
      pointers.convNormBias,
      frameLength,
      hiddenSize,
    ) !== 1
  ) {
    throw new Error("decoder front convolution failed");
  }
  timings.convolution += performance.now() - started;

  const runLstm = (input, layer, output) => {
    if (
      module._lstm_layer_simd_64(
        input,
        layer.pointers.packedInput,
        layer.pointers.packedRecurrent,
        layer.pointers.bias,
        output,
        pointers.lstmHidden,
        pointers.lstmCell,
        pointers.lstmInputProjection,
        frameLength,
        hiddenSize,
      ) !== 1
    ) {
      throw new Error(`decoder front LSTM ${layer.layer} failed`);
    }
  };
  started = performance.now();
  runLstm(pointers.activation1, front.lstmLayers[0], pointers.activation0);
  timings.lstm0 += performance.now() - started;
  started = performance.now();
  runLstm(pointers.activation0, front.lstmLayers[1], pointers.activation2);
  timings.lstm1 += performance.now() - started;
  started = performance.now();
  if (
    module._add_elu_nhwc_in_place(
      pointers.activation2,
      pointers.activation1,
      frameLength * hiddenSize,
    ) !== 1
  ) {
    throw new Error("decoder front output assembly failed");
  }
  timings.assembly += performance.now() - started;
  return pointers.activation2;
}

function runPostConv(module, post, input, layer, output, applyElu) {
  let convolutionInput = input;
  if (layer.paddingLeft !== 0 || layer.paddingRight !== 0) {
    const pad = applyElu ? module._reflect_pad_elu_nhwc : module._reflect_pad_nhwc;
    if (
      pad(
        input,
        post.pointers.padded,
        layer.inputTime,
        layer.inputChannels,
        layer.paddingLeft,
        layer.paddingRight,
      ) !== 1
    ) {
      throw new Error(`decoder post padding ${layer.layer} failed`);
    }
    convolutionInput = post.pointers.padded;
  } else if (
    applyElu &&
    module._elu_nhwc_in_place(input, layer.inputTime * layer.inputChannels) !== 1
  ) {
    throw new Error(`decoder post ELU ${layer.layer} failed`);
  }
  if (
    module._conv1d_nhwc_simd_8x8(
      convolutionInput,
      layer.pointers.packed,
      layer.pointers.bias,
      output,
      layer.paddedInputTime,
      layer.inputChannels,
      layer.outputChannels,
      layer.kernel,
      layer.stride,
    ) !== 1 ||
    module._group_norm_nhwc_in_place(
      output,
      layer.pointers.normScale,
      layer.pointers.normBias,
      layer.outputTime,
      layer.outputChannels,
    ) !== 1
  ) {
    throw new Error(`decoder post convolution ${layer.layer} failed`);
  }
}

function decodeFrames(module, state, metadata, frames) {
  const samplesPerFrame = metadata.channels * metadata.segmentSamples;
  const audio = new Float32Array(frames.length * samplesPerFrame);
  const parts = {
    front: { rvq: 0, convolution: 0, lstm0: 0, lstm1: 0, assembly: 0 },
    convTranspose: 0,
    residual: 0,
    final: 0,
    copy: 0,
  };
  const startedAll = performance.now();
  for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
    const frame = frames[frameIndex];
    let current = runFront(module, state, frame.codes, metadata.front, parts.front);
    for (let layerIndex = 0; layerIndex < metadata.layers.length; layerIndex += 1) {
      const layer = metadata.layers[layerIndex];
      const layerState = state.layers[layerIndex];
      const raw = current === state.output ? state.input : state.output;
      let started = performance.now();
      if (
        module._conv_transpose1d_phase_simd_8x8_nhwc(
          current,
          layerState.packed,
          layerState.bias,
          raw,
          layer.inputTime,
          layer.inputChannels,
          layer.outputChannels,
          layer.stride,
        ) !== 1 ||
        module._group_norm_nhwc_in_place(
          raw,
          layerState.normScale,
          layerState.normBias,
          layer.rawOutputTime,
          layer.outputChannels,
        ) !== 1
      ) {
        throw new Error(`decoder transpose layer ${layerIndex} failed`);
      }
      const cropped = raw === state.output ? state.input : state.output;
      if (
        module._crop_nhwc(
          raw,
          cropped,
          layer.rawOutputTime,
          layer.outputChannels,
          layer.cropLeft,
          layer.cropRight,
        ) !== 1
      ) {
        throw new Error(`decoder crop ${layerIndex} failed`);
      }
      parts.convTranspose += performance.now() - started;

      started = performance.now();
      const [shortcutIndex, reduceIndex, expandIndex] = state.post.blocks[layerIndex];
      const shortcutLayer = state.post.convLayers[shortcutIndex];
      const reduceLayer = state.post.convLayers[reduceIndex];
      const expandLayer = state.post.convLayers[expandIndex];
      runPostConv(
        module,
        state.post,
        cropped,
        shortcutLayer,
        state.post.pointers.shortcut,
        false,
      );
      runPostConv(
        module,
        state.post,
        cropped,
        reduceLayer,
        state.post.pointers.reduced,
        true,
      );
      runPostConv(
        module,
        state.post,
        state.post.pointers.reduced,
        expandLayer,
        raw,
        true,
      );
      if (
        module._add_elu_nhwc_in_place(
          raw,
          state.post.pointers.shortcut,
          layer.croppedOutputTime * layer.outputChannels,
        ) !== 1
      ) {
        throw new Error(`decoder residual block ${layerIndex} failed`);
      }
      current = raw;
      parts.residual += performance.now() - started;
    }

    let started = performance.now();
    const final = state.post.final;
    if (
      module._reflect_pad_nhwc(
        current,
        state.post.pointers.padded,
        final.inputTime,
        final.inputChannels,
        final.paddingLeft,
        final.paddingRight,
      ) !== 1 ||
      module._conv1d_nhwc_simd_8x8(
        state.post.pointers.padded,
        final.pointers.packed,
        final.pointers.bias,
        state.post.pointers.shortcut,
        final.paddedInputTime,
        final.inputChannels,
        final.kernelOutputChannels,
        final.kernel,
        final.stride,
      ) !== 1 ||
      module._compact_nhwc_channels(
        state.post.pointers.shortcut,
        state.post.pointers.reduced,
        final.outputTime,
        final.kernelOutputChannels,
        final.outputChannels,
      ) !== 1 ||
      module._group_norm_nhwc_in_place(
        state.post.pointers.reduced,
        final.pointers.normScale,
        final.pointers.normBias,
        final.outputTime,
        final.outputChannels,
      ) !== 1 ||
      module._scale_nhwc_to_nct(
        state.post.pointers.reduced,
        state.post.pointers.planarOutput,
        Number(frame.scale ?? 1),
        final.outputTime,
        final.outputChannels,
      ) !== 1
    ) {
      throw new Error("decoder final projection failed");
    }
    parts.final += performance.now() - started;
    started = performance.now();
    const offset = state.post.pointers.planarOutput / Float32Array.BYTES_PER_ELEMENT;
    audio.set(
      module.HEAPF32.subarray(offset, offset + samplesPerFrame),
      frameIndex * samplesPerFrame,
    );
    parts.copy += performance.now() - started;
  }
  return {
    audio,
    elapsedMs: performance.now() - startedAll,
    partsMs: parts,
    shape: [frames.length, metadata.channels, metadata.segmentSamples],
  };
}

export async function createCustomDecoder({
  assetRoot,
  kernelModuleUrl,
  bundleMetadata,
  fetchImpl = globalThis.fetch.bind(globalThis),
  versionedAssetUrl = (asset) => asset,
}) {
  const root = normalizeUrl(assetRoot);
  const moduleUrl = kernelModuleUrl instanceof URL
    ? kernelModuleUrl
    : new URL(String(kernelModuleUrl), globalThis.location?.href);
  const metadata = await fetchJson(new URL("metadata.json", root), fetchImpl);
  validateMetadata(metadata, bundleMetadata);
  const [weights, createKernel] = await Promise.all([
    loadDecoderWeights(root, metadata, fetchImpl),
    import(String(versionedAssetUrl(moduleUrl.href))).then((module) => module.default),
  ]);
  const kernel = await createKernel({
    locateFile: (file) => String(versionedAssetUrl(new URL(file, moduleUrl).href)),
  });
  const state = initializeKernel(kernel, metadata, weights);
  weights.clear();
  let released = false;
  return {
    metadata,
    decode(frames) {
      if (released) throw new Error("custom decoder is released");
      return decodeFrames(kernel, state, metadata, frames);
    },
    release() {
      if (released) return;
      released = true;
      for (const pointer of state.allocated) kernel._free(pointer);
    },
  };
}
