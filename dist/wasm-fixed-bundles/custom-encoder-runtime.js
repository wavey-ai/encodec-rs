function normalizeUrl(value) {
  const url = value instanceof URL ? value : new URL(String(value), globalThis.location?.href);
  return new URL(url.href.replace(/\/?$/, "/"));
}

async function fetchJson(url, fetchImpl) {
  const response = await fetchImpl(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`fetch ${url} failed: ${response.status}`);
  return response.json();
}

async function fetchFloat32(url, fetchImpl) {
  const response = await fetchImpl(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`fetch ${url} failed: ${response.status}`);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${url} is not a float32 asset`);
  }
  return new Float32Array(bytes);
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
      throw new Error(`custom encoder ${name} mismatch: ${actual} != ${expected}`);
    }
  }
  if (
    metadata.convLayers?.length !== 18 ||
    metadata.lstmLayers?.length !== 2 ||
    metadata.rvq?.codebooks !== metadata.numCodebooks
  ) {
    throw new Error("custom encoder metadata has an invalid model layout");
  }
}

function encoderWeightNames(metadata) {
  const names = new Set([
    "rvq-embeddings.f32le",
    "rvq-norms.f32le",
  ]);
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

async function loadWeights(root, metadata, fetchImpl) {
  const packedManifestResponse = await fetchImpl(new URL("weights.json", root), {
    cache: "force-cache",
  });
  if (packedManifestResponse.ok) {
    const manifest = await packedManifestResponse.json();
    const response = await fetchImpl(new URL(manifest.file, root), {
      cache: "force-cache",
    });
    if (!response.ok) {
      throw new Error(`fetch packed encoder weights failed: ${response.status}`);
    }
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== manifest.byteLength) {
      throw new Error("packed encoder weight length does not match its manifest");
    }
    return new Map(
      encoderWeightNames(metadata).map((name) => {
        const tensor = manifest.tensors?.[name];
        if (!tensor) throw new Error(`packed encoder weight is missing: ${name}`);
        return [
          name,
          new Float32Array(bytes, tensor.offsetBytes, tensor.length),
        ];
      }),
    );
  }
  if (packedManifestResponse.status !== 404) {
    throw new Error(
      `fetch encoder weight manifest failed: ${packedManifestResponse.status}`,
    );
  }
  return new Map(
    await Promise.all(
      encoderWeightNames(metadata).map(async (name) => [
        name,
        await fetchFloat32(new URL(name, root), fetchImpl),
      ]),
    ),
  );
}

function initializeEncoder(kernel, metadata, weights) {
  const allocated = [];
  const allocate = (length) => {
    if (!Number.isInteger(length) || length <= 0) {
      throw new Error(`invalid encoder allocation length: ${length}`);
    }
    const pointer = kernel._malloc(length * Float32Array.BYTES_PER_ELEMENT);
    if (pointer === 0) {
      throw new Error(`failed to allocate ${length} encoder float32 values`);
    }
    allocated.push(pointer);
    return pointer;
  };
  allocate.free = (pointer) => {
    kernel._free(pointer);
    const index = allocated.indexOf(pointer);
    if (index >= 0) allocated.splice(index, 1);
  };
  const load = (name) => {
    const value = weights.get(name);
    if (!value) throw new Error(`custom encoder weight is missing: ${name}`);
    return value;
  };
  const copy = (pointer, values) => kernel.HEAPF32.set(values, pointer / 4);

  const hiddenSize = metadata.lstmLayers[0].hiddenSize;
  const gateSize = 4 * hiddenSize;
  const audioLength = metadata.channels * metadata.segmentSamples;
  const maxActivationLength = Math.max(
    ...metadata.convLayers.map((layer) => layer.outputTime * layer.outputChannels),
  );
  const maxPaddedLength = Math.max(
    ...metadata.convLayers.map((layer) => layer.paddedInputTime * layer.inputChannels),
  );
  const embeddings = load("rvq-embeddings.f32le");
  const norms = load("rvq-norms.f32le");
  const pointers = {
    audio: allocate(audioLength),
    normalizedAudio: allocate(audioLength),
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
  copy(pointers.rvqEmbeddings, embeddings);
  copy(pointers.rvqNorms, norms);

  const layers = metadata.convLayers.map((layer) => {
    const rawWeights = load(`conv-${layer.layer}-weight.f32le`);
    const bias = load(`conv-${layer.layer}-bias.f32le`);
    const normScale = load(`conv-${layer.layer}-norm-scale.f32le`);
    const normBias = load(`conv-${layer.layer}-norm-bias.f32le`);
    const unpacked = allocate(rawWeights.length);
    const layerPointers = {
      packed: allocate(rawWeights.length),
      bias: allocate(bias.length),
      normScale: allocate(normScale.length),
      normBias: allocate(normBias.length),
    };
    copy(unpacked, rawWeights);
    copy(layerPointers.bias, bias);
    copy(layerPointers.normScale, normScale);
    copy(layerPointers.normBias, normBias);
    if (
      kernel._pack_conv1d_nhwc_weights_8(
        unpacked,
        layerPointers.packed,
        layer.inputChannels,
        layer.outputChannels,
        layer.kernel,
      ) !== 1
    ) {
      throw new Error(`failed to pack encoder convolution ${layer.layer}`);
    }
    allocate.free(unpacked);
    return { ...layer, pointers: layerPointers };
  });

  const lstmLayers = metadata.lstmLayers.map((layer) => {
    const inputWeights = load(`lstm-${layer.layer}-input-weight.f32le`);
    const recurrentWeights = load(`lstm-${layer.layer}-recurrent-weight.f32le`);
    const bias = load(`lstm-${layer.layer}-bias.f32le`);
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
    const inputPacked = kernel._pack_linear_weights_8(
      unpackedInput,
      layerPointers.packedInput,
      hiddenSize,
      gateSize,
    );
    const recurrentPacked = kernel._pack_linear_weights_8(
      unpackedRecurrent,
      layerPointers.packedRecurrent,
      hiddenSize,
      gateSize,
    );
    allocate.free(unpackedInput);
    allocate.free(unpackedRecurrent);
    if (inputPacked !== 1 || recurrentPacked !== 1) {
      throw new Error(`failed to pack encoder LSTM ${layer.layer}`);
    }
    return { ...layer, pointers: layerPointers };
  });

  return {
    allocated,
    audioLength,
    hiddenSize,
    pointers,
    layers,
    lstmLayers,
  };
}

function encodeFrame(kernel, metadata, state, audio) {
  if (!(audio instanceof Float32Array) || audio.length !== state.audioLength) {
    throw new Error(
      `custom encoder input must contain ${state.audioLength} float32 values`,
    );
  }
  const { pointers, layers } = state;
  const activationPointers = [
    pointers.activation0,
    pointers.activation1,
    pointers.activation2,
  ];
  const unused = (...used) =>
    activationPointers.find((pointer) => !used.includes(pointer));
  const runLayer = (layerIndex, input, output, applyElu = false) => {
    const layer = layers[layerIndex];
    let convolutionInput = input;
    if (layer.paddingLeft !== 0 || layer.paddingRight !== 0) {
      const pad = applyElu
        ? kernel._reflect_pad_elu_nhwc
        : kernel._reflect_pad_nhwc;
      if (
        pad(
          input,
          pointers.padded,
          layer.inputTime,
          layer.inputChannels,
          layer.paddingLeft,
          layer.paddingRight,
        ) !== 1
      ) {
        throw new Error(`encoder padding ${layerIndex} failed`);
      }
      convolutionInput = pointers.padded;
    } else if (
      applyElu &&
      kernel._elu_nhwc_in_place(input, layer.inputTime * layer.inputChannels) !== 1
    ) {
      throw new Error(`encoder ELU ${layerIndex} failed`);
    }
    if (
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
      ) !== 1 ||
      kernel._group_norm_nhwc_in_place(
        output,
        layer.pointers.normScale,
        layer.pointers.normBias,
        layer.outputTime,
        layer.outputChannels,
      ) !== 1
    ) {
      throw new Error(`encoder convolution ${layerIndex} failed`);
    }
  };

  const startedAll = performance.now();
  kernel.HEAPF32.set(audio, pointers.audio / 4);
  let started = performance.now();
  const scale = kernel._normalize_audio_planar_to_nhwc(
    pointers.audio,
    pointers.normalizedAudio,
    metadata.segmentSamples,
    metadata.channels,
  );
  let current = pointers.activation0;
  runLayer(0, pointers.normalizedAudio, current);
  for (const [shortcutLayer, firstMainLayer, secondMainLayer, downsampleLayer] of [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
  ]) {
    const shortcut = unused(current);
    const main = unused(current, shortcut);
    runLayer(shortcutLayer, current, shortcut);
    runLayer(firstMainLayer, current, main, true);
    if (
      kernel._elu_nhwc_in_place(
        main,
        layers[firstMainLayer].outputTime * layers[firstMainLayer].outputChannels,
      ) !== 1
    ) {
      throw new Error(`encoder residual ELU ${firstMainLayer} failed`);
    }
    runLayer(secondMainLayer, main, current);
    if (
      kernel._add_nhwc_in_place(
        current,
        shortcut,
        layers[secondMainLayer].outputTime * layers[secondMainLayer].outputChannels,
      ) !== 1
    ) {
      throw new Error(`encoder residual add ${secondMainLayer} failed`);
    }
    runLayer(downsampleLayer, current, shortcut, true);
    current = shortcut;
  }
  const convolutionalFrontMs = performance.now() - started;

  const firstOutput = unused(current);
  const secondOutput = unused(current, firstOutput);
  const runLstm = (input, layer, output) => {
    if (
      kernel._lstm_layer_simd_64(
        input,
        layer.pointers.packedInput,
        layer.pointers.packedRecurrent,
        layer.pointers.bias,
        output,
        pointers.lstmHidden,
        pointers.lstmCell,
        pointers.lstmInputProjection,
        metadata.frameLength,
        state.hiddenSize,
      ) !== 1
    ) {
      throw new Error(`encoder LSTM ${layer.layer} failed`);
    }
  };
  started = performance.now();
  runLstm(current, state.lstmLayers[0], firstOutput);
  runLstm(firstOutput, state.lstmLayers[1], secondOutput);
  if (
    kernel._add_nhwc_in_place(
      secondOutput,
      current,
      metadata.frameLength * state.hiddenSize,
    ) !== 1
  ) {
    throw new Error("encoder LSTM residual add failed");
  }
  const recurrentMs = performance.now() - started;

  started = performance.now();
  runLayer(17, secondOutput, firstOutput, true);
  const projectionMs = performance.now() - started;
  started = performance.now();
  if (
    kernel._rvq_encode_simd_8(
      firstOutput,
      pointers.rvqResidual,
      pointers.rvqEmbeddings,
      pointers.rvqNorms,
      pointers.rvqCodes,
      metadata.frameLength,
      metadata.rvq.dimension,
      metadata.rvq.entries,
      metadata.rvq.codebooks,
    ) !== 1
  ) {
    throw new Error("encoder residual vector quantizer failed");
  }
  const quantizerMs = performance.now() - started;
  const codes = new Int32Array(
    kernel.HEAPF32.buffer,
    pointers.rvqCodes,
    metadata.rvq.codebooks * metadata.frameLength,
  ).slice();
  return {
    codes,
    scale,
    elapsedMs: performance.now() - startedAll,
    partsMs: {
      convolutionalFront: convolutionalFrontMs,
      recurrent: recurrentMs,
      latentProjection: projectionMs,
      residualVectorQuantizer: quantizerMs,
    },
  };
}

export async function createCustomEncoder({
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
    loadWeights(root, metadata, fetchImpl),
    import(String(versionedAssetUrl(moduleUrl.href))).then((module) => module.default),
  ]);
  const kernel = await createKernel({
    locateFile: (file) => String(versionedAssetUrl(new URL(file, moduleUrl).href)),
  });
  const state = initializeEncoder(kernel, metadata, weights);
  weights.clear();
  let released = false;
  return {
    metadata,
    encode(audio) {
      if (released) throw new Error("custom encoder is released");
      return encodeFrame(kernel, metadata, state, audio);
    },
    release() {
      if (released) return;
      released = true;
      for (const pointer of state.allocated) kernel._free(pointer);
    },
  };
}
