import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

export async function createCustomEncoder(root, kernelModulePath) {
  const assetRoot = path.resolve(root);
  const modulePath = path.resolve(kernelModulePath);
  const metadata = JSON.parse(
    fs.readFileSync(path.join(assetRoot, "metadata.json"), "utf8"),
  );
  validateMetadata(metadata);

  const createKernel = (await import(pathToFileURL(modulePath))).default;
  const kernel = await createKernel({
    locateFile: (file) => path.join(path.dirname(modulePath), file),
  });
  const allocate = (length) => {
    const pointer = kernel._malloc(length * Float32Array.BYTES_PER_ELEMENT);
    if (pointer === 0) {
      throw new Error(`failed to allocate ${length} custom encoder values`);
    }
    return pointer;
  };

  const hiddenSize = metadata.lstmLayers[0].hiddenSize;
  const gateSize = 4 * hiddenSize;
  const audioLength = metadata.channels * metadata.segmentSamples;
  const maxActivationLength = Math.max(
    ...metadata.convLayers.map(
      (layer) => layer.outputTime * layer.outputChannels,
    ),
  );
  const maxPaddedLength = Math.max(
    ...metadata.convLayers.map(
      (layer) => layer.paddedInputTime * layer.inputChannels,
    ),
  );
  const embeddings = readFloat32(path.join(assetRoot, "rvq-embeddings.f32le"));
  const norms = readFloat32(path.join(assetRoot, "rvq-norms.f32le"));
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
  kernel.HEAPF32.set(embeddings, pointers.rvqEmbeddings / 4);
  kernel.HEAPF32.set(norms, pointers.rvqNorms / 4);

  const layers = metadata.convLayers.map((layer) => {
    const weights = readFloat32(
      path.join(assetRoot, `conv-${layer.layer}-weight.f32le`),
    );
    const bias = readFloat32(
      path.join(assetRoot, `conv-${layer.layer}-bias.f32le`),
    );
    const normScale = readFloat32(
      path.join(assetRoot, `conv-${layer.layer}-norm-scale.f32le`),
    );
    const normBias = readFloat32(
      path.join(assetRoot, `conv-${layer.layer}-norm-bias.f32le`),
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
    kernel._free(unpackedWeights);
    if (packed !== 1) {
      throw new Error(`failed to pack custom encoder layer ${layer.layer}`);
    }
    return { ...layer, pointers: layerPointers };
  });

  const lstmLayers = metadata.lstmLayers.map((layer) => {
    const inputWeights = readFloat32(
      path.join(assetRoot, `lstm-${layer.layer}-input-weight.f32le`),
    );
    const recurrentWeights = readFloat32(
      path.join(assetRoot, `lstm-${layer.layer}-recurrent-weight.f32le`),
    );
    const bias = readFloat32(
      path.join(assetRoot, `lstm-${layer.layer}-bias.f32le`),
    );
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
    kernel._free(unpackedInput);
    kernel._free(unpackedRecurrent);
    if (inputPacked !== 1 || recurrentPacked !== 1) {
      throw new Error(`failed to pack custom LSTM layer ${layer.layer}`);
    }
    return { ...layer, pointers: layerPointers };
  });

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
      const padded = (applyElu
        ? kernel._reflect_pad_elu_nhwc
        : kernel._reflect_pad_nhwc)(
        input,
        pointers.padded,
        layer.inputTime,
        layer.inputChannels,
        layer.paddingLeft,
        layer.paddingRight,
      );
      if (padded !== 1) {
        throw new Error(`custom encoder padding failed for layer ${layerIndex}`);
      }
      convolutionInput = pointers.padded;
    } else if (applyElu) {
      const length = layer.inputTime * layer.inputChannels;
      if (kernel._elu_nhwc_in_place(input, length) !== 1) {
        throw new Error(`custom encoder ELU failed for layer ${layerIndex}`);
      }
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
      ) !== 1
    ) {
      throw new Error(`custom encoder convolution failed for layer ${layerIndex}`);
    }
    if (
      kernel._group_norm_nhwc_in_place(
        output,
        layer.pointers.normScale,
        layer.pointers.normBias,
        layer.outputTime,
        layer.outputChannels,
      ) !== 1
    ) {
      throw new Error(`custom encoder normalization failed for layer ${layerIndex}`);
    }
  };

  let released = false;
  const encode = (audio) => {
    if (released) throw new Error("custom encoder is released");
    if (!(audio instanceof Float32Array) || audio.length !== audioLength) {
      throw new Error(
        `custom encoder input must contain ${audioLength} float32 values`,
      );
    }
    const totalStarted = performance.now();
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
      const mainLength =
        layers[firstMainLayer].outputTime * layers[firstMainLayer].outputChannels;
      if (kernel._elu_nhwc_in_place(main, mainLength) !== 1) {
        throw new Error(`custom encoder residual ELU failed at layer ${firstMainLayer}`);
      }
      runLayer(secondMainLayer, main, current);
      const residualLength =
        layers[secondMainLayer].outputTime * layers[secondMainLayer].outputChannels;
      if (kernel._add_nhwc_in_place(current, shortcut, residualLength) !== 1) {
        throw new Error(`custom encoder residual add failed at layer ${secondMainLayer}`);
      }
      runLayer(downsampleLayer, current, shortcut, true);
      current = shortcut;
    }
    const frontMs = performance.now() - started;

    const firstOutput = unused(current);
    const secondOutput = unused(current, firstOutput);
    const runLstmLayer = (input, layer, output) => {
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
          hiddenSize,
        ) !== 1
      ) {
        throw new Error(`custom LSTM layer ${layer.layer} failed`);
      }
    };
    started = performance.now();
    runLstmLayer(current, lstmLayers[0], firstOutput);
    runLstmLayer(firstOutput, lstmLayers[1], secondOutput);
    if (
      kernel._add_nhwc_in_place(
        secondOutput,
        current,
        metadata.frameLength * hiddenSize,
      ) !== 1
    ) {
      throw new Error("custom LSTM residual add failed");
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
      throw new Error("custom residual vector quantizer failed");
    }
    const quantizerMs = performance.now() - started;
    const codeLength = metadata.rvq.codebooks * metadata.frameLength;
    const codes = new Int32Array(
      kernel.HEAPF32.buffer,
      pointers.rvqCodes,
      codeLength,
    ).slice();
    return {
      codes,
      scale: new Float32Array([scale]),
      elapsedMs: performance.now() - totalStarted,
      stagesMs: {
        convolutionalFront: frontMs,
        recurrent: recurrentMs,
        latentProjection: projectionMs,
        residualVectorQuantizer: quantizerMs,
      },
    };
  };

  return {
    metadata,
    encode,
    release() {
      if (released) return;
      released = true;
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

function readFloat32(file) {
  const bytes = fs.readFileSync(file);
  const copy = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  return new Float32Array(copy);
}

function validateMetadata(metadata) {
  if (
    !Number.isInteger(metadata.channels) ||
    !Number.isInteger(metadata.segmentSamples) ||
    !Number.isInteger(metadata.segmentStride) ||
    !Number.isInteger(metadata.frameLength) ||
    metadata.convLayers?.length !== 18 ||
    metadata.lstmLayers?.length !== 2 ||
    metadata.rvq?.codebooks !== metadata.numCodebooks
  ) {
    throw new Error("custom encoder metadata is not valid");
  }
}
