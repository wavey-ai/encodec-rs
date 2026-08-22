import init, {
  initPanicHook,
  lmEcdcChunk,
  lmEcdcDecodeChunks,
  lmEcdcFixedHeaderForWeights,
  QuantizedLmChunkDecoder,
  QuantizedLmChunkEncoder,
  ecdcMetadata,
  ecdcOverlapAddForMetadata,
  stableHashHex,
} from "../pkg/encodec_rs.js";
import { createCustomDecoder } from "../browser-runtime/custom-decoder-runtime.js";
import { createCustomEncoder } from "../browser-runtime/custom-encoder-runtime.js";
import { createWebGpuDecoder } from "../browser-runtime/webgpu-decoder-runtime.js";
import { createWebGpuEcdcDecoder } from "../browser-runtime/webgpu-ecdc-decoder-runtime.js";
import { createWebGpuEncoder } from "../browser-runtime/webgpu-encoder-runtime.js";

let wasmReady;
let ort = null;
let ortPromise = null;
const sessionCache = new Map();

window.webgpuMatrix = {
  ready,
  encode,
  decode,
  compareDecoders,
  readyCustom,
  customRoundTrip,
  customRuntimeChunk,
  webGpuRoundTrip,
  webGpuDecodeEcdc,
};

async function ready() {
  const providers = matrixExecutionProviders();
  if (providers.includes("webgpu") && !navigator.gpu) {
    throw new Error(`navigator.gpu is unavailable in ${navigator.userAgent}`);
  }
  await Promise.all([initWasm(), ensureOrt()]);
  return {
    userAgent: navigator.userAgent,
    secureContext: window.isSecureContext,
    hasWebGpu: Boolean(navigator.gpu),
    executionProviders: providers,
    ortWebVersion: ort.env?.versions?.web ?? null,
  };
}

async function readyCustom() {
  await initWasm();
  return {
    userAgent: navigator.userAgent,
    secureContext: window.isSecureContext,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    ortLoaded: ort !== null,
  };
}

async function ensureOrt() {
  if (!ortPromise) {
    ortPromise = import("./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs")
      .then((module) => {
        ort = module;
        ort.env.wasm.wasmPaths = new URL(
          "./node_modules/onnxruntime-web/dist/",
          window.location.href,
        ).href;
        ort.env.wasm.numThreads = 1;
        if (ort.env.webgpu) {
          ort.env.webgpu.powerPreference = "high-performance";
        }
        return ort;
      });
  }
  return ortPromise;
}

async function encode(options) {
  await ready();
  const { bundleName, inputWavUrl, downloadName } = options;
  const bundleRoot = new URL(`../onnx-bundles/${bundleName}/`, window.location.href).href;
  const bundleJson = await fetchText(new URL("bundle.json", bundleRoot).href);
  const meta = JSON.parse(bundleJson);
  if (!meta.lm_quant_weight_model) {
    throw new Error(`${bundleName} is missing lm_quant_weight_model`);
  }
  const weights = new Uint8Array(await fetchArrayBuffer(new URL(meta.lm_quant_weight_model, bundleRoot).href));
  const wav = decodeWav(await fetchArrayBuffer(inputWavUrl));
  if (wav.sampleRate !== meta.sample_rate || wav.channels !== meta.channels) {
    throw new Error(`input WAV ${wav.sampleRate} Hz ${wav.channels}ch does not match ${meta.sample_rate} Hz ${meta.channels}ch`);
  }

  const encodeSession = await getSession(`${bundleName}:encode`, new URL(meta.encode_model, bundleRoot).href);
  const segments = buildSegmentBatch(wav.audio, wav.frames, meta);
  const chunks = [lmEcdcFixedHeaderForWeights(bundleJson, wav.frames, 2, weights)];
  let frameOnnxMs = 0;
  let lmMs = 0;

  for (let index = 0; index < segments.count; index += 1) {
    if (index === 0 || (index + 1) % 10 === 0 || index + 1 === segments.count) {
      console.log(`encode ${bundleName} segment ${index + 1}/${segments.count}`);
    }
    const segment = buildSingleSegment(wav.audio, wav.frames, segments, index, meta);
    const frameStarted = performance.now();
    const outputs = await encodeSession.run({
      [encodeSession.inputNames[0]]: new ort.Tensor("float32", segment.audio, [
        1,
        meta.channels,
        meta.segment_samples,
      ]),
    });
    frameOnnxMs += performance.now() - frameStarted;
    const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
    const frame = buildRawFrame(codesTensor.data, scaleTensor.data, segment, meta, index);
    const lmStarted = performance.now();
    const payload = encodeQ8LmFrame(bundleJson, weights, frame, meta);
    lmMs += performance.now() - lmStarted;
    chunks.push(lmEcdcChunk(payload));
  }

  const ecdc = concatUint8Chunks(chunks);
  const metadata = ecdcMetadata(ecdc);
  downloadBytes(downloadName, ecdc, "application/octet-stream");
  return {
    runtime: "browser-webgpu-macos-arm64",
    bundleName,
    bandwidthKbps: meta.bandwidth_kbps,
    audioSamples: wav.frames,
    segments: segments.count,
    ecdcBytes: ecdc.byteLength,
    lmHash: stableHashHex(weights),
    ecdcMetadata: metadata,
    timings: {
      frameOnnxMs: roundMs(frameOnnxMs),
      lmMs: roundMs(lmMs),
    },
  };
}

async function decode(options) {
  await ready();
  const { bundleName, inputEcdcUrl, downloadName } = options;
  const bundleRoot = new URL(`../onnx-bundles/${bundleName}/`, window.location.href).href;
  const bundleJson = await fetchText(new URL("bundle.json", bundleRoot).href);
  const meta = JSON.parse(bundleJson);
  if (!meta.lm_quant_weight_model) {
    throw new Error(`${bundleName} is missing lm_quant_weight_model`);
  }
  const weights = new Uint8Array(await fetchArrayBuffer(new URL(meta.lm_quant_weight_model, bundleRoot).href));
  const ecdc = new Uint8Array(await fetchArrayBuffer(inputEcdcUrl));
  const metadata = ecdcMetadata(ecdc);
  const acv = metadata.acv ?? metadata.bitstream_version ?? 0;
  if (acv !== 2) {
    throw new Error(`WebGPU matrix runner only accepts q8 acv=2, got acv=${acv}`);
  }
  const expectedHash = metadata.lmh ?? metadata.lm_hash;
  const actualHash = stableHashHex(weights);
  if (expectedHash !== actualHash) {
    throw new Error(`payload requires LM hash ${expectedHash}, browser has ${actualHash}`);
  }

  const parsed = lmEcdcDecodeChunks(bundleJson, ecdc);
  const frames = [];
  let lmMs = 0;
  for (const chunk of parsed.chunks) {
    if (frames.length === 0 || (frames.length + 1) % 10 === 0 || frames.length + 1 === parsed.chunks.length) {
      console.log(`entropy decode ${bundleName} chunk ${frames.length + 1}/${parsed.chunks.length}`);
    }
    const started = performance.now();
    frames.push(decodeQ8LmFrame(bundleJson, weights, meta, chunk));
    lmMs += performance.now() - started;
  }

  const decodeSession = await getSession(`${bundleName}:decode`, new URL(meta.decode_model, bundleRoot).href);
  const decodedFrames = await decodeFrameBatch(decodeSession, frames, meta);
  const audioLength = metadata.al ?? metadata.audio_length;
  const decodedAudio = ecdcOverlapAddForMetadata(bundleJson, JSON.stringify(metadata), decodedFrames.audio);
  const wav = writeWavBytes(decodedAudio, meta.channels, meta.sample_rate);
  downloadBytes(downloadName, wav, "audio/wav");
  return {
    runtime: "browser-webgpu-macos-arm64",
    bundleName,
    bandwidthKbps: meta.bandwidth_kbps,
    parsedFrames: frames.length,
    decodedSamples: audioLength,
    lmHash: actualHash,
    timings: {
      lmMs: roundMs(lmMs),
      decodeOnnxMs: roundMs(decodedFrames.decodeOnnxMs),
    },
    decodedShape: decodedFrames.shape,
  };
}

async function compareDecoders(options) {
  await ready();
  const {
    bundleName,
    inputEcdcUrl,
    customDecoderRootUrl,
    customKernelModuleUrl,
    iterations = 5,
    warmupFrameCount = null,
  } = options;
  const runCount = Math.max(1, Math.floor(Number(iterations) || 1));
  const bundleRoot = new URL(`../onnx-bundles/${bundleName}/`, window.location.href).href;
  const bundleJson = await fetchText(new URL("bundle.json", bundleRoot).href);
  const meta = JSON.parse(bundleJson);
  const weights = new Uint8Array(
    await fetchArrayBuffer(new URL(meta.lm_quant_weight_model, bundleRoot).href),
  );
  const ecdc = new Uint8Array(await fetchArrayBuffer(inputEcdcUrl));
  const ecdcMeta = ecdcMetadata(ecdc);
  const expectedHash = ecdcMeta.lmh ?? ecdcMeta.lm_hash;
  const actualHash = stableHashHex(weights);
  if (expectedHash !== actualHash) {
    throw new Error(`payload requires LM hash ${expectedHash}, browser has ${actualHash}`);
  }

  const entropyStarted = performance.now();
  const parsed = lmEcdcDecodeChunks(bundleJson, ecdc);
  const frames = parsed.chunks.map((chunk) =>
    decodeQ8LmFrame(bundleJson, weights, meta, chunk),
  );
  const entropyMs = performance.now() - entropyStarted;

  const mainSetupStarted = performance.now();
  const main = await getSession(
    `${bundleName}:decode:comparison`,
    new URL(meta.decode_model, bundleRoot).href,
  );
  const mainSetupMs = performance.now() - mainSetupStarted;

  const customSetupStarted = performance.now();
  const custom = await createCustomDecoder({
    assetRoot: customDecoderRootUrl,
    kernelModuleUrl: customKernelModuleUrl,
    bundleMetadata: meta,
  });
  const customSetupMs = performance.now() - customSetupStarted;

  const warmupFrames = frames.slice(
    0,
    Math.min(
      frames.length,
      Math.max(1, Math.floor(Number(warmupFrameCount) || frames.length)),
    ),
  );
  let mainOutput = await decodeFrameBatch(main, warmupFrames, meta, 1);
  let customOutput = custom.decode(warmupFrames);
  const mainSamples = [];
  const customSamples = [];
  for (let iteration = 0; iteration < runCount; iteration += 1) {
    if (iteration % 2 === 0) {
      mainOutput = await decodeFrameBatch(main, frames, meta, 1);
      mainSamples.push(mainOutput.decodeOnnxMs);
      customOutput = custom.decode(frames);
      customSamples.push(customOutput.elapsedMs);
    } else {
      customOutput = custom.decode(frames);
      customSamples.push(customOutput.elapsedMs);
      mainOutput = await decodeFrameBatch(main, frames, meta, 1);
      mainSamples.push(mainOutput.decodeOnnxMs);
    }
  }
  const parity = comparePcm(mainOutput.audio, customOutput.audio);
  custom.release();
  const audioLength = ecdcMeta.al ?? ecdcMeta.audio_length;
  const audioSeconds = audioLength / meta.sample_rate;
  const mainMedianMs = median(mainSamples);
  const customMedianMs = median(customSamples);
  return {
    runtime: "headless browser WASM",
    userAgent: navigator.userAgent,
    crossOriginIsolated,
    executionProviders: matrixExecutionProviders(),
    bundleName,
    frames: frames.length,
    audioSeconds,
    entropyMs: roundMs(entropyMs),
    setupMs: {
      main: roundMs(mainSetupMs),
      custom: roundMs(customSetupMs),
    },
    model: {
      main: {
        samplesMs: mainSamples.map(roundMs),
        medianMs: roundMs(mainMedianMs),
        realtimeFactor: roundRatio(audioSeconds / (mainMedianMs / 1000)),
      },
      custom: {
        samplesMs: customSamples.map(roundMs),
        medianMs: roundMs(customMedianMs),
        realtimeFactor: roundRatio(audioSeconds / (customMedianMs / 1000)),
      },
      speedup: roundRatio(mainMedianMs / customMedianMs),
    },
    parity,
  };
}

async function customRoundTrip(options) {
  await initWasm();
  const {
    bundleName,
    inputWavUrl,
    expectedEcdcUrl,
    referenceWavUrl,
    customEncoderRootUrl,
    customEncoderKernelUrl,
    customDecoderRootUrl,
    customDecoderKernelUrl,
  } = options;
  const bundleRoot = new URL(`../onnx-bundles/${bundleName}/`, window.location.href).href;
  const bundleJson = await fetchText(new URL("bundle.json", bundleRoot).href);
  const meta = JSON.parse(bundleJson);
  const lmWeights = new Uint8Array(
    await fetchArrayBuffer(new URL(meta.lm_quant_weight_model, bundleRoot).href),
  );
  const wav = decodeWav(await fetchArrayBuffer(inputWavUrl));
  if (wav.sampleRate !== meta.sample_rate || wav.channels !== meta.channels) {
    throw new Error("round-trip input WAV does not match the fixed bundle");
  }

  const encoderSetupStarted = performance.now();
  const encoder = await createCustomEncoder({
    assetRoot: customEncoderRootUrl,
    kernelModuleUrl: customEncoderKernelUrl,
    bundleMetadata: meta,
  });
  const encoderSetupMs = performance.now() - encoderSetupStarted;
  const segments = buildSegmentBatch(wav.audio, wav.frames, meta);
  const chunks = [lmEcdcFixedHeaderForWeights(bundleJson, wav.frames, 2, lmWeights)];
  let encodeModelMs = 0;
  let encodeEntropyMs = 0;
  try {
    for (let index = 0; index < segments.count; index += 1) {
      const segment = buildSingleSegment(wav.audio, wav.frames, segments, index, meta);
      const encoded = encoder.encode(segment.audio);
      encodeModelMs += encoded.elapsedMs;
      const frame = buildRawFrame(
        encoded.codes,
        new Float32Array([encoded.scale]),
        segment,
        meta,
        index,
      );
      const entropyStarted = performance.now();
      chunks.push(lmEcdcChunk(encodeQ8LmFrame(bundleJson, lmWeights, frame, meta)));
      encodeEntropyMs += performance.now() - entropyStarted;
    }
  } finally {
    encoder.release();
  }
  const encodedEcdc = concatUint8Chunks(chunks);
  const expectedEcdc = new Uint8Array(await fetchArrayBuffer(expectedEcdcUrl));
  const encodedParity = compareBytes(expectedEcdc, encodedEcdc);

  const entropyDecodeStarted = performance.now();
  const parsed = lmEcdcDecodeChunks(bundleJson, encodedEcdc);
  const frames = parsed.chunks.map((chunk) =>
    decodeQ8LmFrame(bundleJson, lmWeights, meta, chunk),
  );
  const decodeEntropyMs = performance.now() - entropyDecodeStarted;
  const decoderSetupStarted = performance.now();
  const decoder = await createCustomDecoder({
    assetRoot: customDecoderRootUrl,
    kernelModuleUrl: customDecoderKernelUrl,
    bundleMetadata: meta,
  });
  const decoderSetupMs = performance.now() - decoderSetupStarted;
  let decodedFrames;
  try {
    decodedFrames = decoder.decode(frames);
  } finally {
    decoder.release();
  }
  const metadata = ecdcMetadata(encodedEcdc);
  const decodedAudio = ecdcOverlapAddForMetadata(
    bundleJson,
    JSON.stringify(metadata),
    decodedFrames.audio,
  );
  const referenceWav = decodeWav(await fetchArrayBuffer(referenceWavUrl));
  const decodedParity = comparePcm(referenceWav.audio, decodedAudio);
  let nonFiniteSamples = 0;
  for (const sample of decodedAudio) {
    if (!Number.isFinite(sample)) nonFiniteSamples += 1;
  }
  return {
    runtime: "browser custom WASM (ONNX-free)",
    userAgent: navigator.userAgent,
    ortLoaded: ort !== null,
    bundleName,
    inputFrames: wav.frames,
    encodedFrames: frames.length,
    ecdcBytes: encodedEcdc.byteLength,
    encodedParity,
    decodedFrames: decodedAudio.length / meta.channels,
    nonFiniteSamples,
    decodedParity,
    timings: {
      encoderSetupMs: roundMs(encoderSetupMs),
      encodeModelMs: roundMs(encodeModelMs),
      encodeEntropyMs: roundMs(encodeEntropyMs),
      decodeEntropyMs: roundMs(decodeEntropyMs),
      decoderSetupMs: roundMs(decoderSetupMs),
      decodeModelMs: roundMs(decodedFrames.elapsedMs),
    },
  };
}

async function webGpuRoundTrip(options) {
  await initWasm();
  const {
    bundleName,
    inputWavUrl,
    expectedEcdcUrl = null,
    decodeEcdcUrl = null,
    referenceWavUrl = null,
    bundleRootUrl = new URL(
      `../dist/wasm-fixed-bundles/bundles/${bundleName}/`,
      window.location.href,
    ).href,
    downloadPrefix = null,
    onDecodedChunk = null,
  } = options;
  const bundleRoot = new URL(bundleRootUrl, window.location.href);
  const bundleJson = await fetchText(new URL("bundle.json", bundleRoot).href);
  const meta = JSON.parse(bundleJson);
  const lmWeights = new Uint8Array(
    await fetchArrayBuffer(new URL(meta.lm_quant_weight_model, bundleRoot).href),
  );
  const wav = decodeWav(await fetchArrayBuffer(inputWavUrl));
  if (wav.sampleRate !== meta.sample_rate || wav.channels !== meta.channels) {
    throw new Error("WebGPU input WAV does not match the fixed bundle");
  }
  const audioSeconds = wav.frames / meta.sample_rate;
  const segments = buildSegmentBatch(wav.audio, wav.frames, meta);
  const chunks = [lmEcdcFixedHeaderForWeights(bundleJson, wav.frames, 2, lmWeights)];
  const encodedFrames = [];
  const encodeScales = [];
  let minimumCode = Infinity;
  let maximumCode = -Infinity;

  reportWebGpuProgress("encoder setup", 0, segments.count);
  const encoder = await createWebGpuEncoder({
    assetRoot: new URL("encoder/", bundleRoot),
    bundleMetadata: meta,
  });
  let encodeModelMs = 0;
  let encodeEntropyMs = 0;
  try {
    for (let index = 0; index < segments.count; index += 1) {
      const segment = buildSingleSegment(wav.audio, wav.frames, segments, index, meta);
      const encoded = await encoder.encode(segment.audio);
      encodeModelMs += encoded.elapsedMs;
      encodeScales.push(encoded.scale);
      for (const code of encoded.codes) {
        minimumCode = Math.min(minimumCode, code);
        maximumCode = Math.max(maximumCode, code);
      }
      const frame = buildRawFrame(
        encoded.codes,
        new Float32Array([encoded.scale]),
        segment,
        meta,
        index,
      );
      encodedFrames.push(frame);
      const entropyStarted = performance.now();
      chunks.push(lmEcdcChunk(encodeQ8LmFrame(bundleJson, lmWeights, frame, meta)));
      encodeEntropyMs += performance.now() - entropyStarted;
      reportWebGpuProgress("encode", index + 1, segments.count);
      await yieldToPage();
    }
  } finally {
    encoder.release();
  }
  const encodedEcdc = concatUint8Chunks(chunks);
  const expectedEcdc = expectedEcdcUrl
    ? new Uint8Array(await fetchArrayBuffer(expectedEcdcUrl))
    : null;
  const encodedParity = expectedEcdc ? compareBytes(expectedEcdc, encodedEcdc) : null;
  const encodedFrameParity = expectedEcdc
    ? compareEncodedFrames(
      encodedFrames,
      lmEcdcDecodeChunks(bundleJson, expectedEcdc).chunks.map((chunk) =>
        decodeQ8LmFrame(bundleJson, lmWeights, meta, chunk)),
      meta,
    )
    : null;

  const decodeEcdc = decodeEcdcUrl
    ? new Uint8Array(await fetchArrayBuffer(decodeEcdcUrl))
    : encodedEcdc;

  const decodeRequestStarted = performance.now();
  reportWebGpuProgress("decoder setup", 0, segments.count);
  const decoder = await createWebGpuEcdcDecoder({
    encodecModule: {
      lmEcdcDecodeChunks,
      ecdcMetadata,
      QuantizedLmChunkDecoder,
    },
    bundleJson,
    lmWeights,
    assetRoot: new URL("decoder/", bundleRoot),
  });
  let firstPlayableChunkColdMs = null;
  let decoded;
  try {
    decoded = await decoder.decode(decodeEcdc, {
      collectAudio: true,
      async onChunk(chunk) {
        if (firstPlayableChunkColdMs === null) {
          firstPlayableChunkColdMs = performance.now() - decodeRequestStarted;
        }
        if (onDecodedChunk) await onDecodedChunk(chunk);
      },
      onProgress(progress) {
        reportWebGpuProgress("decode", progress.completed, progress.total);
      },
      yieldControl: yieldToPage,
    });
  } finally {
    decoder.release();
  }
  const decodedAudio = decoded.audio;
  const referenceWav = referenceWavUrl
    ? decodeWav(await fetchArrayBuffer(referenceWavUrl))
    : wav;
  const decodedParity = comparePcm(referenceWav.audio, decodedAudio);
  if (downloadPrefix) {
    const encodedWav = writeWavBytes(decodedAudio, meta.channels, meta.sample_rate);
    downloadBytes(`${downloadPrefix}.ecdc`, encodedEcdc, "application/octet-stream");
    downloadBytes(`${downloadPrefix}.decoded.wav`, encodedWav, "audio/wav");
  }
  reportWebGpuProgress("complete", decoded.chunkCount, decoded.chunkCount);
  return {
    runtime: "Mobile Safari WebGPU (ONNX-free)",
    userAgent: navigator.userAgent,
    bundleName,
    inputFrames: wav.frames,
    audioSeconds,
    chunks: decoded.chunkCount,
    ecdcBytes: encodedEcdc.byteLength,
    encodedParity,
    encodedFrameParity,
    decodeSource: decodeEcdcUrl ? "external ECDC" : "WebGPU encode",
    encodeDiagnostics: {
      scales: encodeScales,
      minimumCode,
      maximumCode,
    },
    decodedParity,
    nonFiniteSamples: countNonFinite(decodedAudio),
    setupMs: {
      encoder: roundMs(encoder.setupMs),
      decoder: roundMs(decoder.setupMs),
    },
    timings: {
      encodeModelMs: roundMs(encodeModelMs),
      encodeEntropyMs: roundMs(encodeEntropyMs),
      decodeContainerMs: roundMs(decoded.timings.containerMs),
      decodeEntropyMs: roundMs(decoded.timings.entropyMs),
      decodeModelMs: roundMs(decoded.timings.modelMs),
      decodeAssemblyMs: roundMs(decoded.timings.assemblyMs),
      decodeDeliveryMs: roundMs(decoded.timings.deliveryMs),
      decodePipelineMs: roundMs(decoded.timings.pipelineMs),
      firstPlayableChunkMs: roundMs(decoded.timings.firstChunkMs),
      firstPlayableChunkColdMs: roundMs(firstPlayableChunkColdMs),
    },
    realtimeFactor: {
      encode: roundRatio(audioSeconds / ((encodeModelMs + encodeEntropyMs) / 1000)),
      decode: roundRatio(audioSeconds / (decoded.timings.pipelineMs / 1000)),
    },
    adapter: decoder.adapter,
  };
}

async function webGpuDecodeEcdc(options) {
  await initWasm();
  const {
    bundleName,
    ecdcUrl,
    referenceWavUrl = null,
    bundleRootUrl = new URL(
      `../dist/wasm-fixed-bundles/bundles/${bundleName}/`,
      window.location.href,
    ).href,
    collectAudio = Boolean(referenceWavUrl),
    onDecodedChunk = null,
  } = options;
  const bundleRoot = new URL(bundleRootUrl, window.location.href);
  const [bundleJson, ecdcBuffer, referenceBuffer] = await Promise.all([
    fetchText(new URL("bundle.json", bundleRoot).href),
    fetchArrayBuffer(ecdcUrl),
    referenceWavUrl ? fetchArrayBuffer(referenceWavUrl) : Promise.resolve(null),
  ]);
  const meta = JSON.parse(bundleJson);
  const lmWeights = new Uint8Array(
    await fetchArrayBuffer(new URL(meta.lm_quant_weight_model, bundleRoot).href),
  );
  const ecdc = new Uint8Array(ecdcBuffer);
  const decodeRequestStarted = performance.now();
  reportWebGpuProgress("decoder setup", 0, 1);
  const decoder = await createWebGpuEcdcDecoder({
    encodecModule: {
      lmEcdcDecodeChunks,
      ecdcMetadata,
      QuantizedLmChunkDecoder,
    },
    bundleJson,
    lmWeights,
    assetRoot: new URL("decoder/", bundleRoot),
  });
  let firstPlayableChunkColdMs = null;
  let deliveredChunks = 0;
  let deliveredFrames = 0;
  let nonFiniteSamples = 0;
  let decoded;
  try {
    decoded = await decoder.decode(ecdc, {
      collectAudio,
      async onChunk(chunk) {
        if (firstPlayableChunkColdMs === null) {
          firstPlayableChunkColdMs = performance.now() - decodeRequestStarted;
        }
        deliveredChunks += 1;
        deliveredFrames += chunk.samples;
        nonFiniteSamples += countNonFinite(chunk.pcm);
        if (onDecodedChunk) await onDecodedChunk(chunk);
      },
      onProgress(progress) {
        reportWebGpuProgress("decode", progress.completed, progress.total);
      },
      yieldControl: yieldToPage,
    });
  } finally {
    decoder.release();
  }

  const audioSeconds = decoded.audioLength / meta.sample_rate;
  const referenceWav = referenceBuffer ? decodeWav(referenceBuffer) : null;
  const decodedParity = referenceWav && decoded.audio
    ? comparePcm(referenceWav.audio, decoded.audio)
    : null;
  reportWebGpuProgress("complete", decoded.chunkCount, decoded.chunkCount);
  return {
    runtime: "Mobile Safari WebGPU incremental decode (ONNX-free)",
    userAgent: navigator.userAgent,
    bundleName,
    audioSeconds,
    chunks: decoded.chunkCount,
    ecdcBytes: ecdc.byteLength,
    collectedAudio: Boolean(decoded.audio),
    deliveredChunks,
    deliveredFrames,
    nonFiniteSamples,
    decodedParity,
    setupMs: roundMs(decoder.setupMs),
    timings: {
      decodeContainerMs: roundMs(decoded.timings.containerMs),
      decodeEntropyMs: roundMs(decoded.timings.entropyMs),
      decodeModelMs: roundMs(decoded.timings.modelMs),
      decodeAssemblyMs: roundMs(decoded.timings.assemblyMs),
      decodeDeliveryMs: roundMs(decoded.timings.deliveryMs),
      decodePipelineMs: roundMs(decoded.timings.pipelineMs),
      firstPlayableChunkMs: roundMs(decoded.timings.firstChunkMs),
      firstPlayableChunkColdMs: roundMs(firstPlayableChunkColdMs),
    },
    realtimeFactor: roundRatio(audioSeconds / (decoded.timings.pipelineMs / 1000)),
    adapter: decoder.adapter,
  };
}

function reportWebGpuProgress(phase, completed, total) {
  const detail = { phase, completed, total };
  window.dispatchEvent(new CustomEvent("encodec-webgpu-progress", { detail }));
  console.log(`${phase} ${completed}/${total}`);
}

function yieldToPage() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function countNonFinite(values) {
  let count = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) count += 1;
  }
  return count;
}

function compareEncodedFrames(candidate, reference, meta) {
  if (candidate.length !== reference.length) {
    return {
      comparable: false,
      candidateFrames: candidate.length,
      referenceFrames: reference.length,
    };
  }
  let comparedCodes = 0;
  let codeMismatches = 0;
  let firstCodeMismatch = null;
  let scaleMismatches = 0;
  let maximumScaleError = 0;
  for (let frame = 0; frame < candidate.length; frame += 1) {
    const left = candidate[frame];
    const right = reference[frame];
    const frameLength = Math.min(left.frameLength, right.frameLength);
    const scaleError = Math.abs(Number(left.scale) - Number(right.scale));
    maximumScaleError = Math.max(maximumScaleError, scaleError);
    if (scaleError !== 0) scaleMismatches += 1;
    for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
      for (let step = 0; step < frameLength; step += 1) {
        const index = codebook * meta.frame_length + step;
        comparedCodes += 1;
        if (left.codes[index] === right.codes[index]) continue;
        codeMismatches += 1;
        firstCodeMismatch ??= {
          frame,
          codebook,
          step,
          candidate: left.codes[index],
          reference: right.codes[index],
        };
      }
    }
  }
  return {
    comparable: true,
    exact: codeMismatches === 0 && scaleMismatches === 0,
    comparedCodes,
    codeMismatches,
    firstCodeMismatch,
    scaleMismatches,
    maximumScaleError,
  };
}

async function customRuntimeChunk(options) {
  await initWasm();
  const {
    bundleRootUrl,
    runtimeModuleUrl,
    runtimeRootUrl,
    inputWavUrl,
    expectedEcdcUrl,
    neuralBackend = "auto",
  } = options;
  const bundleJson = await fetchText(new URL("bundle.json", bundleRootUrl).href);
  const meta = JSON.parse(bundleJson);
  const lmWeights = new Uint8Array(
    await fetchArrayBuffer(new URL(meta.lm_quant_weight_model, bundleRootUrl).href),
  );
  const wav = decodeWav(await fetchArrayBuffer(inputWavUrl));
  const segments = buildSegmentBatch(wav.audio, wav.frames, meta);
  const segment = buildSingleSegment(wav.audio, wav.frames, segments, 0, meta);
  const module = await import(runtimeModuleUrl);
  const runtime = module.createEncodecEcdcRuntime({
    encodecWasmBaseUrl: runtimeRootUrl,
    neuralBackend,
  });
  const chunk = await runtime.encodeEcdcChunk({
    sessionKey: "production-runtime-smoke",
    bundleRoot: bundleRootUrl,
    bundleJson,
    lmWeights,
    segment: segment.audio,
    segmentIndex: 0,
    segmentSamples: segment.samples,
    segmentFrameLength: segment.frameLength,
  });
  const expectedEcdc = new Uint8Array(await fetchArrayBuffer(expectedEcdcUrl));
  const expectedFrames = lmEcdcDecodeChunks(bundleJson, expectedEcdc);
  const expectedChunk = lmEcdcChunk(expectedFrames.chunks[0].payload);
  const parity = compareBytes(expectedChunk, chunk);
  const candidateLength = new DataView(
    chunk.buffer,
    chunk.byteOffset,
    chunk.byteLength,
  ).getUint32(0, false);
  if (candidateLength + 8 !== chunk.byteLength) {
    throw new Error("Production runtime returned an invalid framed ECDC chunk");
  }
  const candidateFrame = decodeQ8LmFrame(bundleJson, lmWeights, meta, {
    ...expectedFrames.chunks[0],
    payload: chunk.subarray(8),
  });
  const referenceFrame = decodeQ8LmFrame(
    bundleJson,
    lmWeights,
    meta,
    expectedFrames.chunks[0],
  );
  const frameParity = compareEncodedFrames([candidateFrame], [referenceFrame], meta);
  const diagnosticsBeforeRelease = runtime.diagnostics();
  runtime.releaseJobState("production-runtime-smoke");
  await runtime.clearSessionCache();
  return {
    runtime: "browser production selector",
    ortLoaded: ort !== null,
    parity,
    frameParity,
    diagnosticsBeforeRelease,
    diagnosticsAfterRelease: runtime.diagnostics(),
  };
}

async function initWasm() {
  if (!wasmReady) {
    wasmReady = init(new URL("../pkg/encodec_rs_bg.wasm?v=q8-webgpu-matrix", window.location.href).href).then(() => {
      initPanicHook();
    });
  }
  return wasmReady;
}

async function getSession(key, modelUrl) {
  await ensureOrt();
  const cached = sessionCache.get(key);
  if (cached) {
    return cached;
  }
  const model = new Uint8Array(await fetchArrayBuffer(modelUrl));
  const session = await ort.InferenceSession.create(model, {
    executionProviders: matrixExecutionProviders(),
    graphOptimizationLevel: "all",
  });
  sessionCache.set(key, session);
  return session;
}

function encodeQ8LmFrame(bundleJson, weights, frame, meta) {
  const encoder = new QuantizedLmChunkEncoder(bundleJson, weights, frame.scale);
  try {
    for (let step = 0; step < frame.frameLength; step += 1) {
      encoder.push(frameStepCodes(frame, meta, step));
    }
    return encoder.finish();
  } finally {
    encoder.free();
  }
}

function decodeQ8LmFrame(bundleJson, weights, meta, chunk) {
  const decoder = new QuantizedLmChunkDecoder(bundleJson, weights, Uint8Array.from(chunk.payload));
  try {
    const codes = new Uint16Array(meta.num_codebooks * meta.frame_length);
    if (typeof decoder.pullAll === "function") {
      const activeCodes = decoder.pullAll(chunk.frameLength);
      for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
        const sourceOffset = codebook * chunk.frameLength;
        const targetOffset = codebook * meta.frame_length;
        codes.set(
          activeCodes.subarray(sourceOffset, sourceOffset + chunk.frameLength),
          targetOffset,
        );
      }
    } else {
      for (let step = 0; step < chunk.frameLength; step += 1) {
        const symbols = decoder.pull();
        for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
          codes[codebook * meta.frame_length + step] = symbols[codebook];
        }
      }
    }
    return {
      offset: chunk.offset,
      samples: chunk.samples,
      frameLength: chunk.frameLength,
      scale: decoder.scale(),
      codes,
    };
  } finally {
    decoder.free();
  }
}

async function decodeFrameBatch(session, frames, meta, batchSize = 32) {
  const samplesPerDecodedFrame = meta.channels * meta.segment_samples;
  const audio = new Float32Array(frames.length * samplesPerDecodedFrame);
  let decodeOnnxMs = 0;
  for (let start = 0; start < frames.length; start += batchSize) {
    const end = Math.min(start + batchSize, frames.length);
    const batch = frames.slice(start, end);
    const decoderInputs = buildDecoderInputs(batch, meta);
    const decodeStarted = performance.now();
    const outputs = await session.run({
      [session.inputNames[0]]: new ort.Tensor("int64", decoderInputs.codes, [
        batch.length,
        meta.num_codebooks,
        meta.frame_length,
      ]),
      [session.inputNames[1]]: new ort.Tensor("float32", decoderInputs.scales, [batch.length, 1]),
    });
    decodeOnnxMs += performance.now() - decodeStarted;
    const decodedTensor = findDecodeOutput(outputs);
    const expectedLength = batch.length * samplesPerDecodedFrame;
    if (decodedTensor.data.length < expectedLength) {
      throw new Error(
        `decoded batch ${start}-${end} returned ${decodedTensor.data.length} samples, expected ${expectedLength}`,
      );
    }
    const targetOffset = start * samplesPerDecodedFrame;
    if (targetOffset + expectedLength > audio.length) {
      throw new Error(
        `decoded batch ${start}-${end} overflows ${audio.length} samples at offset ${targetOffset}`,
      );
    }
    audio.set(decodedTensor.data.subarray(0, expectedLength), targetOffset);
    console.log(`frame decode batch ${end}/${frames.length}`);
  }
  return {
    audio,
    decodeOnnxMs,
    shape: [frames.length, meta.channels, meta.segment_samples],
  };
}

function decodeWav(bytes) {
  const view = new DataView(bytes);
  if (readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") {
    throw new Error("input is not a RIFF/WAVE file");
  }
  let offset = 12;
  let fmt = null;
  let dataOffset = 0;
  let dataSize = 0;
  while (offset + 8 <= view.byteLength) {
    const id = readAscii(view, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      const formatTag = view.getUint16(body, true);
      const channels = view.getUint16(body + 2, true);
      const sampleRate = view.getUint32(body + 4, true);
      const bitsPerSample = view.getUint16(body + 14, true);
      const subFormatTag = formatTag === 0xfffe && size >= 40 ? view.getUint32(body + 24, true) : formatTag;
      fmt = { subFormatTag, channels, sampleRate, bitsPerSample };
    } else if (id === "data") {
      dataOffset = body;
      dataSize = size;
    }
    offset = body + size + (size % 2);
  }
  if (!fmt || !dataOffset || !dataSize) {
    throw new Error("WAV is missing fmt or data chunk");
  }
  const bytesPerSample = fmt.bitsPerSample / 8;
  const frames = Math.floor(dataSize / (fmt.channels * bytesPerSample));
  const audio = new Float32Array(fmt.channels * frames);
  let cursor = dataOffset;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < fmt.channels; channel += 1) {
      let sample;
      if (fmt.subFormatTag === 1 && fmt.bitsPerSample === 16) {
        sample = view.getInt16(cursor, true) / 32768;
      } else if (fmt.subFormatTag === 1 && fmt.bitsPerSample === 24) {
        const raw =
          view.getUint8(cursor) |
          (view.getUint8(cursor + 1) << 8) |
          (view.getUint8(cursor + 2) << 16);
        const signed = raw & 0x800000 ? raw | 0xff000000 : raw;
        sample = signed / 8388608;
      } else if (fmt.subFormatTag === 3 && fmt.bitsPerSample === 32) {
        sample = view.getFloat32(cursor, true);
      } else {
        throw new Error(`unsupported WAV format: subFormat=${fmt.subFormatTag} bits=${fmt.bitsPerSample}`);
      }
      audio[channel * frames + frame] = sample;
      cursor += bytesPerSample;
    }
  }
  return {
    sampleRate: fmt.sampleRate,
    channels: fmt.channels,
    frames,
    audio,
  };
}

function writeWavBytes(planar, channels, sampleRate) {
  const frames = Math.floor(planar.length / channels);
  const bytesPerSample = 2;
  const dataBytes = frames * channels * bytesPerSample;
  const out = new Uint8Array(44 + dataBytes);
  const view = new DataView(out.buffer);
  writeAscii(out, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(out, 8, "WAVE");
  writeAscii(out, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeAscii(out, 36, "data");
  view.setUint32(40, dataBytes, true);
  let cursor = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const value = Math.max(-1, Math.min(1, planar[channel * frames + frame]));
      view.setInt16(
        cursor,
        value < 0 ? Math.round(value * 32768) : Math.round(value * 32767),
        true,
      );
      cursor += bytesPerSample;
    }
  }
  return out;
}

function buildSegmentBatch(audio, audioLength, meta) {
  const context = fixedContextSamples(meta);
  const stride = Math.max(1, meta.segment_stride);
  const starts = [];
  for (let offset = 0; offset < audioLength; offset += stride) {
    starts.push(offset);
  }
  return {
    starts,
    context,
    frameLengths: starts.map((offset) => context === null
      ? Math.ceil((Math.min(audioLength - offset, meta.segment_samples) * meta.frame_length) / meta.segment_samples)
      : meta.frame_length),
    count: starts.length,
  };
}

function buildSingleSegment(audio, audioLength, segments, index, meta) {
  const offset = segments.starts[index];
  const context = segments.context ?? 0;
  const samples = Math.min(
    audioLength - offset,
    segments.context === null ? meta.segment_samples : meta.segment_stride,
  );
  const segment = new Float32Array(meta.channels * meta.segment_samples);
  for (let channel = 0; channel < meta.channels; channel += 1) {
    const sourceBase = channel * audioLength;
    const targetBase = channel * meta.segment_samples;
    for (let modelIndex = 0; modelIndex < meta.segment_samples; modelIndex += 1) {
      const sourceIndex = offset - context + modelIndex;
      if (sourceIndex >= 0 && sourceIndex < audioLength) {
        segment[targetBase + modelIndex] = audio[sourceBase + sourceIndex];
      }
    }
  }
  return {
    audio: segment,
    offset,
    samples,
    frameLength: segments.frameLengths[index],
  };
}

function fixedContextSamples(meta) {
  const samples = Number(meta.segment_samples);
  const stride = Number(meta.segment_stride);
  if (
    (samples === 64_960 && stride === 64_000)
    || (samples === 87_360 && stride === 86_400)
  ) {
    return 480;
  }
  return null;
}

function buildRawFrame(codes, scales, segment, meta, segmentIndex) {
  const valuesPerSegment = meta.num_codebooks * meta.frame_length;
  if (codes.length !== valuesPerSegment) {
    throw new Error(`segment ${segmentIndex} codes length ${codes.length} does not match ${valuesPerSegment}`);
  }
  const frameCodes = new Uint16Array(valuesPerSegment);
  for (let index = 0; index < valuesPerSegment; index += 1) {
    frameCodes[index] = toU16Code(codes[index], segmentIndex * valuesPerSegment + index);
  }
  return {
    offset: segment.offset,
    samples: segment.samples,
    frameLength: segment.frameLength,
    scale: Number(scales[0] ?? 1),
    codes: frameCodes,
  };
}

function buildDecoderInputs(frames, meta) {
  const valuesPerSegment = meta.num_codebooks * meta.frame_length;
  const codes = new BigInt64Array(frames.length * valuesPerSegment);
  const scales = new Float32Array(frames.length);
  for (let batchIndex = 0; batchIndex < frames.length; batchIndex += 1) {
    const frame = frames[batchIndex];
    const base = batchIndex * valuesPerSegment;
    for (let index = 0; index < valuesPerSegment; index += 1) {
      codes[base + index] = BigInt(frame.codes[index]);
    }
    scales[batchIndex] = Number(frame.scale ?? 1);
  }
  return { codes, scales };
}

function frameStepCodes(frame, meta, step) {
  const codes = new Uint16Array(meta.num_codebooks);
  for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
    codes[codebook] = frame.codes[codebook * meta.frame_length + step];
  }
  return codes;
}

function findEncodeOutputs(outputs) {
  const tensors = Object.values(outputs);
  const codesTensor = tensors.find((tensor) => tensor.type === "int64");
  const scaleTensor = tensors.find((tensor) => tensor.type === "float32" && tensor.dims.length === 2);
  if (!codesTensor || !scaleTensor) {
    throw new Error(`unexpected encoder outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return { codesTensor, scaleTensor };
}

function findDecodeOutput(outputs) {
  const tensor = Object.values(outputs).find(
    (candidate) => candidate.type === "float32" && candidate.dims.length === 3,
  );
  if (!tensor) {
    throw new Error(`unexpected decoder outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return tensor;
}

function summarizeOutputs(outputs) {
  return Object.fromEntries(Object.entries(outputs).map(([name, tensor]) => [name, {
    type: tensor.type,
    dims: tensor.dims,
    length: tensor.data.length,
  }]));
}

async function fetchText(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`fetch ${url} failed: ${response.status}`);
  }
  return response.text();
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`fetch ${url} failed: ${response.status}`);
  }
  return response.arrayBuffer();
}

function downloadBytes(name, bytes, type) {
  const blob = new Blob([bytes], { type });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function concatUint8Chunks(chunks) {
  const byteLength = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const out = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

function readAscii(view, offset, length) {
  let out = "";
  for (let index = 0; index < length; index += 1) {
    out += String.fromCharCode(view.getUint8(offset + index));
  }
  return out;
}

function matrixExecutionProviders() {
  if (new URLSearchParams(window.location.search).get("provider") === "wasm") {
    return ["wasm"];
  }
  const providers = globalThis.WEBGPU_MATRIX_EXECUTION_PROVIDERS;
  if (Array.isArray(providers) && providers.length) {
    return providers.map((provider) => String(provider)).filter(Boolean);
  }
  return ["webgpu", "wasm"];
}

function comparePcm(reference, candidate) {
  if (reference.length !== candidate.length) {
    return {
      comparable: false,
      referenceLength: reference.length,
      candidateLength: candidate.length,
    };
  }
  let exactMismatches = 0;
  let signal = 0;
  let noise = 0;
  let maxAbsError = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const expected = reference[index];
    const actual = candidate[index];
    const difference = actual - expected;
    if (!Object.is(expected, actual)) exactMismatches += 1;
    signal += expected * expected;
    noise += difference * difference;
    maxAbsError = Math.max(maxAbsError, Math.abs(difference));
  }
  return {
    comparable: true,
    exact: exactMismatches === 0,
    exactMismatches,
    maxAbsError,
    maxErrorDbfs: maxAbsError === 0 ? null : roundRatio(20 * Math.log10(maxAbsError)),
    rmse: Math.sqrt(noise / reference.length),
    snrDb: noise === 0 ? null : roundRatio(10 * Math.log10(signal / noise)),
  };
}

function compareBytes(reference, candidate) {
  const length = Math.min(reference.length, candidate.length);
  let mismatches = Math.abs(reference.length - candidate.length);
  let firstMismatch = null;
  for (let index = 0; index < length; index += 1) {
    if (reference[index] !== candidate[index]) {
      mismatches += 1;
      if (firstMismatch === null) firstMismatch = index;
    }
  }
  return {
    exact: mismatches === 0,
    mismatches,
    firstMismatch,
    referenceBytes: reference.length,
    candidateBytes: candidate.length,
  };
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function writeAscii(out, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    out[offset + index] = text.charCodeAt(index);
  }
}

function toU16Code(raw, index) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0 || value > 65535) {
    throw new Error(`invalid code at ${index}: ${String(raw)}`);
  }
  return value;
}

function roundMs(ms) {
  return Number(ms.toFixed(1));
}

function roundRatio(value) {
  return Number(value.toFixed(3));
}
