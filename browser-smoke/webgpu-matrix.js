import * as ort from "./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs";
import init, {
  initPanicHook,
  lmEcdcChunk,
  lmEcdcDecodeChunks,
  lmEcdcHeaderForWeights,
  QuantizedLmChunkDecoder,
  QuantizedLmChunkEncoder,
  ecdcMetadata,
  ecdcOverlapAdd,
  stableHashHex,
} from "../pkg/encodec_rs.js";

ort.env.wasm.wasmPaths = new URL("./node_modules/onnxruntime-web/dist/", window.location.href).href;
ort.env.wasm.numThreads = 1;
if (ort.env.webgpu) {
  ort.env.webgpu.powerPreference = "high-performance";
}

let wasmReady;
const sessionCache = new Map();

window.webgpuMatrix = {
  ready,
  encode,
  decode,
};

async function ready() {
  if (!navigator.gpu) {
    throw new Error(`navigator.gpu is unavailable in ${navigator.userAgent}`);
  }
  await initWasm();
  return {
    userAgent: navigator.userAgent,
    secureContext: window.isSecureContext,
    hasWebGpu: Boolean(navigator.gpu),
    ortWebVersion: ort.env?.versions?.web ?? null,
  };
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
  const chunks = [lmEcdcHeaderForWeights(bundleJson, wav.frames, 2, weights)];
  let frameOnnxMs = 0;
  let lmMs = 0;

  for (let index = 0; index < segments.count; index += 1) {
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
    const started = performance.now();
    frames.push(decodeQ8LmFrame(bundleJson, weights, meta, chunk));
    lmMs += performance.now() - started;
  }

  const decodeSession = await getSession(`${bundleName}:decode`, new URL(meta.decode_model, bundleRoot).href);
  const decodedFrames = await decodeFrameBatch(decodeSession, frames, meta);
  const audioLength = metadata.al ?? metadata.audio_length;
  const decodedAudio = ecdcOverlapAdd(bundleJson, audioLength, decodedFrames.audio);
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

async function initWasm() {
  if (!wasmReady) {
    wasmReady = init(new URL("../pkg/encodec_rs_bg.wasm?v=q8-webgpu-matrix", window.location.href).href).then(() => {
      initPanicHook();
    });
  }
  return wasmReady;
}

async function getSession(key, modelUrl) {
  const cached = sessionCache.get(key);
  if (cached) {
    return cached;
  }
  const model = new Uint8Array(await fetchArrayBuffer(modelUrl));
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["webgpu", "wasm"],
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
    for (let step = 0; step < chunk.frameLength; step += 1) {
      const symbols = decoder.pull();
      for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
        codes[codebook * meta.frame_length + step] = symbols[codebook];
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
    audio.set(decodedTensor.data, start * samplesPerDecodedFrame);
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
      const value = Math.max(-0.99, Math.min(0.99, planar[channel * frames + frame]));
      view.setInt16(cursor, Math.round(value * 32767), true);
      cursor += bytesPerSample;
    }
  }
  return out;
}

function buildSegmentBatch(audio, audioLength, meta) {
  const starts = [];
  for (let offset = 0; offset < audioLength; offset += Math.max(1, meta.segment_stride)) {
    starts.push(offset);
  }
  return {
    starts,
    frameLengths: starts.map((offset) =>
      Math.ceil((Math.min(audioLength - offset, meta.segment_samples) * meta.frame_length) / meta.segment_samples),
    ),
    count: starts.length,
  };
}

function buildSingleSegment(audio, audioLength, segments, index, meta) {
  const offset = segments.starts[index];
  const samples = Math.min(audioLength - offset, meta.segment_samples);
  const segment = new Float32Array(meta.channels * meta.segment_samples);
  for (let channel = 0; channel < meta.channels; channel += 1) {
    const sourceBase = channel * audioLength + offset;
    const targetBase = channel * meta.segment_samples;
    for (let t = 0; t < samples; t += 1) {
      segment[targetBase + t] = audio[sourceBase + t];
    }
  }
  return {
    audio: segment,
    offset,
    samples,
    frameLength: segments.frameLengths[index],
  };
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
