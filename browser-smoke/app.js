import * as ort from "./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";
import init, {
  initPanicHook,
  rawEcdcDecodeFrames,
  rawEcdcEncode,
  rawEcdcMetadata,
  rawEcdcOverlapAdd,
} from "../pkg/encodec_rs.js";

ort.env.wasm.wasmPaths = new URL(
  "./node_modules/onnxruntime-web/dist/",
  window.location.href,
).href;
ort.env.wasm.numThreads = 1;

const els = {
  bundle: document.querySelector("#bundle"),
  decodePlayback: document.querySelector("#decode-playback"),
  run: document.querySelector("#run"),
  status: document.querySelector("#status"),
  model: document.querySelector("#model"),
  input: document.querySelector("#input"),
  codes: document.querySelector("#codes"),
  scale: document.querySelector("#scale"),
  ecdc: document.querySelector("#ecdc"),
  elapsed: document.querySelector("#elapsed"),
  log: document.querySelector("#log"),
};

let wasmReady;
let sessionCache = new Map();
let audioContext;

els.run.addEventListener("click", () => {
  runEncodeSmoke().catch((error) => {
    setStatus(error?.message ?? String(error), "error");
    writeLog(error?.stack ?? String(error));
  });
});

async function runEncodeSmoke() {
  els.run.disabled = true;
  clearMetrics();
  setStatus("Loading wasm and ONNX assets");
  const started = performance.now();
  const shouldDecode = els.decodePlayback.checked;
  if (shouldDecode) {
    await prepareAudioContext();
  }

  try {
    await initWasm();
    const bundleName = els.bundle.value;
    const bundleRoot = `../onnx-bundles/${bundleName}`;
    const bundleJson = await fetchText(`${bundleRoot}/bundle.json`);
    const meta = JSON.parse(bundleJson);
    const sample = await loadJfkAudio(meta);
    const segments = buildSegmentBatch(sample.audio, sample.audioLength, meta);
    setStatus(`Running encode_frame.onnx on ${segments.count} JFK segments`);

    const session = await getSession(bundleName, `${bundleRoot}/${meta.encode_model}`);
    const feeds = {
      [session.inputNames[0]]: new ort.Tensor("float32", segments.audio, [
        segments.count,
        meta.channels,
        meta.segment_samples,
      ]),
    };
    const outputs = await session.run(feeds);
    const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
    const frames = buildRawFrames(codesTensor.data, scaleTensor.data, segments, meta);
    const ecdc = rawEcdcEncode(bundleJson, sample.audioLength, frames);
    const ecdcMeta = rawEcdcMetadata(ecdc);
    let decodeSummary = null;
    if (shouldDecode) {
      decodeSummary = await decodeAndPlay(bundleName, bundleRoot, bundleJson, meta, ecdc, sample.audioLength);
    }
    const elapsed = performance.now() - started;

    els.model.textContent = `${meta.model_name} ${meta.bandwidth_kbps} kbps`;
    els.input.textContent = `${sample.source.sampleRate} Hz ${sample.source.channels}ch -> [${segments.count}, ${meta.channels}, ${meta.segment_samples}]`;
    els.codes.textContent = `${codesTensor.type} [${codesTensor.dims.join(", ")}]`;
    els.scale.textContent = `${scaleTensor.type} [${scaleTensor.dims.join(", ")}]`;
    els.ecdc.textContent = `${ecdc.byteLength} bytes, ${sample.audioLength} samples, acv=${ecdcMeta.acv ?? ecdcMeta.bitstream_version ?? 0}`;
    els.elapsed.textContent = `${elapsed.toFixed(1)} ms`;

    writeLog(
      JSON.stringify(
        {
          sessionInputs: session.inputNames,
          sessionOutputs: session.outputNames,
          source: sample.source,
          packagedAudio: {
            audioLength: sample.audioLength,
            segments: segments.count,
            segmentSamples: meta.segment_samples,
            segmentStride: meta.segment_stride,
            firstFrame: summarizeFrame(frames[0]),
            lastFrame: summarizeFrame(frames[frames.length - 1]),
          },
          outputSummary: summarizeOutputs(outputs),
          decode: decodeSummary,
          ecdcMetadata: ecdcMeta,
          firstCodes: Array.from(frames[0].codes.slice(0, Math.min(16, frames[0].codes.length))),
          ecdcPrefix: Array.from(ecdc.slice(0, Math.min(32, ecdc.length))),
        },
        null,
        2,
      ),
    );
    setStatus(shouldDecode ? "Encode/decode smoke passed and playback started" : "Encode smoke passed", "ok");
  } finally {
    els.run.disabled = false;
  }
}

async function decodeAndPlay(bundleName, bundleRoot, bundleJson, meta, ecdc, audioLength) {
  setStatus("Parsing raw ECDC frames in wasm");
  const parsed = rawEcdcDecodeFrames(bundleJson, ecdc);
  const frames = parsed.frames;
  const decoderInputs = buildDecoderInputs(frames, meta);
  setStatus(`Running decode_frame.onnx on ${frames.length} JFK segments`);

  const session = await getSession(`${bundleName}:decode`, `${bundleRoot}/${meta.decode_model}`);
  const feeds = {
    [session.inputNames[0]]: new ort.Tensor("int64", decoderInputs.codes, [
      frames.length,
      meta.num_codebooks,
      meta.frame_length,
    ]),
    [session.inputNames[1]]: new ort.Tensor("float32", decoderInputs.scales, [frames.length, 1]),
  };
  const outputs = await session.run(feeds);
  const decodedTensor = findDecodeOutput(outputs);

  setStatus("Overlap-adding decoded frames in wasm");
  const decodedAudio = rawEcdcOverlapAdd(bundleJson, audioLength, decodedTensor.data);
  playPlanarAudio(decodedAudio, meta.channels, meta.sample_rate);

  return {
    parsedFrames: frames.length,
    decoderInputs: session.inputNames,
    decoderOutputs: session.outputNames,
    decodedShape: decodedTensor.dims,
    playbackSeconds: Number((audioLength / meta.sample_rate).toFixed(3)),
  };
}

async function initWasm() {
  if (!wasmReady) {
    wasmReady = init("../pkg/encodec_rs_bg.wasm").then(() => {
      initPanicHook();
    });
  }
  return wasmReady;
}

async function getSession(bundleName, modelPath) {
  const cacheKey = `${bundleName}:${modelPath}`;
  if (!sessionCache.has(cacheKey)) {
    sessionCache.set(
      cacheKey,
      ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
    );
  }
  return sessionCache.get(cacheKey);
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.text();
}

async function loadJfkAudio(meta) {
  const wav = await fetchArrayBuffer("../mel-spec/jfk_f32le.wav");
  const decoded = decodeWav(wav);
  const audio = resampleToPlanar(decoded, meta);
  return {
    audio,
    audioLength: audio.length / meta.channels,
    source: {
      path: "mel-spec/testdata/jfk_f32le.wav",
      sampleRate: decoded.sampleRate,
      channels: decoded.channels,
      frames: decoded.frames,
      seconds: Number((decoded.frames / decoded.sampleRate).toFixed(3)),
    },
  };
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status}`);
  }
  return response.arrayBuffer();
}

function decodeWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") {
    throw new Error("JFK sample is not a RIFF/WAVE file");
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
      fmt = { formatTag, subFormatTag, channels, sampleRate, bitsPerSample };
    } else if (id === "data") {
      dataOffset = body;
      dataSize = size;
    }
    offset = body + size + (size % 2);
  }

  if (!fmt || !dataOffset || !dataSize) {
    throw new Error("JFK WAV is missing fmt or data chunk");
  }
  if (fmt.bitsPerSample !== 32 || fmt.subFormatTag !== 3) {
    throw new Error(
      `Expected 32-bit float WAV, got format=${fmt.formatTag} subFormat=${fmt.subFormatTag} bits=${fmt.bitsPerSample}`,
    );
  }

  const frames = Math.floor(dataSize / (fmt.channels * 4));
  const channels = Array.from({ length: fmt.channels }, () => new Float32Array(frames));
  let cursor = dataOffset;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < fmt.channels; channel += 1) {
      channels[channel][frame] = view.getFloat32(cursor, true);
      cursor += 4;
    }
  }

  return {
    sampleRate: fmt.sampleRate,
    channels: fmt.channels,
    frames,
    data: channels,
  };
}

function resampleToPlanar(decoded, meta) {
  const outputFrames = Math.round((decoded.frames * meta.sample_rate) / decoded.sampleRate);
  const samples = new Float32Array(meta.channels * outputFrames);
  const ratio = decoded.sampleRate / meta.sample_rate;
  for (let t = 0; t < outputFrames; t += 1) {
    const sourcePosition = t * ratio;
    const sourceIndex = Math.floor(sourcePosition);
    const frac = sourcePosition - sourceIndex;
    const nextIndex = Math.min(sourceIndex + 1, decoded.frames - 1);
    for (let channel = 0; channel < meta.channels; channel += 1) {
      const source = decoded.data[Math.min(channel, decoded.channels - 1)];
      const sample = source[sourceIndex] * (1 - frac) + source[nextIndex] * frac;
      samples[channel * outputFrames + t] = sample;
    }
  }
  return samples;
}

function buildSegmentBatch(audio, audioLength, meta) {
  const starts = segmentStarts(audioLength, meta.segment_stride);
  const batch = new Float32Array(starts.length * meta.channels * meta.segment_samples);
  const frameLengths = new Array(starts.length);

  for (let batchIndex = 0; batchIndex < starts.length; batchIndex += 1) {
    const offset = starts[batchIndex];
    const samples = Math.min(audioLength - offset, meta.segment_samples);
    frameLengths[batchIndex] = segmentFrameLength(samples, meta.segment_samples, meta.frame_length);
    for (let channel = 0; channel < meta.channels; channel += 1) {
      const sourceBase = channel * audioLength + offset;
      const targetBase = (batchIndex * meta.channels + channel) * meta.segment_samples;
      for (let t = 0; t < samples; t += 1) {
        batch[targetBase + t] = audio[sourceBase + t];
      }
    }
  }

  return {
    audio: batch,
    audioLength,
    starts,
    frameLengths,
    count: starts.length,
  };
}

function readAscii(view, offset, length) {
  let out = "";
  for (let index = 0; index < length; index += 1) {
    out += String.fromCharCode(view.getUint8(offset + index));
  }
  return out;
}

function findEncodeOutputs(outputs) {
  const tensors = Object.values(outputs);
  const codesTensor = tensors.find((tensor) => tensor.type === "int64");
  const scaleTensor = tensors.find((tensor) => tensor.type === "float32" && tensor.dims.length === 2);
  if (!codesTensor || !scaleTensor) {
    throw new Error(`Unexpected encoder outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return { codesTensor, scaleTensor };
}

function findDecodeOutput(outputs) {
  const tensor = Object.values(outputs).find(
    (candidate) => candidate.type === "float32" && candidate.dims.length === 3,
  );
  if (!tensor) {
    throw new Error(`Unexpected decoder outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return tensor;
}

function toU16Code(raw, index) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0 || value > 65535) {
    throw new Error(`Invalid code at ${index}: ${String(raw)}`);
  }
  return value;
}

function buildRawFrames(codes, scales, segments, meta) {
  const frames = [];
  const valuesPerSegment = meta.num_codebooks * meta.frame_length;
  if (codes.length !== segments.count * valuesPerSegment) {
    throw new Error(
      `Encoder codes length ${codes.length} does not match ${segments.count} * ${valuesPerSegment}`,
    );
  }
  if (scales.length < segments.count) {
    throw new Error(`Encoder scale length ${scales.length} is smaller than segment count ${segments.count}`);
  }

  for (let batchIndex = 0; batchIndex < segments.count; batchIndex += 1) {
    const offset = segments.starts[batchIndex];
    const frameCodes = new Uint16Array(valuesPerSegment);
    const base = batchIndex * valuesPerSegment;
    for (let index = 0; index < valuesPerSegment; index += 1) {
      frameCodes[index] = toU16Code(codes[base + index], base + index);
    }
    frames.push({
      offset,
      samples: Math.min(segments.audioLength - offset, meta.segment_samples),
      frameLength: segments.frameLengths[batchIndex],
      scale: Number(scales[batchIndex] ?? 1),
      codes: frameCodes,
    });
  }
  return frames;
}

function buildDecoderInputs(frames, meta) {
  const valuesPerSegment = meta.num_codebooks * meta.frame_length;
  const codes = new BigInt64Array(frames.length * valuesPerSegment);
  const scales = new Float32Array(frames.length);
  for (let batchIndex = 0; batchIndex < frames.length; batchIndex += 1) {
    const frame = frames[batchIndex];
    if (frame.codes.length !== valuesPerSegment) {
      throw new Error(
        `Decoded raw ECDC frame ${batchIndex} has ${frame.codes.length} codes, expected ${valuesPerSegment}`,
      );
    }
    const base = batchIndex * valuesPerSegment;
    for (let index = 0; index < valuesPerSegment; index += 1) {
      codes[base + index] = BigInt(frame.codes[index]);
    }
    scales[batchIndex] = Number(frame.scale ?? 1);
  }
  return { codes, scales };
}

async function prepareAudioContext() {
  const AudioContextCtor = window.AudioContext ?? window.webkitAudioContext;
  if (!AudioContextCtor) {
    throw new Error("Web Audio is not available in this browser");
  }
  audioContext = audioContext ?? new AudioContextCtor({ sampleRate: 48000 });
  if (audioContext.state !== "running") {
    await audioContext.resume();
  }
  return audioContext;
}

function playPlanarAudio(planar, channels, sampleRate) {
  if (!audioContext) {
    throw new Error("AudioContext was not prepared");
  }
  const frames = planar.length / channels;
  const buffer = audioContext.createBuffer(channels, frames, sampleRate);
  for (let channel = 0; channel < channels; channel += 1) {
    buffer.copyToChannel(planar.subarray(channel * frames, (channel + 1) * frames), channel);
  }
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  source.start();
}

function segmentFrameLength(samples, segmentSamples, frameLength) {
  return Math.ceil((samples * frameLength) / segmentSamples);
}

function segmentStarts(totalSamples, stride) {
  const starts = [];
  for (let offset = 0; offset < totalSamples; offset += Math.max(1, stride)) {
    starts.push(offset);
  }
  return starts;
}

function summarizeFrame(frame) {
  if (!frame) {
    return null;
  }
  return {
    offset: frame.offset,
    samples: frame.samples,
    frameLength: frame.frameLength,
    scale: frame.scale,
  };
}

function summarizeOutputs(outputs) {
  return Object.fromEntries(
    Object.entries(outputs).map(([name, tensor]) => [
      name,
      {
        type: tensor.type,
        dims: tensor.dims,
        length: tensor.data.length,
      },
    ]),
  );
}

function clearMetrics() {
  for (const key of ["model", "input", "codes", "scale", "ecdc", "elapsed"]) {
    els[key].textContent = "-";
  }
  writeLog("");
}

function setStatus(message, kind = "") {
  els.status.textContent = message;
  els.status.className = `status ${kind}`.trim();
}

function writeLog(message) {
  els.log.textContent = message;
}
