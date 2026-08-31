#!/usr/bin/env node
//
// AUDIO-2022-01-19-05-52-58 EnCodec wasm round-trip for two profiles:
//   - encodec_48khz_24kbps_1333ms      (16 codebooks)
//   - encodec_48khz_12kbps_7cb_1333ms  (7 codebooks)
//
// Pipeline (matches westside-chunk-wasm-roundtrip.mjs):
//   1. Read the 48 kHz stereo source WAV.
//   2. Stream it in non-overlapping 1.333s PCM chunks (segment_stride).
//   3. Encode each chunk with the encodec_rs wasm encoder into .ecdc.
//   4. Decode each .ecdc back to PCM with the encodec_rs wasm decoder.
//   5. Concatenate per-chunk PCM into one contiguous roundtrip WAV.
//
// Usage:
//   node scripts/audio-2022-wasm-roundtrip.mjs
//
// Outputs are written under target/audio-2022-wasm-roundtrip and copied to
// ~/Downloads/Lori EP.

import { existsSync, copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as ort from "../browser-smoke/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";
import {
  ecdcMetadata,
  ecdcOverlapAddForMetadata,
  initSync,
  initPanicHook,
  lmEcdcChunk,
  lmEcdcDecodeChunks,
  lmEcdcFixedHeaderForWeights,
  QuantizedLmChunkDecoder,
  QuantizedLmChunkEncoder,
  stableHashHex,
} from "../pkg/encodec_rs.js";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));

function log(message) {
  process.stderr.write(`${message}\n`);
}

const config = {
  inputWav: path.join(
    repoRoot,
    "target/audio-2022-wasm-roundtrip/AUDIO-2022-01-19-05-52-58.48k-stereo.wav",
  ),
  outRoot: path.join(repoRoot, "target/audio-2022-wasm-roundtrip"),
  downloadsDir: "/Users/jamie/Downloads/Lori EP",
  profiles: [
    { bundle: "encodec_48khz_24kbps_1333ms", tag: "24kbps" },
    { bundle: "encodec_48khz_12kbps_7cb_1333ms", tag: "12kbps-7cb" },
  ],
  customDecoderRuntimeSource: path.join(
    repoRoot,
    "browser-runtime/custom-decoder-runtime.js",
  ),
  customDecoderBundleRoot: path.join(
    repoRoot,
    "dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_7cb_1333ms/decoder",
  ),
  maxChunks: process.env.MAX_CHUNKS ? Number(process.env.MAX_CHUNKS) : Infinity,
};

try {
  const summary = await run(config);
  console.log(JSON.stringify(summary, null, 2));
} catch (error) {
  console.error(error?.stack ?? String(error));
  process.exitCode = 1;
}

async function run(options) {
  const started = Date.now();
  log("");
  log("🎧  AUDIO-2022-01-19-05-52-58 wasm round-trip");
  log("──────────────────────────────────────────────────────────");
  log(`📥  input wav : ${path.relative(repoRoot, options.inputWav)}`);
  configureOrt();
  initEncodecWasm();

  const wavBytes = readFileSync(options.inputWav);
  const wav = decodeWav(
    wavBytes.buffer.slice(wavBytes.byteOffset, wavBytes.byteOffset + wavBytes.byteLength),
  );
  log(
    `🎼  source: ${wav.channels}ch ${wav.sampleRate}Hz, ${wav.frames.toLocaleString()} frames ` +
      `(${(wav.frames / wav.sampleRate).toFixed(1)}s)`,
  );

  const customRuntimeMjs = path.join(options.outRoot, "custom-decoder-runtime.mjs");
  if (!existsSync(customRuntimeMjs)) {
    copyFileSync(options.customDecoderRuntimeSource, customRuntimeMjs);
  }

  const summaries = [];
  for (const profile of options.profiles) {
    summaries.push(await runProfile(profile, wav, options, customRuntimeMjs));
  }

  log("──────────────────────────────────────────────────────────");
  log("");
  return { inputWav: path.relative(repoRoot, options.inputWav), profiles: summaries };
}

async function runProfile(profile, wav, options, customRuntimeMjs) {
  const started = Date.now();
  const bundleDir = path.join(repoRoot, "onnx-bundles", profile.bundle);
  const bundleJson = readFileSync(path.join(bundleDir, "bundle.json"), "utf8");
  const meta = JSON.parse(bundleJson);
  if (!meta.lm_quant_weight_model) {
    throw new Error(`bundle ${profile.bundle} has no q8 LM runtime`);
  }
  if (wav.sampleRate !== meta.sample_rate) {
    throw new Error(`WAV sample rate ${wav.sampleRate} != bundle ${meta.sample_rate}`);
  }
  if (wav.channels !== meta.channels) {
    throw new Error(`WAV channels ${wav.channels} != bundle ${meta.channels}`);
  }

  const custom = profile.tag === "12kbps-7cb";
  let customDecoder = null;
  if (custom) {
    const { createCustomDecoder } = await import(pathToFileURL(customRuntimeMjs).href);
    const decoderRoot = options.customDecoderBundleRoot;
    const kernelPath = path.join(
      decoderRoot,
      "encodec-convtranspose-relaxed.mjs",
    );
    const fetchImpl = async (url) => {
      const data = readFileSync(new URL(url).pathname);
      return {
        ok: true,
        status: 200,
        headers: { get: () => "application/octet-stream" },
        json: async () => JSON.parse(data.toString("utf8")),
        arrayBuffer: async () =>
          data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
      };
    };
    customDecoder = await createCustomDecoder({
      assetRoot: pathToFileURL(path.join(decoderRoot, "")),
      kernelModuleUrl: pathToFileURL(kernelPath),
      bundleMetadata: meta,
      fetchImpl,
      versionedAssetUrl: (asset) => {
        const url = new URL(asset);
        return url.protocol === "file:" ? url.pathname : asset;
      },
    });
    log(
      `🗜️   decoder    : custom SIMD WASM (ONNX-free), ${customDecoder.metadata.numCodebooks} codebooks`,
    );
  }

  const weights = new Uint8Array(
    readFileSync(path.join(bundleDir, meta.lm_quant_weight_model)),
  );
  const lmRuntime = { weights, hash: stableHashHex(weights), bitstreamVersion: 2 };
  log(
    `📦  bundle    : ${profile.bundle}   (${meta.num_codebooks} codebooks)  ` +
      `🔑 q8 LM ${lmRuntime.hash.slice(0, 12)}…`,
  );

  const chunkFramesTarget = meta.segment_stride;
  const chunkMs = Number(((chunkFramesTarget / meta.sample_rate) * 1000).toFixed(2));
  const expectedChunks = Math.ceil(wav.frames / chunkFramesTarget);
  log(`✂️   chunk size: ${chunkFramesTarget.toLocaleString()} frames (~${chunkMs}ms) → ${expectedChunks} chunks`);

  const profileOut = path.join(options.outRoot, profile.tag);
  mkdirSync(path.join(profileOut, "ecdc"), { recursive: true });
  mkdirSync(path.join(profileOut, "pcm"), { recursive: true });

  const ecdcName = `AUDIO-2022-01-19-05-52-58.${profile.tag}.1333ms.ecdc`;
  const wavName = `AUDIO-2022-01-19-05-52-58.encodec-rs-wasm.${profile.tag}.1333ms.roundtrip.wav`;
  const ecdcPath = path.join(profileOut, ecdcName);
  const wavPath = path.join(profileOut, wavName);
  const downloadsEcdc = path.join(options.downloadsDir, ecdcName);
  const downloadsWav = path.join(options.downloadsDir, wavName);

  const encodeSession = await createSession(path.join(bundleDir, meta.encode_model));
  const decodeSession = custom
    ? null
    : await createSession(path.join(bundleDir, meta.decode_model));

  const ecdcParts = [];
  const decodedChunks = [];
  const chunkReports = [];
  let totalEcdcBytes = 0;
  let chunkIndex = 0;

  for (const chunk of chunkPcmStreaming(wav, chunkFramesTarget)) {
    if (chunkIndex >= options.maxChunks) break;

    const name = `chunk_${String(chunkIndex).padStart(4, "0")}`;
    const chunkEcdcPath = path.join(profileOut, "ecdc", `${name}.ecdc`);
    const chunkPcmPath = path.join(profileOut, "pcm", `${name}.wav`);
    const chunkStarted = Date.now();

    const ecdc = await encodeChunk(encodeSession, bundleJson, meta, lmRuntime, chunk, wav);
    ecdcParts.push(ecdc);
    writeFileSync(chunkEcdcPath, ecdc);
    totalEcdcBytes += ecdc.byteLength;

    const decoded = custom
      ? await decodeChunkCustom(customDecoder, bundleJson, meta, lmRuntime, chunkEcdcPath)
      : await decodeChunk(decodeSession, bundleJson, meta, lmRuntime, chunkEcdcPath);
    if (decoded.frames !== chunk.frames) {
      throw new Error(`${name}: decoded ${decoded.frames} frames, expected ${chunk.frames}`);
    }
    writeWav(chunkPcmPath, decoded.planar, meta.channels, meta.sample_rate);
    decodedChunks.push(decoded);

    const tookMs = Date.now() - chunkStarted;
    log(
      `  ✅  [${profile.tag}] ${name}  ${String(chunk.frames).padStart(5)} frames  ` +
        `enc→${String(ecdc.byteLength).padStart(5)}B  dec→${name}.wav  (${tookMs}ms)  [${chunkIndex + 1}/${expectedChunks}]`,
    );

    chunkReports.push({
      index: chunkIndex,
      frames: chunk.frames,
      ms: Number(((chunk.frames / meta.sample_rate) * 1000).toFixed(2)),
      ecdc: path.relative(repoRoot, chunkEcdcPath),
      ecdcBytes: ecdc.byteLength,
      pcm: path.relative(repoRoot, chunkPcmPath),
    });
    chunkIndex += 1;
  }

  if (decodedChunks.length === 0) {
    throw new Error("no chunks were produced");
  }

  writeFileSync(ecdcPath, concatUint8Chunks(ecdcParts));

  const totalFrames = decodedChunks.reduce((sum, c) => sum + c.frames, 0);
  const contiguous = new Float32Array(meta.channels * totalFrames);
  let frameCursor = 0;
  for (const c of decodedChunks) {
    for (let channel = 0; channel < meta.channels; channel += 1) {
      contiguous.set(
        c.planar.subarray(channel * c.frames, (channel + 1) * c.frames),
        channel * totalFrames + frameCursor,
      );
    }
    frameCursor += c.frames;
  }
  writeWav(wavPath, contiguous, meta.channels, meta.sample_rate);

  copyFileSync(ecdcPath, downloadsEcdc);
  copyFileSync(wavPath, downloadsWav);

  const tookMs = Date.now() - started;
  log(
    `🎉  [${profile.tag}] done in ${(tookMs / 1000).toFixed(1)}s — ${decodedChunks.length} chunks, ` +
      `${totalEcdcBytes.toLocaleString()} ecdc bytes, ${contiguous.length / meta.channels}/${wav.frames} frames`,
  );
  log(`🌊  outputs → ${path.relative(repoRoot, ecdcPath)} / ${path.relative(repoRoot, wavPath)}`);
  log(`📤  copies  → ${downloadsEcdc} / ${downloadsWav}`);
  log("");

  if (customDecoder) {
    customDecoder.release();
  }

  return {
    bundle: profile.bundle,
    tag: profile.tag,
    numCodebooks: meta.num_codebooks,
    bandwidthKbps: meta.bandwidth_kbps,
    decoderBackend: custom ? "custom SIMD WASM (ONNX-free)" : "onnxruntime-web wasm",
    chunkMs: Number(chunkMs),
    sampleRate: meta.sample_rate,
    channels: meta.channels,
    sourceFrames: wav.frames,
    chunks: decodedChunks.length,
    decodedFrames: totalFrames,
    totalEcdcBytes,
    ecdc: path.relative(repoRoot, ecdcPath),
    wav: path.relative(repoRoot, wavPath),
    downloadsEcdc,
    downloadsWav,
    firstChunk: chunkReports[0],
    lastChunk: chunkReports[chunkReports.length - 1],
  };
}

function* chunkPcmStreaming(wav, chunkFrames) {
  for (let start = 0; start < wav.frames; start += chunkFrames) {
    const frames = Math.min(chunkFrames, wav.frames - start);
    const planar = new Float32Array(wav.channels * frames);
    for (let channel = 0; channel < wav.channels; channel += 1) {
      planar.set(
        wav.audio.subarray(channel * wav.frames + start, channel * wav.frames + start + frames),
        channel * frames,
      );
    }
    yield { start, frames, planar };
  }
}

async function encodeChunk(session, bundleJson, meta, lmRuntime, chunk, wav) {
  const context = fixedContextSamples(meta);
  if (context === null) {
    throw new Error("requires a recognized fixed-context bundle");
  }

  const segment = new Float32Array(meta.channels * meta.segment_samples);
  for (let channel = 0; channel < meta.channels; channel += 1) {
    const sourceBase = channel * wav.frames;
    const targetBase = channel * meta.segment_samples;
    for (let modelIndex = 0; modelIndex < meta.segment_samples; modelIndex += 1) {
      const sourceIndex = chunk.start - context + modelIndex;
      if (sourceIndex >= 0 && sourceIndex < wav.frames) {
        segment[targetBase + modelIndex] = wav.audio[sourceBase + sourceIndex];
      }
    }
  }

  const outputs = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", segment, [
      1,
      meta.channels,
      meta.segment_samples,
    ]),
  });
  const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
  const valuesPerSegment = meta.num_codebooks * meta.frame_length;
  const codes = new Uint16Array(valuesPerSegment);
  for (let i = 0; i < valuesPerSegment; i += 1) {
    codes[i] = toU16Code(codesTensor.data[i], i);
  }
  const scale = Number(scaleTensor.data[0] ?? 1);
  const frameLength = meta.frame_length;

  const header = lmEcdcFixedHeaderForWeights(
    bundleJson,
    chunk.frames,
    lmRuntime.bitstreamVersion,
    lmRuntime.weights,
  );

  const encoder = new QuantizedLmChunkEncoder(bundleJson, lmRuntime.weights, scale);
  let payload;
  try {
    for (let step = 0; step < frameLength; step += 1) {
      const stepCodes = new Uint16Array(meta.num_codebooks);
      for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
        stepCodes[codebook] = codes[codebook * meta.frame_length + step];
      }
      encoder.push(stepCodes);
    }
    payload = encoder.finish();
  } finally {
    encoder.free();
  }

  return concatUint8Chunks([header, lmEcdcChunk(payload)]);
}

async function decodeChunk(session, bundleJson, meta, lmRuntime, ecdcPath) {
  const ecdc = readFileSync(ecdcPath);
  const metadata = ecdcMetadata(ecdc);
  const parsed = lmEcdcDecodeChunks(bundleJson, ecdc);
  if (parsed.chunks.length !== 1) {
    throw new Error(`expected 1 ecdc chunk, got ${parsed.chunks.length}`);
  }
  const chunk = parsed.chunks[0];

  const codes = new Uint16Array(meta.num_codebooks * meta.frame_length);
  const decoder = new QuantizedLmChunkDecoder(
    bundleJson,
    lmRuntime.weights,
    Uint8Array.from(chunk.payload),
  );
  let scale;
  try {
    for (let step = 0; step < chunk.frameLength; step += 1) {
      const symbols = decoder.pull();
      for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
        codes[codebook * meta.frame_length + step] = symbols[codebook];
      }
    }
    scale = decoder.scale();
  } finally {
    decoder.free();
  }

  const decoderCodes = new BigInt64Array(codes.length);
  for (let i = 0; i < codes.length; i += 1) {
    decoderCodes[i] = BigInt(codes[i]);
  }
  const outputs = await session.run({
    [session.inputNames[0]]: new ort.Tensor("int64", decoderCodes, [
      1,
      meta.num_codebooks,
      meta.frame_length,
    ]),
    [session.inputNames[1]]: new ort.Tensor("float32", new Float32Array([scale]), [1, 1]),
  });
  const decodedTensor = findDecodeOutput(outputs);

  const planar = ecdcOverlapAddForMetadata(
    bundleJson,
    JSON.stringify(metadata),
    decodedTensor.data,
  );
  const frames = Math.floor(planar.length / meta.channels);
  return { frames, planar };
}

async function decodeChunkCustom(customDecoder, bundleJson, meta, lmRuntime, ecdcPath) {
  const ecdc = readFileSync(ecdcPath);
  const metadata = ecdcMetadata(ecdc);
  const parsed = lmEcdcDecodeChunks(bundleJson, ecdc);
  if (parsed.chunks.length !== 1) {
    throw new Error(`expected 1 ecdc chunk, got ${parsed.chunks.length}`);
  }
  const chunk = parsed.chunks[0];

  const codes = new Uint16Array(meta.num_codebooks * meta.frame_length);
  const decayed = new QuantizedLmChunkDecoder(
    bundleJson,
    lmRuntime.weights,
    Uint8Array.from(chunk.payload),
  );
  let scale;
  try {
    for (let step = 0; step < chunk.frameLength; step += 1) {
      const symbols = decayed.pull();
      for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
        codes[codebook * meta.frame_length + step] = symbols[codebook];
      }
    }
    scale = decayed.scale();
  } finally {
    decayed.free();
  }

  const decoded = customDecoder.decode([{ codes, scale }]);
  const planar = ecdcOverlapAddForMetadata(
    bundleJson,
    JSON.stringify(metadata),
    decoded.audio,
  );
  const frames = Math.floor(planar.length / meta.channels);
  return { frames, planar };
}

function configureOrt() {
  ort.env.wasm.wasmPaths = pathToFileURL(
    path.join(repoRoot, "browser-smoke/node_modules/onnxruntime-web/dist") + path.sep,
  ).href;
  ort.env.wasm.numThreads = 1;
}

function initEncodecWasm() {
  initSync({ module: readFileSync(path.join(repoRoot, "pkg/encodec_rs_bg.wasm")) });
  initPanicHook();
}

async function createSession(modelPath) {
  return ort.InferenceSession.create(readFileSync(modelPath), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
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
      fmt = { formatTag, subFormatTag, channels, sampleRate, bitsPerSample };
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
  return { sampleRate: fmt.sampleRate, channels: fmt.channels, frames, audio };
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

function toU16Code(raw, index) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0 || value > 65535) {
    throw new Error(`invalid code at ${index}: ${String(raw)}`);
  }
  return value;
}

function readAscii(view, offset, length) {
  let out = "";
  for (let index = 0; index < length; index += 1) {
    out += String.fromCharCode(view.getUint8(offset + index));
  }
  return out;
}

function writeWav(outputPath, planar, channels, sampleRate) {
  const frames = Math.floor(planar.length / channels);
  const bytesPerSample = 2;
  const dataBytes = frames * channels * bytesPerSample;
  const out = Buffer.alloc(44 + dataBytes);
  out.write("RIFF", 0, "ascii");
  out.writeUInt32LE(36 + dataBytes, 4);
  out.write("WAVE", 8, "ascii");
  out.write("fmt ", 12, "ascii");
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(1, 20);
  out.writeUInt16LE(channels, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * channels * bytesPerSample, 28);
  out.writeUInt16LE(channels * bytesPerSample, 32);
  out.writeUInt16LE(16, 34);
  out.write("data", 36, "ascii");
  out.writeUInt32LE(dataBytes, 40);
  let cursor = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const value = Math.max(-1, Math.min(1, planar[channel * frames + frame]));
      out.writeInt16LE(
        value < 0 ? Math.round(value * 32768) : Math.round(value * 32767),
        cursor,
      );
      cursor += bytesPerSample;
    }
  }
  writeFileSync(outputPath, out);
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

function summarizeOutputs(outputs) {
  return Object.fromEntries(
    Object.entries(outputs).map(([name, tensor]) => [
      name,
      { type: tensor.type, dims: tensor.dims, length: tensor.data.length },
    ]),
  );
}