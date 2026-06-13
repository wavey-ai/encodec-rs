#!/usr/bin/env node
//
// Westside 1.333s-chunk WASM round-trip test.
//
// Pipeline:
//   1. Read the Westside WAV (target/lori-asha-wasm-native/wav).
//   2. soundkit-style streaming chunker splits it into non-overlapping
//      1.333s PCM chunks (one chunk == the 1333ms bundle's segment_stride,
//      so each chunk is a single, gapless ECDC segment).
//   3. Each PCM chunk is encoded with the WASM encoder into its own .ecdc,
//      saved separately under testdata/out/ecdc.
//   4. Each .ecdc is decoded back to PCM with the WASM decoder (reusing the
//      same header / decode path as scripts/wasm-encode-fixture.mjs) and the
//      per-chunk PCM is written to testdata/out/pcm.
//   5. All per-chunk PCM is concatenated into one contiguous WAV.
//
// No production code is touched: this only uses the existing wasm exports in
// pkg/encodec_rs.js and onnxruntime-web.
//
// Usage:
//   node scripts/westside-chunk-wasm-roundtrip.mjs
//   WESTSIDE_MAX_CHUNKS=2 node scripts/westside-chunk-wasm-roundtrip.mjs   # quick smoke

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
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

// Chatty progress goes to stderr so the final JSON summary on stdout stays clean.
function log(message) {
  process.stderr.write(`${message}\n`);
}

const config = {
  inputWav: path.join(
    repoRoot,
    "target/lori-asha-wasm-native/wav/02 - Lori Asha - Westside.48k-stereo.wav",
  ),
  bundleDir: path.join(repoRoot, "onnx-bundles/encodec_48khz_12kbps_1333ms"),
  outRoot: path.join(repoRoot, "testdata/out"),
  ecdcDir: path.join(repoRoot, "testdata/out/ecdc"),
  pcmDir: path.join(repoRoot, "testdata/out/pcm"),
  contiguousWav: path.join(repoRoot, "testdata/out/westside.contiguous.wav"),
  maxChunks: process.env.WESTSIDE_MAX_CHUNKS
    ? Number(process.env.WESTSIDE_MAX_CHUNKS)
    : Infinity,
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
  log("🎧  Westside 1.333s-chunk WASM round-trip");
  log("──────────────────────────────────────────────────────────");
  log(`📥  input wav : ${path.relative(repoRoot, options.inputWav)}`);
  log(`📦  bundle    : ${path.relative(repoRoot, options.bundleDir)}`);
  log("⚙️   booting onnxruntime-web + encodec_rs wasm…");
  configureOrt();
  initEncodecWasm();

  const bundleJson = readFileSync(path.join(options.bundleDir, "bundle.json"), "utf8");
  const meta = JSON.parse(bundleJson);
  if (!meta.lm_quant_weight_model) {
    throw new Error("bundle has no q8 LM runtime (lm_quant_weight_model)");
  }
  const weights = new Uint8Array(
    readFileSync(path.join(options.bundleDir, meta.lm_quant_weight_model)),
  );
  const lmRuntime = { weights, hash: stableHashHex(weights), bitstreamVersion: 2 };
  log(`🔑  q8 LM weights loaded (hash ${lmRuntime.hash.slice(0, 12)}…)`);

  log("🔍  reading + decoding source wav…");
  const wavBytes = readFileSync(options.inputWav);
  const wav = decodeWav(
    wavBytes.buffer.slice(wavBytes.byteOffset, wavBytes.byteOffset + wavBytes.byteLength),
  );
  if (wav.sampleRate !== meta.sample_rate) {
    throw new Error(`WAV sample rate ${wav.sampleRate} != bundle ${meta.sample_rate}`);
  }
  if (wav.channels !== meta.channels) {
    throw new Error(`WAV channels ${wav.channels} != bundle ${meta.channels}`);
  }
  log(
    `🎼  source: ${wav.channels}ch ${wav.sampleRate}Hz, ${wav.frames.toLocaleString()} frames ` +
      `(${(wav.frames / wav.sampleRate).toFixed(1)}s)`,
  );

  mkdirSync(options.ecdcDir, { recursive: true });
  mkdirSync(options.pcmDir, { recursive: true });

  // "1.333s chunk mode": one chunk spans the bundle's non-overlapping hop, so
  // re-using the standard fixed header yields a single-segment ECDC per chunk.
  const chunkFramesTarget = meta.segment_stride;
  const chunkMs = Number(((chunkFramesTarget / meta.sample_rate) * 1000).toFixed(2));
  const expectedChunks = Math.ceil(wav.frames / chunkFramesTarget);
  log(`✂️   chunk size: ${chunkFramesTarget.toLocaleString()} frames (~${chunkMs}ms) → ${expectedChunks} chunks`);
  log(`📁  ecdc out : ${path.relative(repoRoot, options.ecdcDir)}`);
  log(`📁  pcm  out : ${path.relative(repoRoot, options.pcmDir)}`);
  log("──────────────────────────────────────────────────────────");

  const encodeSession = await createSession(path.join(options.bundleDir, meta.encode_model));
  const decodeSession = await createSession(path.join(options.bundleDir, meta.decode_model));

  const decodedChunks = [];
  const chunkReports = [];
  let totalEcdcBytes = 0;
  let chunkIndex = 0;

  for (const chunk of chunkPcmStreaming(wav, chunkFramesTarget)) {
    if (chunkIndex >= options.maxChunks) break;

    const name = `chunk_${String(chunkIndex).padStart(4, "0")}`;
    const ecdcPath = path.join(options.ecdcDir, `${name}.ecdc`);
    const pcmPath = path.join(options.pcmDir, `${name}.wav`);
    const chunkStarted = Date.now();

    // --- WASM encode: PCM chunk -> standalone .ecdc ---
    const ecdc = await encodeChunk(encodeSession, bundleJson, meta, lmRuntime, chunk);
    writeFileSync(ecdcPath, ecdc);
    totalEcdcBytes += ecdc.byteLength;

    // --- WASM decode: .ecdc (read back from disk) -> PCM ---
    const decoded = await decodeChunk(decodeSession, bundleJson, meta, lmRuntime, ecdcPath);
    if (decoded.frames !== chunk.frames) {
      throw new Error(
        `${name}: decoded ${decoded.frames} frames, expected ${chunk.frames}`,
      );
    }
    writeWav(pcmPath, decoded.planar, meta.channels, meta.sample_rate);
    decodedChunks.push(decoded);

    const tookMs = Date.now() - chunkStarted;
    log(
      `  ✅  ${name}  ${String(chunk.frames).padStart(5)} frames  ` +
        `enc→${String(ecdc.byteLength).padStart(5)}B ${name}.ecdc  ` +
        `dec→${name}.wav  (${tookMs}ms)  [${chunkIndex + 1}/${expectedChunks}]`,
    );

    chunkReports.push({
      index: chunkIndex,
      frames: chunk.frames,
      ms: Number(((chunk.frames / meta.sample_rate) * 1000).toFixed(2)),
      ecdc: path.relative(repoRoot, ecdcPath),
      ecdcBytes: ecdc.byteLength,
      pcm: path.relative(repoRoot, pcmPath),
    });
    chunkIndex += 1;
  }

  if (decodedChunks.length === 0) {
    throw new Error("no chunks were produced");
  }

  // --- Combine per-chunk PCM into one contiguous WAV ---
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
  mkdirSync(path.dirname(options.contiguousWav), { recursive: true });
  writeWav(options.contiguousWav, contiguous, meta.channels, meta.sample_rate);

  log("──────────────────────────────────────────────────────────");
  log(
    `🎉  done in ${((Date.now() - started) / 1000).toFixed(1)}s — ${decodedChunks.length} chunks, ` +
      `${totalEcdcBytes.toLocaleString()} ecdc bytes, ` +
      `${totalFrames.toLocaleString()}/${wav.frames.toLocaleString()} frames reconstructed`,
  );
  log(`🌊  contiguous wav → ${path.relative(repoRoot, options.contiguousWav)}`);
  log("");

  return {
    inputWav: path.relative(repoRoot, options.inputWav),
    bundleDir: path.relative(repoRoot, options.bundleDir),
    runtime: "onnxruntime-web wasm + encodec_rs wasm",
    chunkMode: { ms: chunkMs, frames: chunkFramesTarget },
    sampleRate: meta.sample_rate,
    channels: meta.channels,
    sourceFrames: wav.frames,
    chunks: decodedChunks.length,
    decodedFrames: totalFrames,
    totalEcdcBytes,
    ecdcDir: path.relative(repoRoot, options.ecdcDir),
    pcmDir: path.relative(repoRoot, options.pcmDir),
    contiguousWav: path.relative(repoRoot, options.contiguousWav),
    firstChunk: chunkReports[0],
    lastChunk: chunkReports[chunkReports.length - 1],
  };
}

// soundkit-style streaming chunker: walks the planar PCM and emits fixed-size,
// non-overlapping PCM frames (mirrors soundkit's WavStreamProcessor +
// AudioEncoder frame chunking), with a trailing partial frame at EOF.
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

async function encodeChunk(session, bundleJson, meta, lmRuntime, chunk) {
  // Pad the chunk into a full segment (segment_samples) for the encode model.
  const segment = new Float32Array(meta.channels * meta.segment_samples);
  for (let channel = 0; channel < meta.channels; channel += 1) {
    segment.set(
      chunk.planar.subarray(channel * chunk.frames, (channel + 1) * chunk.frames),
      channel * meta.segment_samples,
    );
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
  const frameLength = segmentFrameLength(
    Math.min(chunk.frames, meta.segment_samples),
    meta.segment_samples,
    meta.frame_length,
  );

  // Re-use the same fixed header as wasm-encode-fixture.mjs, scoped to this chunk.
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
    payload = encoder.finishPadded(meta.frame_length);
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

  // Reuse the production overlap/trim helper; for a single non-overlapping
  // segment it simply trims to the chunk's true sample count.
  const planar = ecdcOverlapAddForMetadata(
    bundleJson,
    JSON.stringify(metadata),
    decodedTensor.data,
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

function segmentFrameLength(samples, segmentSamples, frameLength) {
  return Math.ceil((samples * frameLength) / segmentSamples);
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
      out.writeInt16LE(Math.round(value * 32767), cursor);
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
