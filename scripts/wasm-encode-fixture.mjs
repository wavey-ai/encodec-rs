#!/usr/bin/env node

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

const options = parseArgs(process.argv.slice(2));

try {
  const summary = await run(options);
  console.log(JSON.stringify(summary, null, 2));
} catch (error) {
  console.error(error?.stack ?? String(error));
  process.exitCode = 1;
}

async function run(options) {
  configureOrt();
  initEncodecWasm();

  if (options.command === "decode") {
    return decodeFixture(options);
  }
  return encodeFixture(options);
}

async function encodeFixture(options) {
  const bundleJson = readFileSync(path.join(options.bundleDir, "bundle.json"), "utf8");
  const meta = JSON.parse(bundleJson);
  if (options.coding !== "lm") {
    throw new Error("matrix WASM fixture only supports q8 LM coding");
  }
  if (!meta.lm_quant_weight_model) {
    throw new Error("LM coding requested, but bundle has no q8 LM runtime");
  }

  const wavBytes = readFileSync(options.inputWav);
  const wav = decodeWav(wavBytes.buffer.slice(wavBytes.byteOffset, wavBytes.byteOffset + wavBytes.byteLength));
  if (wav.sampleRate !== meta.sample_rate) {
    throw new Error(
      `fixture sample rate ${wav.sampleRate} does not match bundle sample rate ${meta.sample_rate}`,
    );
  }
  if (wav.channels !== meta.channels) {
    throw new Error(`fixture channel count ${wav.channels} does not match bundle channels ${meta.channels}`);
  }

  const segments = buildSegmentBatch(wav.audio, wav.frames, meta);
  const sessionsStarted = performance.now();
  const encodeSession = await createSession(path.join(options.bundleDir, meta.encode_model));
  const lmRuntime = options.coding === "lm" ? await getLmRuntime(options.bundleDir, meta, options) : null;
  const sessionMs = performance.now() - sessionsStarted;

  const encodedStarted = performance.now();
  const chunks = [lmEcdcFixedHeaderForWeights(bundleJson, wav.frames, lmRuntime.bitstreamVersion, lmRuntime.weights)];
  const frames = [];
  let frameOnnxMs = 0;
  let lmOnnxMs = 0;
  let lmDeterministicMs = 0;
  let arithmeticMs = 0;

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
    frames.push(frame);

    const lmFrame = await encodeLmFrame(lmRuntime, bundleJson, frame, meta);
    lmOnnxMs += lmFrame.lmOnnxMs;
    lmDeterministicMs += lmFrame.lmDeterministicMs;
    arithmeticMs += lmFrame.arithmeticMs;
    chunks.push(lmEcdcChunk(lmFrame.payload));
  }

  const ecdc = concatUint8Chunks(chunks);
  mkdirSync(path.dirname(options.outputEcdc), { recursive: true });
  writeFileSync(options.outputEcdc, ecdc);
  const totalEncodeMs = performance.now() - encodedStarted;
  const metadata = ecdcMetadata(ecdc);

  return {
    inputWav: path.relative(repoRoot, options.inputWav),
    outputEcdc: path.relative(repoRoot, options.outputEcdc),
    bundleDir: path.relative(repoRoot, options.bundleDir),
    coding: options.coding,
    runtime: "onnxruntime-web wasm",
    lmRuntime: summarizeLmRuntime(lmRuntime),
    modelName: meta.model_name,
    bandwidthKbps: meta.bandwidth_kbps,
    audioSamples: wav.frames,
    audioSeconds: Number((wav.frames / meta.sample_rate).toFixed(3)),
    segments: segments.count,
    ecdcBytes: ecdc.byteLength,
    ecdcMetadata: metadata,
    timings: {
      sessionMs: roundMs(sessionMs),
      frameOnnxMs: roundMs(frameOnnxMs),
      lmOnnxMs: roundMs(lmOnnxMs),
      lmDeterministicMs: roundMs(lmDeterministicMs),
      arithmeticMs: roundMs(arithmeticMs),
      totalEncodeMs: roundMs(totalEncodeMs),
    },
    firstFrame: summarizeFrame(frames[0]),
    lastFrame: summarizeFrame(frames[frames.length - 1]),
  };
}

async function decodeFixture(options) {
  const bundleJson = readFileSync(path.join(options.bundleDir, "bundle.json"), "utf8");
  const meta = JSON.parse(bundleJson);
  const ecdc = readFileSync(options.inputEcdc);
  const metadata = ecdcMetadata(ecdc);
  const acv = metadata.acv ?? metadata.bitstream_version ?? 0;
  const audioLength = metadata.al ?? metadata.audio_length;
  if (!Number.isInteger(audioLength) || audioLength < 0) {
    throw new Error(`invalid ECDC audio length: ${String(audioLength)}`);
  }

  const started = performance.now();
  const parseStarted = performance.now();
  let frames;
  let lmSessionMs = 0;
  let lmOnnxMs = 0;
  let lmDeterministicMs = 0;
  let arithmeticMs = 0;
  let lmRuntime = null;
  if (acv === 2) {
    const parsed = lmEcdcDecodeChunks(bundleJson, ecdc);
    const lmSessionStarted = performance.now();
    lmRuntime = await getLmRuntime(options.bundleDir, meta, options, acv);
    assertLmRuntimeMatchesMetadata(metadata, lmRuntime);
    lmSessionMs = performance.now() - lmSessionStarted;
    frames = [];
    for (let index = 0; index < parsed.chunks.length; index += 1) {
      const decoded = await decodeLmFrame(lmRuntime, bundleJson, meta, parsed.chunks[index]);
      frames.push(decoded.frame);
      lmOnnxMs += decoded.lmOnnxMs;
      lmDeterministicMs += decoded.lmDeterministicMs;
      arithmeticMs += decoded.arithmeticMs;
    }
  } else {
    throw new Error(`unsupported ECDC coding: acv=${acv}`);
  }
  const parseMs = performance.now() - parseStarted;

  const decodeSessionStarted = performance.now();
  const decodeSession = await createSession(path.join(options.bundleDir, meta.decode_model));
  const decodeSessionMs = performance.now() - decodeSessionStarted;
  const decodedFrames = await decodeFrameBatch(decodeSession, frames, meta);
  const overlapStarted = performance.now();
  const decodedAudio = ecdcOverlapAddForMetadata(bundleJson, JSON.stringify(metadata), decodedFrames.audio);
  const overlapMs = performance.now() - overlapStarted;
  mkdirSync(path.dirname(options.outputWav), { recursive: true });
  writeWav(options.outputWav, decodedAudio, meta.channels, meta.sample_rate);

  return {
    inputEcdc: path.relative(repoRoot, options.inputEcdc),
    outputWav: path.relative(repoRoot, options.outputWav),
    bundleDir: path.relative(repoRoot, options.bundleDir),
    runtime: "onnxruntime-web wasm",
    lmRuntime: summarizeLmRuntime(lmRuntime),
    ecdcMetadata: metadata,
    parsedFrames: frames.length,
    decodedSamples: audioLength,
    sampleRate: meta.sample_rate,
    channels: meta.channels,
    timings: {
      parseMs: roundMs(parseMs),
      lmSessionMs: roundMs(lmSessionMs),
      lmOnnxMs: roundMs(lmOnnxMs),
      lmDeterministicMs: roundMs(lmDeterministicMs),
      arithmeticMs: roundMs(arithmeticMs),
      decodeSessionMs: roundMs(decodeSessionMs),
      decodeOnnxMs: roundMs(decodedFrames.decodeOnnxMs),
      overlapMs: roundMs(overlapMs),
      totalDecodeMs: roundMs(performance.now() - started),
    },
    decoderBatchSize: decodedFrames.batchSize,
    decodedShape: decodedFrames.shape,
    decoderOutputs: decodedFrames.outputSummary,
  };
}

async function decodeFrameBatch(session, frames, meta, batchSize = 32) {
  const samplesPerDecodedFrame = meta.channels * meta.segment_samples;
  const audio = new Float32Array(frames.length * samplesPerDecodedFrame);
  let decodeOnnxMs = 0;
  let outputSummary = null;

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
    const expected = batch.length * samplesPerDecodedFrame;
    if (decodedTensor.data.length !== expected) {
      throw new Error(`decoder batch ${start}-${end} returned ${decodedTensor.data.length} samples, expected ${expected}`);
    }
    audio.set(decodedTensor.data, start * samplesPerDecodedFrame);
    outputSummary = summarizeOutputs(outputs);
  }

  return {
    audio,
    batchSize,
    decodeOnnxMs,
    outputSummary,
    shape: [frames.length, meta.channels, meta.segment_samples],
  };
}

function parseArgs(args) {
  const command = args[0] === "decode" || args[0] === "encode" ? args.shift() : "encode";
  const out = {
    command,
    inputWav: path.join(repoRoot, "testdata/westside_4s_48khz_stereo.wav"),
    inputEcdc: path.join(repoRoot, "target/wasm-smoke/westside_4s_48khz_stereo.lm.ecdc"),
    outputEcdc: path.join(repoRoot, "target/wasm-smoke/westside_4s_48khz_stereo.lm.ecdc"),
    outputWav: path.join(repoRoot, "target/wasm-smoke/westside_4s_wasm_decoded.wav"),
    bundleDir: path.join(repoRoot, "onnx-bundles/encodec_48khz_12kbps"),
    coding: "lm",
    lmBackend: "q8",
  };
  const positional = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--bundle") {
      out.bundleDir = path.resolve(args[++index]);
    } else if (arg === "--coding") {
      out.coding = args[++index];
    } else if (arg === "--output") {
      const output = path.resolve(args[++index]);
      out.outputEcdc = output;
      out.outputWav = output;
    } else if (arg === "--lm-backend") {
      out.lmBackend = args[++index].toLowerCase();
    } else if (arg === "--help" || arg === "-h") {
      printUsageAndExit();
    } else {
      positional.push(arg);
    }
  }
  if (out.command === "decode") {
    if (positional[0]) {
      out.inputEcdc = path.resolve(positional[0]);
    }
    if (positional[1]) {
      out.outputWav = path.resolve(positional[1]);
    }
  } else if (positional[0]) {
    out.inputWav = path.resolve(positional[0]);
    if (positional[1]) {
      out.outputEcdc = path.resolve(positional[1]);
    }
  }
  if (out.coding !== "lm") {
    throw new Error(`--coding must be "lm", got ${out.coding}`);
  }
  return out;
}

function printUsageAndExit() {
  console.log(
    [
      "Usage:",
      "  node scripts/wasm-encode-fixture.mjs encode [input.wav] [output.ecdc]",
      "  node scripts/wasm-encode-fixture.mjs decode [input.ecdc] [output.wav]",
      "",
      "Options:",
      "  --bundle <dir>    ONNX bundle directory",
      "  --coding <lm>    ECDC coding mode, fixed to q8 LM",
      "  --lm-backend <q8> LM backend for matrix runs, fixed to q8",
      "  --output <path>   Output ECDC path",
    ].join("\n"),
  );
  process.exit(0);
}

function configureOrt() {
  ort.env.wasm.wasmPaths = pathToFileURL(
    path.join(repoRoot, "browser-smoke/node_modules/onnxruntime-web/dist") + path.sep,
  ).href;
  ort.env.wasm.numThreads = 1;
}

function initEncodecWasm() {
  const wasmPath = path.join(repoRoot, "pkg/encodec_rs_bg.wasm");
  initSync({ module: readFileSync(wasmPath) });
  initPanicHook();
}

async function getLmRuntime(bundleDir, meta, options = {}, requiredAcv = null) {
  const requested = (options.lmBackend || "q8").toLowerCase();
  if (requested !== "q8") {
    throw new Error(`matrix WASM fixture only supports --lm-backend q8, got ${requested}`);
  }
  if (requiredAcv != null && requiredAcv !== 2) {
    throw new Error(`matrix WASM fixture only supports q8 acv=2 payloads, got acv=${requiredAcv}`);
  }
  if (!meta.lm_quant_weight_model) {
    throw new Error("q8 LM requested, but bundle has no lm_quant_weight_model");
  }
  const weights = new Uint8Array(readFileSync(path.join(bundleDir, meta.lm_quant_weight_model)));
  return {
    kind: "q8",
    weights,
    hash: stableHashHex(weights),
    bitstreamVersion: 2,
    label: "Rust wasm q8 LM",
  };
}

async function createSession(modelPath) {
  const model = readFileSync(modelPath);
  return ort.InferenceSession.create(model, {
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

  return {
    sampleRate: fmt.sampleRate,
    channels: fmt.channels,
    frames,
    audio,
  };
}

function buildSegmentBatch(audio, audioLength, meta) {
  const starts = segmentStarts(audioLength, meta.segment_stride);
  return {
    audio,
    audioLength,
    starts,
    frameLengths: starts.map((offset) =>
      segmentFrameLength(Math.min(audioLength - offset, meta.segment_samples), meta.segment_samples, meta.frame_length),
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

async function encodeLmFrame(lmRuntime, bundleJson, frame, meta) {
  if (lmRuntime.kind !== "q8") {
    throw new Error(`matrix WASM fixture only supports q8 LM, got ${lmRuntime.kind}`);
  }
  const encoder = new QuantizedLmChunkEncoder(bundleJson, lmRuntime.weights, frame.scale);
  try {
    let lmOnnxMs = 0;
    let lmDeterministicMs = 0;
    let arithmeticMs = 0;

    for (let step = 0; step < frame.frameLength; step += 1) {
      const stepCodes = frameStepCodes(frame, meta, step);
      const lmStarted = performance.now();
      encoder.push(stepCodes);
      lmDeterministicMs += performance.now() - lmStarted;
    }

    return {
      payload: encoder.finishPadded(meta.frame_length),
      lmOnnxMs,
      lmDeterministicMs,
      arithmeticMs,
    };
  } finally {
    encoder.free();
  }
}

async function decodeLmFrame(lmRuntime, bundleJson, meta, chunk) {
  if (lmRuntime.kind !== "q8") {
    throw new Error(`matrix WASM fixture only supports q8 LM, got ${lmRuntime.kind}`);
  }
  const decoder = new QuantizedLmChunkDecoder(bundleJson, lmRuntime.weights, Uint8Array.from(chunk.payload));
  try {
    const codes = new Uint16Array(meta.num_codebooks * meta.frame_length);
    let lmOnnxMs = 0;
    let lmDeterministicMs = 0;
    let arithmeticMs = 0;

    for (let step = 0; step < chunk.frameLength; step += 1) {
      const lmStarted = performance.now();
      const symbols = decoder.pull();
      lmDeterministicMs += performance.now() - lmStarted;
      for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
        codes[codebook * meta.frame_length + step] = symbols[codebook];
      }
    }

    return {
      frame: {
        offset: chunk.offset,
        samples: chunk.samples,
        frameLength: chunk.frameLength,
        scale: decoder.scale(),
        codes,
      },
      lmOnnxMs,
      lmDeterministicMs,
      arithmeticMs,
    };
  } finally {
    decoder.free();
  }
}

async function runLmStep(session, meta, inputValues, offset, states) {
  const feeds = {
    indices: new ort.Tensor("int64", new BigInt64Array(inputValues), [1, meta.num_codebooks, 1]),
    offset: new ort.Tensor("int64", new BigInt64Array([BigInt(offset)]), []),
  };
  for (let index = 0; index < states.length; index += 1) {
    feeds[`state_${index}`] = new ort.Tensor("float32", states[index].data, states[index].dims);
  }
  const outputs = await session.run(feeds);
  return {
    logits: outputs.logits ?? findLmLogitsOutput(outputs, meta),
    nextOffset: Number((outputs.offset_out ?? findLmOffsetOutput(outputs)).data[0]),
    nextStates: states.map((_, index) => {
      const tensor = outputs[`next_state_${index}`];
      if (!tensor) {
        throw new Error(`LM output next_state_${index} was not returned`);
      }
      return { data: tensor.data, dims: tensor.dims };
    }),
  };
}

function initialLmStates(meta) {
  return Array.from({ length: meta.lm_num_layers }, () => ({
    data: new Float32Array(meta.lm_dim),
    dims: [1, 1, meta.lm_dim],
  }));
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

function findLmLogitsOutput(outputs, meta) {
  const tensor = Object.values(outputs).find(
    (candidate) =>
      candidate.type === "float32" &&
      candidate.dims.length === 4 &&
      candidate.dims[1] === meta.lm_cardinality &&
      candidate.dims[2] === meta.num_codebooks,
  );
  if (!tensor) {
    throw new Error(`unexpected LM outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return tensor;
}

function findLmOffsetOutput(outputs) {
  const tensor = Object.values(outputs).find((candidate) => candidate.type === "int64" && candidate.data.length === 1);
  if (!tensor) {
    throw new Error(`unexpected LM outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return tensor;
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
    if (frame.codes.length !== valuesPerSegment) {
      throw new Error(`frame ${batchIndex} has ${frame.codes.length} codes, expected ${valuesPerSegment}`);
    }
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

function lmInputFromCodes(codes) {
  const input = new BigInt64Array(codes.length);
  for (let index = 0; index < codes.length; index += 1) {
    input[index] = BigInt(codes[index] + 1);
  }
  return input;
}

function toU16Code(raw, index) {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0 || value > 65535) {
    throw new Error(`invalid code at ${index}: ${String(raw)}`);
  }
  return value;
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
      const value = Math.max(-0.99, Math.min(0.99, planar[channel * frames + frame]));
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

function assertLmRuntimeMatchesMetadata(metadata, lmRuntime) {
  if (!lmRuntime) {
    throw new Error("LM payload requires a q8 LM runtime");
  }
  const acv = metadata.acv ?? metadata.bitstream_version ?? 0;
  if (acv !== lmRuntime.bitstreamVersion) {
    throw new Error(`payload requires acv=${acv}, but WASM runtime provides acv=${lmRuntime.bitstreamVersion}`);
  }
  const expectedHash = metadata.lmh ?? metadata.lm_hash;
  if (!expectedHash) {
    throw new Error("q8 LM payload is missing required lmh");
  }
  if (expectedHash !== lmRuntime.hash) {
    throw new Error(`payload requires LM hash ${expectedHash}, but WASM runtime provides ${lmRuntime.hash}`);
  }
}

function summarizeLmRuntime(lmRuntime) {
  if (!lmRuntime) {
    return null;
  }
  return {
    kind: lmRuntime.kind,
    label: lmRuntime.label,
    bitstreamVersion: lmRuntime.bitstreamVersion,
    hash: lmRuntime.hash,
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

function roundMs(ms) {
  return Number(ms.toFixed(1));
}
