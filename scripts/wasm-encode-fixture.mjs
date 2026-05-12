#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as ort from "../browser-smoke/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs";
import {
  initSync,
  initPanicHook,
  lmEcdcChunk,
  lmEcdcHeader,
  LmChunkEncoder,
  rawEcdcFramePayload,
  rawEcdcHeader,
  rawEcdcMetadata,
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

  const bundleJson = readFileSync(path.join(options.bundleDir, "bundle.json"), "utf8");
  const meta = JSON.parse(bundleJson);
  if (options.coding === "lm" && !meta.lm_model) {
    throw new Error("LM coding requested, but bundle has no lm_model");
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
  const lmSession =
    options.coding === "lm" ? await createSession(path.join(options.bundleDir, meta.lm_model)) : null;
  const sessionMs = performance.now() - sessionsStarted;

  const encodedStarted = performance.now();
  const chunks =
    options.coding === "lm" ? [lmEcdcHeader(bundleJson, wav.frames)] : [rawEcdcHeader(bundleJson, wav.frames)];
  const frames = [];
  let frameOnnxMs = 0;
  let lmOnnxMs = 0;
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

    if (options.coding === "lm") {
      const lmFrame = await encodeLmFrame(lmSession, bundleJson, frame, meta);
      lmOnnxMs += lmFrame.lmOnnxMs;
      arithmeticMs += lmFrame.arithmeticMs;
      chunks.push(lmEcdcChunk(lmFrame.payload));
    } else {
      chunks.push(rawEcdcFramePayload(bundleJson, frame.codes, frame.scale, frame.frameLength));
    }
  }

  const ecdc = concatUint8Chunks(chunks);
  mkdirSync(path.dirname(options.outputEcdc), { recursive: true });
  writeFileSync(options.outputEcdc, ecdc);
  const totalEncodeMs = performance.now() - encodedStarted;
  const metadata = rawEcdcMetadata(ecdc);

  return {
    inputWav: path.relative(repoRoot, options.inputWav),
    outputEcdc: path.relative(repoRoot, options.outputEcdc),
    bundleDir: path.relative(repoRoot, options.bundleDir),
    coding: options.coding,
    runtime: "onnxruntime-web wasm",
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
      arithmeticMs: roundMs(arithmeticMs),
      totalEncodeMs: roundMs(totalEncodeMs),
    },
    firstFrame: summarizeFrame(frames[0]),
    lastFrame: summarizeFrame(frames[frames.length - 1]),
  };
}

function parseArgs(args) {
  const out = {
    inputWav: path.join(repoRoot, "testdata/westside_4s_48khz_stereo.wav"),
    outputEcdc: path.join(repoRoot, "target/wasm-smoke/westside_4s_48khz_stereo.lm.ecdc"),
    bundleDir: path.join(repoRoot, "onnx-bundles/encodec_48khz_12kbps"),
    coding: "lm",
  };
  const positional = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--bundle") {
      out.bundleDir = path.resolve(args[++index]);
    } else if (arg === "--coding") {
      out.coding = args[++index];
    } else if (arg === "--output") {
      out.outputEcdc = path.resolve(args[++index]);
    } else if (arg === "--help" || arg === "-h") {
      printUsageAndExit();
    } else {
      positional.push(arg);
    }
  }
  if (positional[0]) {
    out.inputWav = path.resolve(positional[0]);
  }
  if (positional[1]) {
    out.outputEcdc = path.resolve(positional[1]);
  }
  if (!["lm", "raw"].includes(out.coding)) {
    throw new Error(`--coding must be "lm" or "raw", got ${out.coding}`);
  }
  return out;
}

function printUsageAndExit() {
  console.log(
    [
      "Usage: node scripts/wasm-encode-fixture.mjs [input.wav] [output.ecdc]",
      "",
      "Options:",
      "  --bundle <dir>    ONNX bundle directory",
      "  --coding <lm|raw> ECDC coding mode, defaults to lm",
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

async function encodeLmFrame(lmSession, bundleJson, frame, meta) {
  const encoder = new LmChunkEncoder(bundleJson, frame.scale);
  let states = initialLmStates(meta);
  let offset = 0;
  let inputValues = new BigInt64Array(meta.num_codebooks);
  let lmOnnxMs = 0;
  let arithmeticMs = 0;

  for (let step = 0; step < frame.frameLength; step += 1) {
    const lmStarted = performance.now();
    const lm = await runLmStep(lmSession, meta, inputValues, offset, states);
    lmOnnxMs += performance.now() - lmStarted;
    const stepCodes = frameStepCodes(frame, meta, step);
    const arithmeticStarted = performance.now();
    encoder.push(lm.logits.data, stepCodes);
    arithmeticMs += performance.now() - arithmeticStarted;
    inputValues = lmInputFromCodes(stepCodes);
    states = lm.nextStates;
    offset = lm.nextOffset;
  }

  return {
    payload: encoder.finish(),
    lmOnnxMs,
    arithmeticMs,
  };
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
