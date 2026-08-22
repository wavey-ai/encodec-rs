#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { createCustomEncoder } from "./custom-encoder-runtime.mjs";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));

const options = parseArgs(process.argv.slice(2));
const customDecoderOnnxFree = options.command === "decode" && options.customDecoderRoot
  ? JSON.parse(readFileSync(path.join(options.customDecoderRoot, "metadata.json"), "utf8"))
      .onnxFree === true
  : false;
const encodec = await import(
  pathToFileURL(path.join(options.encodecWasmRoot, "encodec-rs/pkg/encodec_rs.js")),
);
const needsOrt = options.command === "decode"
  ? !customDecoderOnnxFree
  : options.customEncoderRoot === null;
const ort = needsOrt
  ? await import(
      pathToFileURL(
        path.join(options.wasmRoot, "onnxruntime-web/ort.wasm.min.mjs"),
      ),
    )
  : null;
const {
  ecdcMetadata,
  ecdcOverlapAddForMetadata,
  lmEcdcChunk,
  lmEcdcDecodeChunks,
  lmEcdcFixedHeaderForWeights,
  QuantizedLmChunkDecoder,
  QuantizedLmChunkEncoder,
  stableHashHex,
  triangleOverlapAddPlanarFrames,
} = encodec;

try {
  const summary = await run(options);
  if (options.report) {
    mkdirSync(path.dirname(options.report), { recursive: true });
    writeFileSync(options.report, `${JSON.stringify(summary, null, 2)}\n`);
  }
  console.log(JSON.stringify(summary, null, 2));
  if (summary.signalIntegrity && !summary.signalIntegrity.pass) {
    process.exitCode = 1;
  }
} catch (error) {
  console.error(error?.stack ?? String(error));
  process.exitCode = 1;
}

async function run(options) {
  if (ort !== null) configureOrt();
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
  const customEncoder = options.customEncoderRoot
    ? await createCustomEncoder(
        options.customEncoderRoot,
        options.customEncoderKernelModule,
      )
    : null;
  if (customEncoder) validateCustomEncoderMetadata(customEncoder.metadata, meta);
  const encodeSession = customEncoder
    ? null
    : await createSession(path.join(options.bundleDir, meta.encode_model));
  const lmRuntime = options.coding === "lm" ? await getLmRuntime(options.bundleDir, meta, options) : null;
  const sessionMs = performance.now() - sessionsStarted;

  const encodedStarted = performance.now();
  const chunks = [lmEcdcFixedHeaderForWeights(bundleJson, wav.frames, lmRuntime.bitstreamVersion, lmRuntime.weights)];
  const frames = [];
  let frameModelMs = 0;
  const customEncoderStagesMs = {
    convolutionalFront: 0,
    recurrent: 0,
    latentProjection: 0,
    residualVectorQuantizer: 0,
  };
  let lmOnnxMs = 0;
  let lmDeterministicMs = 0;
  let arithmeticMs = 0;

  for (let index = 0; index < segments.count; index += 1) {
    const segment = buildSingleSegment(wav.audio, wav.frames, segments, index, meta);
    const frameStarted = performance.now();
    let codes;
    let scales;
    if (customEncoder) {
      const output = customEncoder.encode(segment.audio);
      codes = output.codes;
      scales = output.scale;
      for (const [name, value] of Object.entries(output.stagesMs)) {
        customEncoderStagesMs[name] += value;
      }
    } else {
      const outputs = await encodeSession.run({
        [encodeSession.inputNames[0]]: new ort.Tensor("float32", segment.audio, [
          1,
          meta.channels,
          meta.segment_samples,
        ]),
      });
      const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
      codes = codesTensor.data;
      scales = scaleTensor.data;
    }
    frameModelMs += performance.now() - frameStarted;
    const frame = buildRawFrame(codes, scales, segment, meta, index);
    frames.push(frame);

    const lmFrame = await encodeLmFrame(lmRuntime, bundleJson, frame, meta);
    lmOnnxMs += lmFrame.lmOnnxMs;
    lmDeterministicMs += lmFrame.lmDeterministicMs;
    arithmeticMs += lmFrame.arithmeticMs;
    chunks.push(lmEcdcChunk(lmFrame.payload));
    reportProgress("encode", index + 1, segments.count, encodedStarted, wav, options);
  }

  const ecdc = concatUint8Chunks(chunks);
  mkdirSync(path.dirname(options.outputEcdc), { recursive: true });
  writeFileSync(options.outputEcdc, ecdc);
  const totalEncodeMs = performance.now() - encodedStarted;
  const metadata = ecdcMetadata(ecdc);
  customEncoder?.release();
  await encodeSession?.release?.();

  return {
    inputWav: path.relative(repoRoot, options.inputWav),
    outputEcdc: path.relative(repoRoot, options.outputEcdc),
    bundleDir: path.relative(repoRoot, options.bundleDir),
    coding: options.coding,
    runtime: customEncoder ? "custom SIMD WASM" : "onnxruntime-web wasm",
    encoderBackend: customEncoder ? "custom SIMD WASM" : "ONNX Runtime WASM",
    onnxSessionOptions: customEncoder ? null : summarizeOnnxSessionOptions(options),
    lmRuntime: summarizeLmRuntime(lmRuntime),
    modelName: meta.model_name,
    bandwidthKbps: meta.bandwidth_kbps,
    audioSamples: wav.frames,
    audioSeconds: Number((wav.frames / meta.sample_rate).toFixed(3)),
    sourceSignal: summarizeSignal(wav.audio, wav.channels, wav.frames),
    segments: segments.count,
    ecdcBytes: ecdc.byteLength,
    ecdcMetadata: metadata,
    timings: {
      sessionMs: roundMs(sessionMs),
      frameModelMs: roundMs(frameModelMs),
      frameOnnxMs: customEncoder ? null : roundMs(frameModelMs),
      frameCustomWasmMs: customEncoder ? roundMs(frameModelMs) : null,
      customEncoderStagesMs: customEncoder
        ? Object.fromEntries(
            Object.entries(customEncoderStagesMs).map(([name, value]) => [
              name,
              roundMs(value),
            ]),
          )
        : null,
      lmOnnxMs: roundMs(lmOnnxMs),
      lmDeterministicMs: roundMs(lmDeterministicMs),
      arithmeticMs: roundMs(arithmeticMs),
      totalEncodeMs: roundMs(totalEncodeMs),
      realtimeFactor: roundRatio((wav.frames / meta.sample_rate) / (totalEncodeMs / 1000)),
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
      reportProgress(
        "entropy decode",
        index + 1,
        parsed.chunks.length,
        parseStarted,
        { frames: audioLength, sampleRate: meta.sample_rate },
        options,
      );
    }
  } else {
    throw new Error(`unsupported ECDC coding: acv=${acv}`);
  }
  const parseMs = performance.now() - parseStarted;

  const decodeSessionStarted = performance.now();
  const decoder = options.customDecoderRoot
    ? await createCustomDecoder(options.customDecoderRoot, options.customKernelModule, meta)
    : await createSession(path.join(options.bundleDir, meta.decode_model));
  const decodeSessionMs = performance.now() - decodeSessionStarted;
  let decodedFrames;
  try {
    decodedFrames = options.customDecoderRoot
      ? await decodeFrameCustom(decoder, frames, meta)
      : await decodeFrameBatch(
        decoder,
        frames,
        meta,
        options.decodeBatchSize,
        options.preallocateDecoderOutput,
      );
  } finally {
    decoder.release?.();
  }
  let preRepairWav = null;
  if (options.preRepairWav) {
    const preRepairAudio = cropFixedContextWithoutRepair(
      decodedFrames.audio,
      audioLength,
      meta,
    );
    mkdirSync(path.dirname(options.preRepairWav), { recursive: true });
    writeWav(
      options.preRepairWav,
      preRepairAudio,
      meta.channels,
      meta.sample_rate,
      true,
    );
    preRepairWav = path.relative(repoRoot, options.preRepairWav);
  }
  const overlapStarted = performance.now();
  const decodedAudio = ecdcOverlapAddForMetadata(bundleJson, JSON.stringify(metadata), decodedFrames.audio);
  const overlapMs = performance.now() - overlapStarted;
  const seamListeningPack = options.seamPackRoot
    ? writeSeamListeningPack({
      outputRoot: options.seamPackRoot,
      hardAudio: decodedAudio,
      decodedFrames: decodedFrames.audio,
      frameCount: frames.length,
      audioLength,
      meta,
      inputEcdc: options.inputEcdc,
    })
    : null;
  mkdirSync(path.dirname(options.outputWav), { recursive: true });
  writeWav(
    options.outputWav,
    decodedAudio,
    meta.channels,
    meta.sample_rate,
    options.floatWav,
  );

  const decodedSignal = summarizeSignal(decodedAudio, meta.channels, audioLength);
  let signalIntegrity = null;
  let referencePcmParity = null;
  if (options.referenceWav) {
    const referenceBytes = readFileSync(options.referenceWav);
    const reference = decodeWav(
      referenceBytes.buffer.slice(
        referenceBytes.byteOffset,
        referenceBytes.byteOffset + referenceBytes.byteLength,
      ),
    );
    signalIntegrity = compareSignalIntegrity(
      reference,
      {
        sampleRate: meta.sample_rate,
        channels: meta.channels,
        frames: audioLength,
        audio: decodedAudio,
      },
      options.levelToleranceDb,
    );
    referencePcmParity = comparePcm(reference.audio, decodedAudio);
  }

  const warmTotalDecodeMs = (parseMs - lmSessionMs) + decodedFrames.decodeOnnxMs + overlapMs;

  return {
    inputEcdc: path.relative(repoRoot, options.inputEcdc),
    outputWav: path.relative(repoRoot, options.outputWav),
    preRepairWav,
    seamListeningPack,
    bundleDir: path.relative(repoRoot, options.bundleDir),
    runtime: customDecoderOnnxFree
      ? "custom SIMD WASM"
      : "onnxruntime-web wasm",
    decoderBackend: customDecoderOnnxFree
      ? "custom SIMD WASM (ONNX-free)"
      : options.customDecoderRoot
        ? "hybrid custom ConvTranspose + ONNX-WASM"
        : "ONNX-WASM",
    onnxSessionOptions: customDecoderOnnxFree
      ? null
      : summarizeOnnxSessionOptions(options),
    lmRuntime: summarizeLmRuntime(lmRuntime),
    ecdcMetadata: metadata,
    parsedFrames: frames.length,
    decodedSamples: audioLength,
    sampleRate: meta.sample_rate,
    channels: meta.channels,
    decodedSignal,
    referenceWav: options.referenceWav
      ? path.relative(repoRoot, options.referenceWav)
      : null,
    signalIntegrity,
    referencePcmParity,
    timings: {
      parseMs: roundMs(parseMs),
      lmSessionMs: roundMs(lmSessionMs),
      lmOnnxMs: roundMs(lmOnnxMs),
      lmDeterministicMs: roundMs(lmDeterministicMs),
      arithmeticMs: roundMs(arithmeticMs),
      decodeSessionMs: roundMs(decodeSessionMs),
      decodeOnnxMs: customDecoderOnnxFree
        ? 0
        : roundMs(decodedFrames.decodeOnnxMs),
      decodeModelMs: roundMs(decodedFrames.decodeOnnxMs),
      overlapMs: roundMs(overlapMs),
      warmTotalDecodeMs: roundMs(warmTotalDecodeMs),
      warmRealtimeFactor: roundRatio((audioLength / meta.sample_rate) / (warmTotalDecodeMs / 1000)),
      totalDecodeMs: roundMs(performance.now() - started),
    },
    decoderBatchSize: decodedFrames.batchSize,
    decodedShape: decodedFrames.shape,
    decoderOutputs: decodedFrames.outputSummary,
    customDecoderBreakdown: decodedFrames.customBreakdown ?? null,
  };
}

async function decodeFrameBatch(session, frames, meta, batchSize, preallocateOutput) {
  const samplesPerDecodedFrame = meta.channels * meta.segment_samples;
  const audio = new Float32Array(frames.length * samplesPerDecodedFrame);
  const outputTensors = new Map();
  let decodeOnnxMs = 0;
  let outputSummary = null;

  for (let start = 0; start < frames.length; start += batchSize) {
    const end = Math.min(start + batchSize, frames.length);
    const batch = frames.slice(start, end);
    const decodeStarted = performance.now();
    const decoderInputs = buildDecoderInputs(batch, meta);
    const expected = batch.length * samplesPerDecodedFrame;
    const feeds = {
      [session.inputNames[0]]: new ort.Tensor("int64", decoderInputs.codes, [
        batch.length,
        meta.num_codebooks,
        meta.frame_length,
      ]),
      [session.inputNames[1]]: new ort.Tensor("float32", decoderInputs.scales, [batch.length, 1]),
    };
    let fetches;
    if (preallocateOutput) {
      if (session.outputNames.length !== 1) {
        throw new Error(`decoder output preallocation requires one output, got ${session.outputNames.length}`);
      }
      let output = outputTensors.get(batch.length);
      if (!output) {
        output = new ort.Tensor(
          "float32",
          new Float32Array(expected),
          [batch.length, meta.channels, meta.segment_samples],
        );
        outputTensors.set(batch.length, output);
      }
      fetches = { [session.outputNames[0]]: output };
    }
    const outputs = fetches
      ? await session.run(feeds, fetches)
      : await session.run(feeds);
    decodeOnnxMs += performance.now() - decodeStarted;
    const decodedTensor = findDecodeOutput(outputs);
    if (decodedTensor.data.length !== expected) {
      throw new Error(`decoder batch ${start}-${end} returned ${decodedTensor.data.length} samples, expected ${expected}`);
    }
    audio.set(decodedTensor.data, start * samplesPerDecodedFrame);
    outputSummary = summarizeOutputs(outputs);
    reportProgress("neural decode", end, frames.length, null, null, options);
  }

  return {
    audio,
    batchSize,
    decodeOnnxMs,
    outputSummary,
    shape: [frames.length, meta.channels, meta.segment_samples],
  };
}

async function createCustomDecoder(root, kernelModulePath, bundleMeta) {
  const metadata = JSON.parse(readFileSync(path.join(root, "metadata.json"), "utf8"));
  validateCustomDecoderMetadata(metadata, bundleMeta);
  const sessions = Array.from({ length: metadata.stages }, () => null);
  const firstOnnxStage = metadata.onnxFree
    ? metadata.stages
    : metadata.front
      ? 1
      : 0;
  for (let stage = firstOnnxStage; stage < metadata.stages; stage += 1) {
    sessions[stage] = await createSession(path.join(root, `stage-${stage}.onnx`));
  }

  const createKernelModule = (await import(pathToFileURL(kernelModulePath))).default;
  const kernel = await createKernelModule({
    locateFile: (file) => path.join(path.dirname(kernelModulePath), file),
  });
  const kernelState = initializeCustomKernel(kernel, root, metadata);
  return {
    metadata,
    sessions,
    kernel,
    kernelState,
    release() {
      releaseCustomKernel(kernel, kernelState);
      for (const session of sessions) session?.release?.();
    },
  };
}

async function decodeFrameCustom(decoder, frames, meta) {
  const { metadata, sessions, kernel, kernelState } = decoder;
  if (metadata.onnxFree) {
    return decodeFrameFullyCustom(decoder, frames, meta);
  }
  const samplesPerDecodedFrame = meta.channels * meta.segment_samples;
  const audio = new Float32Array(frames.length * samplesPerDecodedFrame);
  const stageMs = Array.from({ length: metadata.stages }, () => 0);
  const kernelMs = Array.from({ length: metadata.layers.length }, () => 0);
  const transferMs = Array.from({ length: metadata.layers.length }, () => 0);
  let frontMs = 0;
  const decodeStarted = performance.now();

  for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
    const frame = frames[frameIndex];
    const frameScale = new Float32Array([Number(frame.scale ?? 1)]);
    let activation = null;
    let outputs = null;
    let started = performance.now();
    if (kernelState.front) {
      runCustomDecoderFront(kernel, kernelState, frame.codes, metadata.front);
      frontMs += performance.now() - started;
    } else {
      const decoderInputs = buildDecoderInputs([frame], meta);
      const frameFeeds = {
        codes: new ort.Tensor("int64", decoderInputs.codes, [
          1,
          meta.num_codebooks,
          meta.frame_length,
        ]),
        scale: new ort.Tensor("float32", decoderInputs.scales, [1, 1]),
      };
      outputs = await sessions[0].run(frameFeeds);
      stageMs[0] += performance.now() - started;
      activation = outputs[sessions[0].outputNames[0]].data;
    }

    for (let layerIndex = 0; layerIndex < metadata.layers.length; layerIndex += 1) {
      const layer = metadata.layers[layerIndex];
      const layerState = kernelState.layers[layerIndex];

      if (!(kernelState.front && layerIndex === 0)) {
        started = performance.now();
        kernel.HEAPF32.set(
          activation,
          kernelState.input / Float32Array.BYTES_PER_ELEMENT,
        );
        transferMs[layerIndex] += performance.now() - started;
      }

      started = performance.now();
      const ok = kernel._conv_transpose1d_phase_simd_8x8(
        kernelState.input,
        layerState.packed,
        layerState.bias,
        kernelState.output,
        layer.inputTime,
        layer.inputChannels,
        layer.outputChannels,
        layer.stride,
      );
      if (ok !== 1) {
        throw new Error(`custom decoder kernel rejected layer ${layerIndex}`);
      }
      kernelMs[layerIndex] += performance.now() - started;

      started = performance.now();
      const outputLength = layer.outputChannels * layer.rawOutputTime;
      const outputOffset = kernelState.output / Float32Array.BYTES_PER_ELEMENT;
      const customOutput = kernel.HEAPF32.slice(
        outputOffset,
        outputOffset + outputLength,
      );
      const activationTensor = new ort.Tensor("float32", customOutput, [
        1,
        layer.outputChannels,
        layer.rawOutputTime,
      ]);
      transferMs[layerIndex] += performance.now() - started;

      const stage = sessions[layerIndex + 1];
      const feeds = {};
      for (const inputName of stage.inputNames) {
        feeds[inputName] = inputName === "scale"
          ? new ort.Tensor("float32", frameScale, [1, 1])
          : activationTensor;
      }
      started = performance.now();
      outputs = await stage.run(feeds);
      stageMs[layerIndex + 1] += performance.now() - started;
      activation = outputs[stage.outputNames[0]].data;
    }

    if (activation.length !== samplesPerDecodedFrame) {
      throw new Error(
        `custom decoder frame ${frameIndex} returned ${activation.length} samples; expected ${samplesPerDecodedFrame}`,
      );
    }
    audio.set(activation, frameIndex * samplesPerDecodedFrame);
    reportProgress("custom neural decode", frameIndex + 1, frames.length, decodeStarted, null, options);
  }

  const decodeOnnxMs = performance.now() - decodeStarted;
  return {
    audio,
    batchSize: 1,
    decodeOnnxMs,
    outputSummary: {
      stages: sessions.length,
      customConvTransposeLayers: metadata.layers.length,
      output: [frames.length, meta.channels, meta.segment_samples],
    },
    customBreakdown: {
      stageMs: stageMs.map(roundMs),
      frontMs: roundMs(frontMs),
      kernelMs: kernelMs.map(roundMs),
      transferMs: transferMs.map(roundMs),
      stageTotalMs: roundMs(stageMs.reduce((sum, value) => sum + value, 0)),
      kernelTotalMs: roundMs(kernelMs.reduce((sum, value) => sum + value, 0)),
      transferTotalMs: roundMs(transferMs.reduce((sum, value) => sum + value, 0)),
    },
    shape: [frames.length, meta.channels, meta.segment_samples],
  };
}

async function decodeFrameFullyCustom(decoder, frames, meta) {
  const { metadata, kernel, kernelState } = decoder;
  const samplesPerDecodedFrame = meta.channels * meta.segment_samples;
  const audio = new Float32Array(frames.length * samplesPerDecodedFrame);
  const breakdown = {
    front: 0,
    frontParts: {
      rvq: 0,
      convolution: 0,
      lstm0: 0,
      lstm1: 0,
      assembly: 0,
    },
    convTranspose: 0,
    residual: 0,
    final: 0,
    copy: 0,
  };
  const decodeStarted = performance.now();

  for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
    const frame = frames[frameIndex];
    let started = performance.now();
    let current = runCustomDecoderFront(
      kernel,
      kernelState,
      frame.codes,
      metadata.front,
      false,
      breakdown.frontParts,
    );
    breakdown.front += performance.now() - started;

    for (let layerIndex = 0; layerIndex < metadata.layers.length; layerIndex += 1) {
      const layer = metadata.layers[layerIndex];
      const layerState = kernelState.layers[layerIndex];
      const raw = current === kernelState.output
        ? kernelState.input
        : kernelState.output;
      started = performance.now();
      if (
        kernel._conv_transpose1d_phase_simd_8x8_nhwc(
          current,
          layerState.packed,
          layerState.bias,
          raw,
          layer.inputTime,
          layer.inputChannels,
          layer.outputChannels,
          layer.stride,
        ) !== 1 ||
        kernel._group_norm_nhwc_in_place(
          raw,
          layerState.normScale,
          layerState.normBias,
          layer.rawOutputTime,
          layer.outputChannels,
        ) !== 1
      ) {
        throw new Error(`custom decoder ConvTranspose ${layerIndex} failed`);
      }
      const cropped = raw === kernelState.output
        ? kernelState.input
        : kernelState.output;
      if (
        kernel._crop_nhwc(
          raw,
          cropped,
          layer.rawOutputTime,
          layer.outputChannels,
          layer.cropLeft,
          layer.cropRight,
        ) !== 1
      ) {
        throw new Error(`custom decoder crop ${layerIndex} failed`);
      }
      breakdown.convTranspose += performance.now() - started;

      started = performance.now();
      const [shortcutIndex, reduceIndex, expandIndex] =
        kernelState.post.blocks[layerIndex];
      const shortcutLayer = kernelState.post.convLayers[shortcutIndex];
      const reduceLayer = kernelState.post.convLayers[reduceIndex];
      const expandLayer = kernelState.post.convLayers[expandIndex];
      runCustomDecoderPostConv(
        kernel,
        kernelState.post,
        cropped,
        shortcutLayer,
        kernelState.post.pointers.shortcut,
        false,
      );
      runCustomDecoderPostConv(
        kernel,
        kernelState.post,
        cropped,
        reduceLayer,
        kernelState.post.pointers.reduced,
        true,
      );
      runCustomDecoderPostConv(
        kernel,
        kernelState.post,
        kernelState.post.pointers.reduced,
        expandLayer,
        raw,
        true,
      );
      const activationLength =
        layer.croppedOutputTime * layer.outputChannels;
      if (
        kernel._add_elu_nhwc_in_place(
          raw,
          kernelState.post.pointers.shortcut,
          activationLength,
        ) !== 1
      ) {
        throw new Error(`custom decoder residual block ${layerIndex} failed`);
      }
      current = raw;
      breakdown.residual += performance.now() - started;
    }

    started = performance.now();
    const final = kernelState.post.final;
    if (
      kernel._reflect_pad_nhwc(
        current,
        kernelState.post.pointers.padded,
        final.inputTime,
        final.inputChannels,
        final.paddingLeft,
        final.paddingRight,
      ) !== 1 ||
      kernel._conv1d_nhwc_simd_8x8(
        kernelState.post.pointers.padded,
        final.pointers.packed,
        final.pointers.bias,
        kernelState.post.pointers.shortcut,
        final.paddedInputTime,
        final.inputChannels,
        final.kernelOutputChannels,
        final.kernel,
        final.stride,
      ) !== 1 ||
      kernel._compact_nhwc_channels(
        kernelState.post.pointers.shortcut,
        kernelState.post.pointers.reduced,
        final.outputTime,
        final.kernelOutputChannels,
        final.outputChannels,
      ) !== 1 ||
      kernel._group_norm_nhwc_in_place(
        kernelState.post.pointers.reduced,
        final.pointers.normScale,
        final.pointers.normBias,
        final.outputTime,
        final.outputChannels,
      ) !== 1 ||
      kernel._scale_nhwc_to_nct(
        kernelState.post.pointers.reduced,
        kernelState.post.pointers.planarOutput,
        Number(frame.scale ?? 1),
        final.outputTime,
        final.outputChannels,
      ) !== 1
    ) {
      throw new Error("custom decoder final projection failed");
    }
    breakdown.final += performance.now() - started;

    started = performance.now();
    const outputOffset =
      kernelState.post.pointers.planarOutput / Float32Array.BYTES_PER_ELEMENT;
    audio.set(
      kernel.HEAPF32.subarray(
        outputOffset,
        outputOffset + samplesPerDecodedFrame,
      ),
      frameIndex * samplesPerDecodedFrame,
    );
    breakdown.copy += performance.now() - started;
    reportProgress(
      "custom neural decode",
      frameIndex + 1,
      frames.length,
      decodeStarted,
      null,
      options,
    );
  }

  const decodeModelMs = performance.now() - decodeStarted;
  return {
    audio,
    batchSize: 1,
    decodeOnnxMs: decodeModelMs,
    outputSummary: {
      onnxFree: true,
      normalization: "ONNX-order two-pass",
      customConvTransposeLayers: metadata.layers.length,
      customConvLayers: metadata.post.convLayers.length + 2,
      output: [frames.length, meta.channels, meta.segment_samples],
    },
    customBreakdown: {
      onnxFree: true,
      frontMs: roundMs(breakdown.front),
      frontPartsMs: Object.fromEntries(
        Object.entries(breakdown.frontParts).map(([name, value]) => [
          name,
          roundMs(value),
        ]),
      ),
      convTransposeMs: roundMs(breakdown.convTranspose),
      residualMs: roundMs(breakdown.residual),
      finalMs: roundMs(breakdown.final),
      copyMs: roundMs(breakdown.copy),
    },
    shape: [frames.length, meta.channels, meta.segment_samples],
  };
}

function runCustomDecoderPostConv(
  module,
  post,
  input,
  layer,
  output,
  applyElu,
) {
  let convolutionInput = input;
  if (layer.paddingLeft !== 0 || layer.paddingRight !== 0) {
    const pad = applyElu
      ? module._reflect_pad_elu_nhwc
      : module._reflect_pad_nhwc;
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
      throw new Error(`custom decoder post padding ${layer.layer} failed`);
    }
    convolutionInput = post.pointers.padded;
  } else if (applyElu) {
    if (
      module._elu_nhwc_in_place(
        input,
        layer.inputTime * layer.inputChannels,
      ) !== 1
    ) {
      throw new Error(`custom decoder post ELU ${layer.layer} failed`);
    }
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
    throw new Error(`custom decoder post Conv ${layer.layer} failed`);
  }
}

function validateCustomDecoderMetadata(metadata, bundleMeta) {
  const checks = [
    ["frame length", metadata.frameLength, bundleMeta.frame_length],
    ["codebook count", metadata.numCodebooks, bundleMeta.num_codebooks],
    ["channel count", metadata.channels, bundleMeta.channels],
    ["segment samples", metadata.segmentSamples, bundleMeta.segment_samples],
  ];
  for (const [label, actual, expected] of checks) {
    if (actual !== expected) {
      throw new Error(`custom decoder ${label} mismatch: ${actual} != ${expected}`);
    }
  }
  if (!Array.isArray(metadata.layers) || metadata.layers.length + 1 !== metadata.stages) {
    throw new Error("custom decoder metadata has inconsistent stages and layers");
  }
}

function validateCustomEncoderMetadata(metadata, bundleMeta) {
  const comparisons = [
    ["sample rate", metadata.sampleRate, bundleMeta.sample_rate],
    ["channels", metadata.channels, bundleMeta.channels],
    ["segment samples", metadata.segmentSamples, bundleMeta.segment_samples],
    ["segment stride", metadata.segmentStride, bundleMeta.segment_stride],
    ["frame length", metadata.frameLength, bundleMeta.frame_length],
    ["codebooks", metadata.numCodebooks, bundleMeta.num_codebooks],
  ];
  for (const [label, actual, expected] of comparisons) {
    if (actual !== expected) {
      throw new Error(
        `custom encoder ${label} mismatch: ${actual} != ${expected}`,
      );
    }
  }
}

function initializeCustomKernel(module, root, metadata) {
  const layers = metadata.layers;
  const allocate = (length) => {
    const pointer = module._malloc(length * Float32Array.BYTES_PER_ELEMENT);
    if (pointer === 0) {
      throw new Error(`failed to allocate ${length} custom-kernel float32 values`);
    }
    return pointer;
  };
  const maxInput = Math.max(
    ...layers.map((layer) => layer.inputChannels * layer.inputTime),
  );
  const maxOutput = Math.max(
    ...layers.map((layer) => layer.outputChannels * layer.rawOutputTime),
  );
  const activationBufferLength = Math.max(maxInput, maxOutput);
  const state = {
    input: allocate(activationBufferLength),
    output: allocate(activationBufferLength),
    layers: [],
  };

  for (const layer of layers) {
    const weights = readFloat32(path.join(root, `layer-${layer.layer}-weight.f32le`));
    const bias = readFloat32(path.join(root, `layer-${layer.layer}-bias.f32le`));
    const rawWeights = allocate(weights.length);
    const packed = allocate(weights.length);
    const biasPointer = allocate(bias.length);
    module.HEAPF32.set(weights, rawWeights / Float32Array.BYTES_PER_ELEMENT);
    module.HEAPF32.set(bias, biasPointer / Float32Array.BYTES_PER_ELEMENT);
    const ok = module._pack_conv_transpose1d_weights(
      rawWeights,
      packed,
      layer.inputChannels,
      layer.outputChannels,
      layer.stride,
    );
    module._free(rawWeights);
    if (ok !== 1) {
      throw new Error(`failed to pack custom decoder layer ${layer.layer}`);
    }
    const layerState = { packed, bias: biasPointer };
    if (metadata.post) {
      const normScale = readFloat32(
        path.join(root, `layer-${layer.layer}-norm-scale.f32le`),
      );
      const normBias = readFloat32(
        path.join(root, `layer-${layer.layer}-norm-bias.f32le`),
      );
      layerState.normScale = allocate(normScale.length);
      layerState.normBias = allocate(normBias.length);
      module.HEAPF32.set(normScale, layerState.normScale / 4);
      module.HEAPF32.set(normBias, layerState.normBias / 4);
    }
    state.layers.push(layerState);
  }
  state.front = metadata.front
    ? initializeCustomDecoderFront(module, root, metadata.front, allocate)
    : null;
  state.post = metadata.post
    ? initializeCustomDecoderPost(module, root, metadata, allocate)
    : null;
  return state;
}

function initializeCustomDecoderFront(module, root, metadata, allocate) {
  const frameLength = metadata.conv.inputTime;
  const hiddenSize = metadata.conv.outputChannels;
  const activationLength = frameLength * hiddenSize;
  const embeddings = readFloat32(path.join(root, "front-rvq-embeddings.f32le"));
  const pointers = {
    codes: allocate(metadata.rvq.codebooks * frameLength),
    embeddings: allocate(embeddings.length),
    padded: allocate(
      metadata.conv.paddedInputTime * metadata.conv.inputChannels,
    ),
    activation0: allocate(activationLength),
    activation1: allocate(activationLength),
    activation2: allocate(activationLength),
    convPacked: allocate(
      metadata.conv.outputChannels *
        metadata.conv.inputChannels *
        metadata.conv.kernel,
    ),
    convBias: allocate(metadata.conv.outputChannels),
    convNormScale: allocate(metadata.conv.outputChannels),
    convNormBias: allocate(metadata.conv.outputChannels),
    lstmHidden: allocate(hiddenSize),
    lstmCell: allocate(hiddenSize),
    lstmInputProjection: allocate(frameLength * 4 * hiddenSize),
  };
  module.HEAPF32.set(embeddings, pointers.embeddings / 4);

  const convWeights = readFloat32(path.join(root, "front-conv-weight.f32le"));
  const unpackedConv = allocate(convWeights.length);
  module.HEAPF32.set(convWeights, unpackedConv / 4);
  module.HEAPF32.set(
    readFloat32(path.join(root, "front-conv-bias.f32le")),
    pointers.convBias / 4,
  );
  module.HEAPF32.set(
    readFloat32(path.join(root, "front-conv-norm-scale.f32le")),
    pointers.convNormScale / 4,
  );
  module.HEAPF32.set(
    readFloat32(path.join(root, "front-conv-norm-bias.f32le")),
    pointers.convNormBias / 4,
  );
  const convPacked = module._pack_conv1d_nhwc_weights_8(
    unpackedConv,
    pointers.convPacked,
    metadata.conv.inputChannels,
    metadata.conv.outputChannels,
    metadata.conv.kernel,
  );
  module._free(unpackedConv);
  if (convPacked !== 1) {
    throw new Error("failed to pack custom decoder-front convolution");
  }

  const lstmLayers = metadata.lstmLayers.map((layer) => {
    const inputWeights = readFloat32(
      path.join(root, `front-lstm-${layer.layer}-input-weight.f32le`),
    );
    const recurrentWeights = readFloat32(
      path.join(root, `front-lstm-${layer.layer}-recurrent-weight.f32le`),
    );
    const bias = readFloat32(
      path.join(root, `front-lstm-${layer.layer}-bias.f32le`),
    );
    const unpackedInput = allocate(inputWeights.length);
    const unpackedRecurrent = allocate(recurrentWeights.length);
    const layerPointers = {
      packedInput: allocate(inputWeights.length),
      packedRecurrent: allocate(recurrentWeights.length),
      bias: allocate(bias.length),
    };
    module.HEAPF32.set(inputWeights, unpackedInput / 4);
    module.HEAPF32.set(recurrentWeights, unpackedRecurrent / 4);
    module.HEAPF32.set(bias, layerPointers.bias / 4);
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
    module._free(unpackedInput);
    module._free(unpackedRecurrent);
    if (inputPacked !== 1 || recurrentPacked !== 1) {
      throw new Error(`failed to pack custom decoder-front LSTM ${layer.layer}`);
    }
    return { ...layer, pointers: layerPointers };
  });

  return { pointers, lstmLayers };
}

function initializeCustomDecoderPost(module, root, metadata, allocate) {
  const convLayers = metadata.post.convLayers.map((layer) => {
    const weights = readFloat32(
      path.join(root, `post-conv-${layer.layer}-weight.f32le`),
    );
    const bias = readFloat32(
      path.join(root, `post-conv-${layer.layer}-bias.f32le`),
    );
    const normScale = readFloat32(
      path.join(root, `post-conv-${layer.layer}-norm-scale.f32le`),
    );
    const normBias = readFloat32(
      path.join(root, `post-conv-${layer.layer}-norm-bias.f32le`),
    );
    const unpacked = allocate(weights.length);
    const pointers = {
      packed: allocate(weights.length),
      bias: allocate(bias.length),
      normScale: allocate(normScale.length),
      normBias: allocate(normBias.length),
    };
    module.HEAPF32.set(weights, unpacked / 4);
    module.HEAPF32.set(bias, pointers.bias / 4);
    module.HEAPF32.set(normScale, pointers.normScale / 4);
    module.HEAPF32.set(normBias, pointers.normBias / 4);
    const packed = module._pack_conv1d_nhwc_weights_8(
      unpacked,
      pointers.packed,
      layer.inputChannels,
      layer.outputChannels,
      layer.kernel,
    );
    module._free(unpacked);
    if (packed !== 1) {
      throw new Error(`failed to pack custom decoder post Conv ${layer.layer}`);
    }
    return { ...layer, pointers };
  });

  const final = metadata.post.finalConv;
  const finalWeights = readFloat32(path.join(root, "final-conv-weight.f32le"));
  const finalBias = readFloat32(path.join(root, "final-conv-bias.f32le"));
  const finalNormScale = readFloat32(
    path.join(root, "final-conv-norm-scale.f32le"),
  );
  const finalNormBias = readFloat32(
    path.join(root, "final-conv-norm-bias.f32le"),
  );
  const paddedFinalWeights = new Float32Array(
    final.kernelOutputChannels * final.inputChannels * final.kernel,
  );
  paddedFinalWeights.set(finalWeights);
  const paddedFinalBias = new Float32Array(final.kernelOutputChannels);
  paddedFinalBias.set(finalBias);
  const unpackedFinal = allocate(paddedFinalWeights.length);
  const finalPointers = {
    packed: allocate(paddedFinalWeights.length),
    bias: allocate(paddedFinalBias.length),
    normScale: allocate(finalNormScale.length),
    normBias: allocate(finalNormBias.length),
  };
  module.HEAPF32.set(paddedFinalWeights, unpackedFinal / 4);
  module.HEAPF32.set(paddedFinalBias, finalPointers.bias / 4);
  module.HEAPF32.set(finalNormScale, finalPointers.normScale / 4);
  module.HEAPF32.set(finalNormBias, finalPointers.normBias / 4);
  const finalPacked = module._pack_conv1d_nhwc_weights_8(
    unpackedFinal,
    finalPointers.packed,
    final.inputChannels,
    final.kernelOutputChannels,
    final.kernel,
  );
  module._free(unpackedFinal);
  if (finalPacked !== 1) {
    throw new Error("failed to pack final custom decoder Conv");
  }

  const maxActivationLength = Math.max(
    ...metadata.layers.map(
      (layer) => layer.croppedOutputTime * layer.outputChannels,
    ),
  );
  const maxReducedLength = Math.max(
    ...convLayers.map((layer) => layer.outputTime * layer.outputChannels),
    final.outputTime * final.outputChannels,
  );
  const maxPaddedLength = Math.max(
    ...convLayers.map(
      (layer) => layer.paddedInputTime * layer.inputChannels,
    ),
    final.paddedInputTime * final.inputChannels,
  );
  const pointers = {
    shortcut: allocate(maxActivationLength),
    reduced: allocate(maxReducedLength),
    padded: allocate(maxPaddedLength),
    planarOutput: allocate(final.outputTime * final.outputChannels),
  };
  return {
    convLayers,
    blocks: metadata.post.blocks,
    final: { ...final, pointers: finalPointers },
    pointers,
  };
}

function runCustomDecoderFront(
  module,
  state,
  codes,
  metadata,
  transposeOutput = true,
  timings = null,
) {
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
    throw new Error("custom decoder-front codebook decode failed");
  }
  if (timings) timings.rvq += performance.now() - started;
  started = performance.now();
  if (
    module._reflect_pad_nhwc(
      pointers.activation0,
      pointers.padded,
      frameLength,
      metadata.conv.inputChannels,
      metadata.conv.paddingLeft,
      metadata.conv.paddingRight,
    ) !== 1
  ) {
    throw new Error("custom decoder-front padding failed");
  }
  if (
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
    throw new Error("custom decoder-front convolution failed");
  }
  if (timings) timings.convolution += performance.now() - started;

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
      throw new Error(`custom decoder-front LSTM ${layer.layer} failed`);
    }
  };
  started = performance.now();
  runLstm(pointers.activation1, front.lstmLayers[0], pointers.activation0);
  if (timings) timings.lstm0 += performance.now() - started;
  started = performance.now();
  runLstm(pointers.activation0, front.lstmLayers[1], pointers.activation2);
  if (timings) timings.lstm1 += performance.now() - started;
  started = performance.now();
  if (
    module._add_elu_nhwc_in_place(
      pointers.activation2,
      pointers.activation1,
      frameLength * hiddenSize,
    ) !== 1
  ) {
    throw new Error("custom decoder-front output assembly failed");
  }
  if (transposeOutput) {
    if (
      module._nhwc_to_nct(
        pointers.activation2,
        state.input,
        frameLength,
        hiddenSize,
      ) !== 1
    ) {
      throw new Error("custom decoder-front layout conversion failed");
    }
    return state.input;
  }
  if (timings) timings.assembly += performance.now() - started;
  return pointers.activation2;
}

function releaseCustomKernel(module, state) {
  module._free(state.input);
  module._free(state.output);
  for (const layer of state.layers) {
    for (const pointer of Object.values(layer)) module._free(pointer);
  }
  if (state.front) {
    for (const pointer of Object.values(state.front.pointers)) {
      module._free(pointer);
    }
    for (const layer of state.front.lstmLayers) {
      for (const pointer of Object.values(layer.pointers)) {
        module._free(pointer);
      }
    }
  }
  if (state.post) {
    for (const pointer of Object.values(state.post.pointers)) {
      module._free(pointer);
    }
    for (const layer of state.post.convLayers) {
      for (const pointer of Object.values(layer.pointers)) {
        module._free(pointer);
      }
    }
    for (const pointer of Object.values(state.post.final.pointers)) {
      module._free(pointer);
    }
  }
}

function readFloat32(file) {
  const bytes = readFileSync(file);
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(copy);
}

function parseArgs(args) {
  const command = args[0] === "decode" || args[0] === "encode" ? args.shift() : "encode";
  const out = {
    command,
    inputWav: path.join(repoRoot, "testdata/westside_4s_48khz_stereo.wav"),
    inputEcdc: path.join(repoRoot, "target/wasm-smoke/westside_4s_48khz_stereo.lm.ecdc"),
    outputEcdc: path.join(repoRoot, "target/wasm-smoke/westside_4s_48khz_stereo.lm.ecdc"),
    outputWav: path.join(repoRoot, "target/wasm-smoke/westside_4s_wasm_decoded.wav"),
    wasmRoot: path.resolve(repoRoot, "../vin.yl.vendor/wasm"),
    encodecWasmRoot: null,
    bundleDir: null,
    customEncoderRoot: null,
    customEncoderKernelModule: null,
    customDecoderRoot: null,
    customKernelModule: null,
    coding: "lm",
    lmBackend: "q8",
    threads: 1,
    graphOptimizationLevel: "all",
    enableCpuMemArena: false,
    enableMemPattern: false,
    preallocateDecoderOutput: false,
    decodeBatchSize: 1,
    progressEvery: 10,
    floatWav: false,
    report: null,
    preRepairWav: null,
    seamPackRoot: null,
    referenceWav: null,
    levelToleranceDb: 1,
  };
  const positional = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--bundle") {
      out.bundleDir = path.resolve(args[++index]);
    } else if (arg === "--wasm-root") {
      out.wasmRoot = path.resolve(args[++index]);
    } else if (arg === "--encodec-wasm-root") {
      out.encodecWasmRoot = path.resolve(args[++index]);
    } else if (arg === "--custom-encoder-root") {
      out.customEncoderRoot = path.resolve(args[++index]);
    } else if (arg === "--custom-encoder-kernel-module") {
      out.customEncoderKernelModule = path.resolve(args[++index]);
    } else if (arg === "--custom-decoder-root") {
      out.customDecoderRoot = path.resolve(args[++index]);
    } else if (arg === "--custom-kernel-module") {
      out.customKernelModule = path.resolve(args[++index]);
    } else if (arg === "--coding") {
      out.coding = args[++index];
    } else if (arg === "--output") {
      const output = path.resolve(args[++index]);
      out.outputEcdc = output;
      out.outputWav = output;
    } else if (arg === "--lm-backend") {
      out.lmBackend = args[++index].toLowerCase();
    } else if (arg === "--threads") {
      out.threads = Number(args[++index]);
    } else if (arg === "--graph-optimization-level") {
      out.graphOptimizationLevel = args[++index];
    } else if (arg === "--enable-cpu-mem-arena") {
      out.enableCpuMemArena = true;
    } else if (arg === "--enable-mem-pattern") {
      out.enableMemPattern = true;
    } else if (arg === "--preallocate-decoder-output") {
      out.preallocateDecoderOutput = true;
    } else if (arg === "--decode-batch-size") {
      out.decodeBatchSize = Number(args[++index]);
    } else if (arg === "--progress-every") {
      out.progressEvery = Number(args[++index]);
    } else if (arg === "--float-wav") {
      out.floatWav = true;
    } else if (arg === "--report") {
      out.report = path.resolve(args[++index]);
    } else if (arg === "--pre-repair-wav") {
      out.preRepairWav = path.resolve(args[++index]);
    } else if (arg === "--seam-pack-root") {
      out.seamPackRoot = path.resolve(args[++index]);
    } else if (arg === "--reference-wav") {
      out.referenceWav = path.resolve(args[++index]);
    } else if (arg === "--level-tolerance-db") {
      out.levelToleranceDb = Number(args[++index]);
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
  if (!out.bundleDir) {
    out.bundleDir = path.join(
      out.wasmRoot,
      "encodec-rs/bundles/encodec_48khz_12kbps_1333ms",
    );
  }
  if (!out.encodecWasmRoot) {
    out.encodecWasmRoot = out.wasmRoot;
  }
  if ((out.customDecoderRoot === null) !== (out.customKernelModule === null)) {
    throw new Error(
      "--custom-decoder-root and --custom-kernel-module must be provided together",
    );
  }
  if (
    (out.customEncoderRoot === null) !==
    (out.customEncoderKernelModule === null)
  ) {
    throw new Error(
      "--custom-encoder-root and --custom-encoder-kernel-module must be provided together",
    );
  }
  if (out.command === "encode" && out.customDecoderRoot) {
    throw new Error("custom decoder options only apply to decode commands");
  }
  if (out.command === "decode" && out.customEncoderRoot) {
    throw new Error("custom encoder options only apply to encode commands");
  }
  if (out.coding !== "lm") {
    throw new Error(`--coding must be "lm", got ${out.coding}`);
  }
  if (!Number.isInteger(out.threads) || out.threads < 1) {
    throw new Error(`--threads must be a positive integer, got ${out.threads}`);
  }
  if (!["disabled", "basic", "extended", "layout", "all"].includes(out.graphOptimizationLevel)) {
    throw new Error(
      `--graph-optimization-level must be disabled, basic, extended, layout, or all; got ${out.graphOptimizationLevel}`,
    );
  }
  if (!Number.isInteger(out.progressEvery) || out.progressEvery < 1) {
    throw new Error(`--progress-every must be a positive integer, got ${out.progressEvery}`);
  }
  if (!Number.isInteger(out.decodeBatchSize) || out.decodeBatchSize < 1) {
    throw new Error(`--decode-batch-size must be positive, got ${out.decodeBatchSize}`);
  }
  if (!Number.isFinite(out.levelToleranceDb) || out.levelToleranceDb <= 0) {
    throw new Error(
      `--level-tolerance-db must be positive, got ${out.levelToleranceDb}`,
    );
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
      "  --wasm-root <dir> Production WASM asset root",
      "  --encodec-wasm-root <dir> Optional encodec-rs WASM root",
      "  --bundle <dir>    ONNX bundle directory",
      "  --custom-encoder-root <dir> Custom encoder weights and metadata",
      "  --custom-encoder-kernel-module <path> Custom encoder WASM module",
      "  --custom-decoder-root <dir> Split decoder ONNX stages and weights",
      "  --custom-kernel-module <path> Custom ConvTranspose WASM module",
      "  --coding <lm>    ECDC coding mode, fixed to q8 LM",
      "  --lm-backend <q8> LM backend for matrix runs, fixed to q8",
      "  --output <path>   Output ECDC path",
      "  --threads <count> ONNX Runtime WASM thread count",
      "  --graph-optimization-level <level> ONNX graph optimization level",
      "  --enable-cpu-mem-arena Enable the ONNX CPU memory arena",
      "  --enable-mem-pattern Enable ONNX memory-pattern reuse",
      "  --preallocate-decoder-output Reuse the decoder output tensor",
      "  --decode-batch-size <count> Fixed-frame decoder batch size",
      "  --progress-every <n> Progress interval in chunks",
      "  --float-wav       Write IEEE float32 decoded WAV output",
      "  --report <path>   Write the JSON timing report",
      "  --pre-repair-wav <path> Write fixed-context PCM before seam repair",
      "  --seam-pack-root <dir> Write hard-crop and triangle-overlap float WAVs",
      "  --reference-wav <path> Compare decoded PCM with the source WAV",
      "  --level-tolerance-db <dB> Maximum decoded RMS change per channel",
    ].join("\n"),
  );
  process.exit(0);
}

function configureOrt() {
  ort.env.wasm.wasmPaths = pathToFileURL(
    path.join(options.wasmRoot, "onnxruntime-web") + path.sep,
  ).href;
  ort.env.wasm.numThreads = options.threads;
}

function initEncodecWasm() {
  const wasmPath = path.join(options.encodecWasmRoot, "encodec-rs/pkg/encodec_rs_bg.wasm");
  encodec.initSync({ module: readFileSync(wasmPath) });
  encodec.initPanicHook?.();
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
    graphOptimizationLevel: options.graphOptimizationLevel,
    enableCpuMemArena: options.enableCpuMemArena,
    enableMemPattern: options.enableMemPattern,
  });
}

function summarizeOnnxSessionOptions(runOptions) {
  return {
    threads: runOptions.threads,
    graphOptimizationLevel: runOptions.graphOptimizationLevel,
    enableCpuMemArena: runOptions.enableCpuMemArena,
    enableMemPattern: runOptions.enableMemPattern,
    preallocateDecoderOutput: runOptions.preallocateDecoderOutput,
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
      } else if (fmt.subFormatTag === 1 && fmt.bitsPerSample === 24) {
        const unsigned = view.getUint8(cursor)
          | (view.getUint8(cursor + 1) << 8)
          | (view.getUint8(cursor + 2) << 16);
        sample = ((unsigned << 8) >> 8) / 8388608;
      } else if (fmt.subFormatTag === 1 && fmt.bitsPerSample === 32) {
        sample = view.getInt32(cursor, true) / 2147483648;
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

function summarizeSignal(audio, channels, frames) {
  const channelSignals = [];
  let totalEnergy = 0;
  let totalFiniteSamples = 0;
  let peak = 0;
  let nonFiniteSampleCount = 0;
  let clippedSampleCount = 0;

  for (let channel = 0; channel < channels; channel += 1) {
    let energy = 0;
    let finiteSamples = 0;
    let channelPeak = 0;
    let channelNonFinite = 0;
    let channelClipped = 0;
    const start = channel * frames;
    const end = start + frames;
    for (let index = start; index < end; index += 1) {
      const sample = audio[index];
      if (!Number.isFinite(sample)) {
        channelNonFinite += 1;
        continue;
      }
      const magnitude = Math.abs(sample);
      energy += sample * sample;
      finiteSamples += 1;
      channelPeak = Math.max(channelPeak, magnitude);
      if (magnitude >= 1) {
        channelClipped += 1;
      }
    }
    const rms = finiteSamples > 0 ? Math.sqrt(energy / finiteSamples) : null;
    channelSignals.push({
      channel,
      rms: roundSignal(rms),
      rmsDbfs: roundSignalDb(rms),
      peak: roundSignal(channelPeak),
      peakDbfs: roundSignalDb(channelPeak),
      nonFiniteSampleCount: channelNonFinite,
      clippedSampleCount: channelClipped,
    });
    totalEnergy += energy;
    totalFiniteSamples += finiteSamples;
    peak = Math.max(peak, channelPeak);
    nonFiniteSampleCount += channelNonFinite;
    clippedSampleCount += channelClipped;
  }

  const rms = totalFiniteSamples > 0
    ? Math.sqrt(totalEnergy / totalFiniteSamples)
    : null;
  return {
    frames,
    channels,
    rms: roundSignal(rms),
    rmsDbfs: roundSignalDb(rms),
    peak: roundSignal(peak),
    peakDbfs: roundSignalDb(peak),
    nonFiniteSampleCount,
    clippedSampleCount,
    channelSignals,
  };
}

function compareSignalIntegrity(reference, candidate, levelToleranceDb) {
  const referenceSignal = summarizeSignal(
    reference.audio,
    reference.channels,
    reference.frames,
  );
  const candidateSignal = summarizeSignal(
    candidate.audio,
    candidate.channels,
    candidate.frames,
  );
  const formatMatches = reference.sampleRate === candidate.sampleRate
    && reference.channels === candidate.channels
    && reference.frames === candidate.frames;
  const channelLevelDeltasDb = [];
  if (reference.channels === candidate.channels) {
    for (let channel = 0; channel < reference.channels; channel += 1) {
      channelLevelDeltasDb.push(
        levelDeltaDb(
          referenceSignal.channelSignals[channel].rms,
          candidateSignal.channelSignals[channel].rms,
        ),
      );
    }
  }
  const levelWithinTolerance = channelLevelDeltasDb.length === reference.channels
    && channelLevelDeltasDb.every(
      (delta) => delta !== null && Math.abs(delta) <= levelToleranceDb,
    );
  const finite = referenceSignal.nonFiniteSampleCount === 0
    && candidateSignal.nonFiniteSampleCount === 0;
  const unclipped = candidateSignal.clippedSampleCount === 0;
  const noAdditionalClipping =
    candidateSignal.clippedSampleCount <= referenceSignal.clippedSampleCount;

  return {
    pass: formatMatches && levelWithinTolerance && finite && noAdditionalClipping,
    levelToleranceDb,
    checks: {
      formatMatches,
      levelWithinTolerance,
      finite,
      unclipped,
      noAdditionalClipping,
    },
    overallLevelDeltaDb: levelDeltaDb(referenceSignal.rms, candidateSignal.rms),
    channelLevelDeltasDb,
    peakDeltaDb: levelDeltaDb(referenceSignal.peak, candidateSignal.peak),
    reference: {
      sampleRate: reference.sampleRate,
      ...referenceSignal,
    },
    candidate: {
      sampleRate: candidate.sampleRate,
      ...candidateSignal,
    },
  };
}

function comparePcm(reference, candidate) {
  if (reference.length !== candidate.length) {
    return {
      comparable: false,
      referenceSamples: reference.length,
      candidateSamples: candidate.length,
    };
  }
  const referenceBits = new Uint32Array(
    reference.buffer,
    reference.byteOffset,
    reference.length,
  );
  const candidateBits = new Uint32Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.length,
  );
  let exactMismatches = 0;
  let maxAbsError = 0;
  let squaredError = 0;
  let squaredSignal = 0;
  for (let index = 0; index < reference.length; index += 1) {
    if (referenceBits[index] !== candidateBits[index]) {
      exactMismatches += 1;
    }
    const error = reference[index] - candidate[index];
    maxAbsError = Math.max(maxAbsError, Math.abs(error));
    squaredError += error * error;
    squaredSignal += reference[index] * reference[index];
  }
  return {
    comparable: true,
    exact: exactMismatches === 0,
    exactMismatches,
    maxAbsError,
    maxErrorDbfs: maxAbsError === 0 ? null : roundRatio(20 * Math.log10(maxAbsError)),
    rmse: Math.sqrt(squaredError / reference.length),
    snrDb: squaredError === 0 ? null : roundRatio(10 * Math.log10(squaredSignal / squaredError)),
  };
}

function levelDeltaDb(reference, candidate) {
  if (!(reference > 0) || !(candidate > 0)) {
    return reference === candidate ? 0 : null;
  }
  return Number((20 * Math.log10(candidate / reference)).toFixed(3));
}

function roundSignal(value) {
  return value === null ? null : Number(value.toFixed(8));
}

function roundSignalDb(value) {
  return value > 0 ? Number((20 * Math.log10(value)).toFixed(3)) : null;
}

function buildSegmentBatch(audio, audioLength, meta) {
  const starts = segmentStarts(audioLength, meta.segment_stride);
  const context = fixedContextSamples(meta);
  return {
    audio,
    audioLength,
    starts,
    context,
    frameLengths: starts.map((offset) => context === null
      ? segmentFrameLength(Math.min(audioLength - offset, meta.segment_samples), meta.segment_samples, meta.frame_length)
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
      payload: encoder.finish(),
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

    if (typeof decoder.pullAll === "function") {
      const lmStarted = performance.now();
      const symbols = decoder.pullAll(chunk.frameLength);
      lmDeterministicMs += performance.now() - lmStarted;
      for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
        const sourceBase = codebook * chunk.frameLength;
        const targetBase = codebook * meta.frame_length;
        codes.set(symbols.subarray(sourceBase, sourceBase + chunk.frameLength), targetBase);
      }
    } else {
      for (let step = 0; step < chunk.frameLength; step += 1) {
        const lmStarted = performance.now();
        const symbols = decoder.pull();
        lmDeterministicMs += performance.now() - lmStarted;
        for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
          codes[codebook * meta.frame_length + step] = symbols[codebook];
        }
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

function cropFixedContextWithoutRepair(decodedFrames, audioLength, meta) {
  const context = fixedContextSamples(meta);
  if (context === null) {
    throw new Error("--pre-repair-wav requires a fixed-context bundle");
  }
  const frameCount = Math.ceil(audioLength / meta.segment_stride);
  const expected = frameCount * meta.channels * meta.segment_samples;
  if (decodedFrames.length !== expected) {
    throw new Error(
      `decoded frame buffer has ${decodedFrames.length} values; expected ${expected}`,
    );
  }
  const output = new Float32Array(meta.channels * audioLength);
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const offset = frameIndex * meta.segment_stride;
    const ownedSamples = Math.min(meta.segment_stride, audioLength - offset);
    const frameBase = frameIndex * meta.channels * meta.segment_samples;
    for (let channel = 0; channel < meta.channels; channel += 1) {
      const sourceStart = frameBase + channel * meta.segment_samples + context;
      const targetStart = channel * audioLength + offset;
      output.set(
        decodedFrames.subarray(sourceStart, sourceStart + ownedSamples),
        targetStart,
      );
    }
  }
  return output;
}

function writeSeamListeningPack({
  outputRoot,
  hardAudio,
  decodedFrames,
  frameCount,
  audioLength,
  meta,
  inputEcdc,
}) {
  const context = fixedContextSamples(meta);
  if (context === null) {
    throw new Error("--seam-pack-root requires a fixed-context bundle");
  }
  const triangleFull = triangleOverlapAddPlanarFrames(
    decodedFrames,
    frameCount,
    meta.channels,
    meta.segment_samples,
    meta.segment_stride,
  );
  const triangleFullFrames =
    meta.segment_stride * (frameCount - 1) + meta.segment_samples;
  const triangleAudio = cropPlanarAudio(
    triangleFull,
    meta.channels,
    triangleFullFrames,
    context,
    audioLength,
  );

  mkdirSync(outputRoot, { recursive: true });
  const strategies = [
    {
      id: "hard-owned-crop",
      filename: "01-hard-owned-crop.wav",
      audio: hardAudio,
      description: "Crop both 480-sample model guards and concatenate owned PCM unchanged.",
    },
    {
      id: "triangle-guard-overlap",
      filename: "02-triangle-guard-overlap.wav",
      audio: triangleAudio,
      description: "Triangle overlap-add over full decoded model windows. Adjacent windows overlap across both 480-sample guards.",
    },
  ];
  for (const strategy of strategies) {
    writeWav(
      path.join(outputRoot, strategy.filename),
      strategy.audio,
      meta.channels,
      meta.sample_rate,
      true,
    );
  }

  const manifest = {
    schema: "wavey.encodec.seam-listening-pack",
    schemaVersion: 1,
    sourceEcdc: path.relative(repoRoot, inputEcdc),
    sampleRate: meta.sample_rate,
    channels: meta.channels,
    audioSamples: audioLength,
    modelWindows: frameCount,
    modelWindowSamples: meta.segment_samples,
    ownedStrideSamples: meta.segment_stride,
    modelGuardSamplesPerSide: context,
    decodedWindowOverlapSamples: meta.segment_samples - meta.segment_stride,
    upstreamReferenceCommit: "0e2d0aed29362c8e8f52494baf3e6f99056b214f",
    strategies: strategies.map(({ id, filename, description }) => ({
      id,
      wav: filename,
      description,
    })),
  };
  writeFileSync(
    path.join(outputRoot, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  return {
    root: path.relative(repoRoot, outputRoot),
    manifest: path.relative(repoRoot, path.join(outputRoot, "manifest.json")),
    strategies: manifest.strategies,
  };
}

function cropPlanarAudio(audio, channels, frames, start, length) {
  if (audio.length !== channels * frames) {
    throw new Error(
      `planar buffer has ${audio.length} values; expected ${channels * frames}`,
    );
  }
  if (start < 0 || length < 0 || start + length > frames) {
    throw new Error(
      `invalid planar crop start=${start} length=${length} for ${frames} frames`,
    );
  }
  const output = new Float32Array(channels * length);
  for (let channel = 0; channel < channels; channel += 1) {
    const sourceStart = channel * frames + start;
    output.set(
      audio.subarray(sourceStart, sourceStart + length),
      channel * length,
    );
  }
  return output;
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

function writeWav(outputPath, planar, channels, sampleRate, floatOutput = false) {
  const frames = Math.floor(planar.length / channels);
  const bytesPerSample = floatOutput ? 4 : 2;
  const dataBytes = frames * channels * bytesPerSample;
  const out = Buffer.alloc(44 + dataBytes);
  out.write("RIFF", 0, "ascii");
  out.writeUInt32LE(36 + dataBytes, 4);
  out.write("WAVE", 8, "ascii");
  out.write("fmt ", 12, "ascii");
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(floatOutput ? 3 : 1, 20);
  out.writeUInt16LE(channels, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(sampleRate * channels * bytesPerSample, 28);
  out.writeUInt16LE(channels * bytesPerSample, 32);
  out.writeUInt16LE(bytesPerSample * 8, 34);
  out.write("data", 36, "ascii");
  out.writeUInt32LE(dataBytes, 40);
  let cursor = 44;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = planar[channel * frames + frame];
      if (floatOutput) {
        out.writeFloatLE(sample, cursor);
      } else {
        const value = Math.max(-1, Math.min(1, sample));
        out.writeInt16LE(
          value < 0 ? Math.round(value * 32768) : Math.round(value * 32767),
          cursor,
        );
      }
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

function roundRatio(value) {
  return Number(value.toFixed(3));
}

function reportProgress(label, completed, total, started, wav, runOptions) {
  if (completed !== 1 && completed !== total && completed % runOptions.progressEvery !== 0) {
    return;
  }
  let timing = "";
  if (started != null && wav != null) {
    const elapsedSeconds = (performance.now() - started) / 1000;
    const audioSeconds = (wav.frames / wav.sampleRate) * (completed / total);
    timing = `, ${elapsedSeconds.toFixed(1)}s wall, ${(audioSeconds / elapsedSeconds).toFixed(2)}x processed-audio realtime`;
  }
  process.stderr.write(`[wasm-fixture] ${label} ${completed}/${total}${timing}\n`);
}
