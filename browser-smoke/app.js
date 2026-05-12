import * as ort from "./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs";
import init, {
  initPanicHook,
  lmEcdcChunk,
  lmEcdcDecodeChunks,
  lmEcdcHeader,
  LmChunkDecoder,
  LmChunkEncoder,
  rawEcdcDecodeFrames,
  rawEcdcEncode,
  rawEcdcFramePayload,
  rawEcdcHeader,
  rawEcdcMetadata,
  rawEcdcOverlapAdd,
} from "../pkg/encodec_rs.js?v=lm-coding-v2";

ort.env.wasm.wasmPaths = new URL(
  "./node_modules/onnxruntime-web/dist/",
  window.location.href,
).href;
ort.env.wasm.numThreads = 1;
if (ort.env.webgpu) {
  ort.env.webgpu.powerPreference = "high-performance";
}

const els = {
  bundle: document.querySelector("#bundle"),
  runtime: document.querySelector("#runtime"),
  encodeMode: document.querySelector("#encode-mode"),
  coding: document.querySelector("#coding"),
  decodePlayback: document.querySelector("#decode-playback"),
  run: document.querySelector("#run"),
  status: document.querySelector("#status"),
  model: document.querySelector("#model"),
  input: document.querySelector("#input"),
  codes: document.querySelector("#codes"),
  scale: document.querySelector("#scale"),
  ecdc: document.querySelector("#ecdc"),
  ecdcBytes: document.querySelector("#ecdc-bytes"),
  runtimeUsed: document.querySelector("#runtime-used"),
  encodeTime: document.querySelector("#encode-time"),
  decodeTime: document.querySelector("#decode-time"),
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
  const started = performance.now();

  try {
    const runtime = selectedRuntime();
    const encodeMode = els.encodeMode.value;
    const coding = els.coding.value;
    const shouldDecode = els.decodePlayback.checked;
    els.runtimeUsed.textContent = runtime.label;
    setStatus(`Loading wasm and ONNX assets (${runtime.label})`);
    if (shouldDecode) {
      await prepareAudioContext();
    }

    await initWasm();
    const bundleName = els.bundle.value;
    const bundleRoot = `../onnx-bundles/${bundleName}`;
    const bundleJson = await fetchText(`${bundleRoot}/bundle.json`);
    const meta = JSON.parse(bundleJson);
    const sample = await loadJfkAudio(meta);
    const segments = buildSegmentBatch(sample.audio, sample.audioLength, meta);
    const encodeStarted = performance.now();
    const encodeSessionStarted = performance.now();
    const encodeSession = await getSession(bundleName, `${bundleRoot}/${meta.encode_model}`, runtime);
    const lmSession =
      coding === "lm"
        ? await getSession(`${bundleName}:lm`, `${bundleRoot}/${meta.lm_model}`, runtime)
        : null;
    const encodeSessionMs = performance.now() - encodeSessionStarted;
    const encodeSummary =
      coding === "lm"
        ? encodeMode === "incremental"
          ? await encodeLmIncremental(encodeSession, lmSession, bundleJson, sample, segments, meta)
          : await encodeLmBatch(encodeSession, lmSession, bundleJson, sample, segments, meta)
        : encodeMode === "incremental"
          ? await encodeIncremental(encodeSession, bundleJson, sample, segments, meta)
          : await encodeBatch(encodeSession, bundleJson, sample, segments, meta);
    encodeSummary.totalMs = performance.now() - encodeStarted;
    encodeSummary.sessionMs = encodeSessionMs;
    encodeSummary.log.coding = coding;
    encodeSummary.log.runtime = runtime.label;
    encodeSummary.log.executionProviders = runtime.executionProviders;
    encodeSummary.log.timings.sessionMs = roundMs(encodeSessionMs);
    encodeSummary.log.timings.totalMs = roundMs(encodeSummary.totalMs);
    const { ecdc, frames } = encodeSummary;
    const ecdcMeta = rawEcdcMetadata(ecdc);
    let decodeSummary = null;
    if (shouldDecode) {
      decodeSummary = await decodeAndPlay(
        bundleName,
        bundleRoot,
        bundleJson,
        meta,
        ecdc,
        sample.audioLength,
        runtime,
      );
    }
    const elapsed = performance.now() - started;

    els.model.textContent = `${meta.model_name} ${meta.bandwidth_kbps} kbps`;
    els.input.textContent = `${sample.source.sampleRate} Hz ${sample.source.channels}ch -> [${segments.count}, ${meta.channels}, ${meta.segment_samples}]`;
    els.codes.textContent = encodeSummary.codes;
    els.scale.textContent = encodeSummary.scale;
    els.ecdc.textContent = `${sample.audioLength} samples, acv=${ecdcMeta.acv ?? ecdcMeta.bitstream_version ?? 0}`;
    els.ecdcBytes.textContent = `${formatInteger(ecdc.byteLength)} bytes`;
    els.runtimeUsed.textContent = runtime.label;
    els.encodeTime.textContent = formatMs(encodeSummary.totalMs);
    els.decodeTime.textContent = decodeSummary ? formatMs(decodeSummary.totalMs) : "skipped";
    els.elapsed.textContent = formatMs(elapsed);

    writeLog(
      JSON.stringify(
        {
          runtime: {
            selected: runtime.label,
            executionProviders: runtime.executionProviders,
            browserGpu: runtime.id === "webgpu" ? "WebGPU; Metal-backed on macOS browser implementations" : null,
          },
          sessionInputs: encodeSession.inputNames,
          sessionOutputs: encodeSession.outputNames,
          lmSessionInputs: lmSession?.inputNames ?? null,
          lmSessionOutputs: lmSession?.outputNames ?? null,
          encode: encodeSummary.log,
          source: sample.source,
          packagedAudio: {
            audioLength: sample.audioLength,
            segments: segments.count,
            segmentSamples: meta.segment_samples,
            segmentStride: meta.segment_stride,
            firstFrame: summarizeFrame(frames[0]),
            lastFrame: summarizeFrame(frames[frames.length - 1]),
          },
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

async function encodeBatch(session, bundleJson, sample, segments, meta) {
  setStatus(`Running batch encode on ${segments.count} JFK segments`);
  const started = performance.now();
  const feeds = {
    [session.inputNames[0]]: new ort.Tensor("float32", segments.audio, [
      segments.count,
      meta.channels,
      meta.segment_samples,
    ]),
  };
  const onnxStarted = performance.now();
  const outputs = await session.run(feeds);
  const onnxMs = performance.now() - onnxStarted;
  const packStarted = performance.now();
  const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
  const frames = buildRawFrames(codesTensor.data, scaleTensor.data, segments, meta);
  const ecdc = rawEcdcEncode(bundleJson, sample.audioLength, frames);
  const packMs = performance.now() - packStarted;
  const workMs = performance.now() - started;

  return {
    ecdc,
    frames,
    codes: `${codesTensor.type} [${codesTensor.dims.join(", ")}]`,
    scale: `${scaleTensor.type} [${scaleTensor.dims.join(", ")}]`,
    log: {
      mode: "batch",
      emittedChunks: 1,
      timings: {
        onnxMs: roundMs(onnxMs),
        packMs: roundMs(packMs),
        workMs: roundMs(workMs),
      },
      outputSummary: summarizeOutputs(outputs),
    },
  };
}

async function encodeIncremental(session, bundleJson, sample, segments, meta) {
  setStatus(`Writing raw ECDC header and encoding ${segments.count} segments incrementally`);
  const started = performance.now();
  let packStarted = performance.now();
  const chunks = [rawEcdcHeader(bundleJson, sample.audioLength)];
  let packMs = performance.now() - packStarted;
  let onnxMs = 0;
  const frames = [];
  let lastOutputSummary = null;

  for (let index = 0; index < segments.count; index += 1) {
    setStatus(`Incremental encode ${index + 1}/${segments.count}`);
    const segment = buildSingleSegment(sample.audio, sample.audioLength, segments, index, meta);
    const feeds = {
      [session.inputNames[0]]: new ort.Tensor("float32", segment.audio, [
        1,
        meta.channels,
        meta.segment_samples,
      ]),
    };
    const onnxStarted = performance.now();
    const outputs = await session.run(feeds);
    onnxMs += performance.now() - onnxStarted;
    packStarted = performance.now();
    const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
    const frame = buildRawFrame(codesTensor.data, scaleTensor.data, segment, meta, index);
    chunks.push(rawEcdcFramePayload(bundleJson, frame.codes, frame.scale, frame.frameLength));
    packMs += performance.now() - packStarted;
    frames.push(frame);
    lastOutputSummary = summarizeOutputs(outputs);
    await nextAnimationFrame();
  }
  packStarted = performance.now();
  const ecdc = concatUint8Chunks(chunks);
  packMs += performance.now() - packStarted;
  const workMs = performance.now() - started;

  return {
    ecdc,
    frames,
    codes: `int64 [1, ${meta.num_codebooks}, ${meta.frame_length}] x ${segments.count}`,
    scale: `float32 [1, 1] x ${segments.count}`,
    log: {
      mode: "incremental",
      emittedChunks: chunks.length,
      headerBytes: chunks[0].byteLength,
      payloadBytes: chunks.slice(1).reduce((sum, chunk) => sum + chunk.byteLength, 0),
      timings: {
        onnxMs: roundMs(onnxMs),
        packMs: roundMs(packMs),
        workMs: roundMs(workMs),
      },
      lastOutputSummary,
    },
  };
}

async function encodeLmBatch(encodeSession, lmSession, bundleJson, sample, segments, meta) {
  if (!lmSession) {
    throw new Error("LM coding requires lm_logits.onnx");
  }
  setStatus(`Running batch encode on ${segments.count} JFK segments for LM coding`);
  const started = performance.now();
  const feeds = {
    [encodeSession.inputNames[0]]: new ort.Tensor("float32", segments.audio, [
      segments.count,
      meta.channels,
      meta.segment_samples,
    ]),
  };
  const frameStarted = performance.now();
  const outputs = await encodeSession.run(feeds);
  const frameOnnxMs = performance.now() - frameStarted;
  const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
  const frames = buildRawFrames(codesTensor.data, scaleTensor.data, segments, meta);
  const lmSummary = await encodeLmFrames(lmSession, bundleJson, sample.audioLength, frames, meta);
  const workMs = performance.now() - started;

  return {
    ecdc: lmSummary.ecdc,
    frames,
    codes: `${codesTensor.type} [${codesTensor.dims.join(", ")}]`,
    scale: `${scaleTensor.type} [${scaleTensor.dims.join(", ")}]`,
    log: {
      mode: "batch",
      emittedChunks: lmSummary.emittedChunks,
      headerBytes: lmSummary.headerBytes,
      payloadBytes: lmSummary.payloadBytes,
      timings: {
        onnxMs: roundMs(frameOnnxMs + lmSummary.lmOnnxMs),
        frameOnnxMs: roundMs(frameOnnxMs),
        lmOnnxMs: roundMs(lmSummary.lmOnnxMs),
        arithmeticMs: roundMs(lmSummary.arithmeticMs),
        packMs: roundMs(lmSummary.packMs),
        workMs: roundMs(workMs),
      },
      outputSummary: summarizeOutputs(outputs),
      lastLmOutputSummary: lmSummary.lastOutputSummary,
    },
  };
}

async function encodeLmIncremental(encodeSession, lmSession, bundleJson, sample, segments, meta) {
  if (!lmSession) {
    throw new Error("LM coding requires lm_logits.onnx");
  }
  setStatus(`Writing LM ECDC header and encoding ${segments.count} segments incrementally`);
  const started = performance.now();
  let packStarted = performance.now();
  const chunks = [lmEcdcHeader(bundleJson, sample.audioLength)];
  let packMs = performance.now() - packStarted;
  let frameOnnxMs = 0;
  let lmOnnxMs = 0;
  let arithmeticMs = 0;
  const frames = [];
  let lastOutputSummary = null;
  let lastLmOutputSummary = null;

  for (let index = 0; index < segments.count; index += 1) {
    setStatus(`LM incremental encode ${index + 1}/${segments.count}`);
    const segment = buildSingleSegment(sample.audio, sample.audioLength, segments, index, meta);
    const feeds = {
      [encodeSession.inputNames[0]]: new ort.Tensor("float32", segment.audio, [
        1,
        meta.channels,
        meta.segment_samples,
      ]),
    };
    const frameStarted = performance.now();
    const outputs = await encodeSession.run(feeds);
    frameOnnxMs += performance.now() - frameStarted;
    const { codesTensor, scaleTensor } = findEncodeOutputs(outputs);
    const frame = buildRawFrame(codesTensor.data, scaleTensor.data, segment, meta, index);
    const lmFrame = await encodeLmFrame(lmSession, bundleJson, frame, meta);
    lmOnnxMs += lmFrame.lmOnnxMs;
    arithmeticMs += lmFrame.arithmeticMs;
    packStarted = performance.now();
    chunks.push(lmEcdcChunk(lmFrame.payload));
    packMs += performance.now() - packStarted;
    frames.push(frame);
    lastOutputSummary = summarizeOutputs(outputs);
    lastLmOutputSummary = lmFrame.lastOutputSummary;
    await nextAnimationFrame();
  }
  packStarted = performance.now();
  const ecdc = concatUint8Chunks(chunks);
  packMs += performance.now() - packStarted;
  const workMs = performance.now() - started;

  return {
    ecdc,
    frames,
    codes: `int64 [1, ${meta.num_codebooks}, ${meta.frame_length}] x ${segments.count}`,
    scale: `float32 [1, 1] x ${segments.count}`,
    log: {
      mode: "incremental",
      emittedChunks: chunks.length,
      headerBytes: chunks[0].byteLength,
      payloadBytes: chunks.slice(1).reduce((sum, chunk) => sum + chunk.byteLength, 0),
      timings: {
        onnxMs: roundMs(frameOnnxMs + lmOnnxMs),
        frameOnnxMs: roundMs(frameOnnxMs),
        lmOnnxMs: roundMs(lmOnnxMs),
        arithmeticMs: roundMs(arithmeticMs),
        packMs: roundMs(packMs),
        workMs: roundMs(workMs),
      },
      lastOutputSummary,
      lastLmOutputSummary,
    },
  };
}

async function encodeLmFrames(lmSession, bundleJson, audioLength, frames, meta) {
  let packStarted = performance.now();
  const chunks = [lmEcdcHeader(bundleJson, audioLength)];
  let packMs = performance.now() - packStarted;
  let lmOnnxMs = 0;
  let arithmeticMs = 0;
  let lastOutputSummary = null;

  for (let index = 0; index < frames.length; index += 1) {
    setStatus(`LM coding frame ${index + 1}/${frames.length}`);
    const lmFrame = await encodeLmFrame(lmSession, bundleJson, frames[index], meta);
    lmOnnxMs += lmFrame.lmOnnxMs;
    arithmeticMs += lmFrame.arithmeticMs;
    lastOutputSummary = lmFrame.lastOutputSummary;
    packStarted = performance.now();
    chunks.push(lmEcdcChunk(lmFrame.payload));
    packMs += performance.now() - packStarted;
    await nextAnimationFrame();
  }
  packStarted = performance.now();
  const ecdc = concatUint8Chunks(chunks);
  packMs += performance.now() - packStarted;

  return {
    ecdc,
    emittedChunks: chunks.length,
    headerBytes: chunks[0].byteLength,
    payloadBytes: chunks.slice(1).reduce((sum, chunk) => sum + chunk.byteLength, 0),
    lmOnnxMs,
    arithmeticMs,
    packMs,
    lastOutputSummary,
  };
}

async function encodeLmFrame(lmSession, bundleJson, frame, meta) {
  const encoder = new LmChunkEncoder(bundleJson, frame.scale);
  let states = initialLmStates(meta);
  let offset = 0;
  let inputValues = new BigInt64Array(meta.num_codebooks);
  let lmOnnxMs = 0;
  let arithmeticMs = 0;
  let lastOutputSummary = null;

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
    lastOutputSummary = lm.outputSummary;
  }

  return {
    payload: encoder.finish(),
    lmOnnxMs,
    arithmeticMs,
    lastOutputSummary,
  };
}

async function decodeAndPlay(bundleName, bundleRoot, bundleJson, meta, ecdc, audioLength, runtime) {
  const metadata = rawEcdcMetadata(ecdc);
  const acv = metadata.acv ?? metadata.bitstream_version ?? 0;
  if (metadata.lm === true || metadata.use_lm === true || acv === 4 || acv === 5) {
    return decodeLmAndPlay(bundleName, bundleRoot, bundleJson, meta, ecdc, audioLength, runtime);
  }
  return decodeRawAndPlay(bundleName, bundleRoot, bundleJson, meta, ecdc, audioLength, runtime);
}

async function decodeRawAndPlay(bundleName, bundleRoot, bundleJson, meta, ecdc, audioLength, runtime) {
  const started = performance.now();
  setStatus("Parsing raw ECDC frames in wasm");
  const parseStarted = performance.now();
  const parsed = rawEcdcDecodeFrames(bundleJson, ecdc);
  const frames = parsed.frames;
  const decoderInputs = buildDecoderInputs(frames, meta);
  const parseMs = performance.now() - parseStarted;
  setStatus(`Running decode_frame.onnx on ${frames.length} JFK segments`);

  const sessionStarted = performance.now();
  const session = await getSession(`${bundleName}:decode`, `${bundleRoot}/${meta.decode_model}`, runtime);
  const sessionMs = performance.now() - sessionStarted;
  const feeds = {
    [session.inputNames[0]]: new ort.Tensor("int64", decoderInputs.codes, [
      frames.length,
      meta.num_codebooks,
      meta.frame_length,
    ]),
    [session.inputNames[1]]: new ort.Tensor("float32", decoderInputs.scales, [frames.length, 1]),
  };
  const onnxStarted = performance.now();
  const outputs = await session.run(feeds);
  const onnxMs = performance.now() - onnxStarted;
  const decodedTensor = findDecodeOutput(outputs);

  setStatus("Overlap-adding decoded frames in wasm");
  const overlapStarted = performance.now();
  const decodedAudio = rawEcdcOverlapAdd(bundleJson, audioLength, decodedTensor.data);
  const overlapMs = performance.now() - overlapStarted;
  playPlanarAudio(decodedAudio, meta.channels, meta.sample_rate);
  const totalMs = performance.now() - started;

  return {
    parsedFrames: frames.length,
    decoderInputs: session.inputNames,
    decoderOutputs: session.outputNames,
    decodedShape: decodedTensor.dims,
    playbackSeconds: Number((audioLength / meta.sample_rate).toFixed(3)),
    totalMs,
    timings: {
      parseMs: roundMs(parseMs),
      sessionMs: roundMs(sessionMs),
      onnxMs: roundMs(onnxMs),
      overlapMs: roundMs(overlapMs),
      totalMs: roundMs(totalMs),
    },
  };
}

async function decodeLmAndPlay(bundleName, bundleRoot, bundleJson, meta, ecdc, audioLength, runtime) {
  const started = performance.now();
  setStatus("Parsing LM ECDC chunks in wasm");
  const parseStarted = performance.now();
  const parsed = lmEcdcDecodeChunks(bundleJson, ecdc);
  const parseMs = performance.now() - parseStarted;

  setStatus(`Running lm_logits.onnx on ${parsed.chunks.length} LM chunks`);
  const lmSessionStarted = performance.now();
  const lmSession = await getSession(`${bundleName}:lm`, `${bundleRoot}/${meta.lm_model}`, runtime);
  const lmSessionMs = performance.now() - lmSessionStarted;

  const frames = [];
  let lmOnnxMs = 0;
  let arithmeticMs = 0;
  let lastLmOutputSummary = null;
  for (let index = 0; index < parsed.chunks.length; index += 1) {
    setStatus(`LM decode chunk ${index + 1}/${parsed.chunks.length}`);
    const frame = await decodeLmFrame(lmSession, bundleJson, meta, parsed.chunks[index]);
    frames.push(frame.frame);
    lmOnnxMs += frame.lmOnnxMs;
    arithmeticMs += frame.arithmeticMs;
    lastLmOutputSummary = frame.lastOutputSummary;
    await nextAnimationFrame();
  }

  const decoderInputs = buildDecoderInputs(frames, meta);
  setStatus(`Running decode_frame.onnx on ${frames.length} JFK segments`);
  const decodeSessionStarted = performance.now();
  const decodeSession = await getSession(`${bundleName}:decode`, `${bundleRoot}/${meta.decode_model}`, runtime);
  const decodeSessionMs = performance.now() - decodeSessionStarted;
  const feeds = {
    [decodeSession.inputNames[0]]: new ort.Tensor("int64", decoderInputs.codes, [
      frames.length,
      meta.num_codebooks,
      meta.frame_length,
    ]),
    [decodeSession.inputNames[1]]: new ort.Tensor("float32", decoderInputs.scales, [frames.length, 1]),
  };
  const decodeStarted = performance.now();
  const outputs = await decodeSession.run(feeds);
  const decodeOnnxMs = performance.now() - decodeStarted;
  const decodedTensor = findDecodeOutput(outputs);

  setStatus("Overlap-adding decoded frames in wasm");
  const overlapStarted = performance.now();
  const decodedAudio = rawEcdcOverlapAdd(bundleJson, audioLength, decodedTensor.data);
  const overlapMs = performance.now() - overlapStarted;
  playPlanarAudio(decodedAudio, meta.channels, meta.sample_rate);
  const totalMs = performance.now() - started;

  return {
    parsedFrames: frames.length,
    lmInputs: lmSession.inputNames,
    lmOutputs: lmSession.outputNames,
    decoderInputs: decodeSession.inputNames,
    decoderOutputs: decodeSession.outputNames,
    decodedShape: decodedTensor.dims,
    playbackSeconds: Number((audioLength / meta.sample_rate).toFixed(3)),
    totalMs,
    timings: {
      parseMs: roundMs(parseMs),
      lmSessionMs: roundMs(lmSessionMs),
      lmOnnxMs: roundMs(lmOnnxMs),
      arithmeticMs: roundMs(arithmeticMs),
      decodeSessionMs: roundMs(decodeSessionMs),
      decodeOnnxMs: roundMs(decodeOnnxMs),
      overlapMs: roundMs(overlapMs),
      totalMs: roundMs(totalMs),
    },
    lastLmOutputSummary,
  };
}

async function decodeLmFrame(lmSession, bundleJson, meta, chunk) {
  const decoder = new LmChunkDecoder(bundleJson, Uint8Array.from(chunk.payload));
  let states = initialLmStates(meta);
  let offset = 0;
  let inputValues = new BigInt64Array(meta.num_codebooks);
  const codes = new Uint16Array(meta.num_codebooks * meta.frame_length);
  let lmOnnxMs = 0;
  let arithmeticMs = 0;
  let lastOutputSummary = null;

  for (let step = 0; step < chunk.frameLength; step += 1) {
    const lmStarted = performance.now();
    const lm = await runLmStep(lmSession, meta, inputValues, offset, states);
    lmOnnxMs += performance.now() - lmStarted;
    const arithmeticStarted = performance.now();
    const symbols = decoder.pull(lm.logits.data);
    arithmeticMs += performance.now() - arithmeticStarted;
    for (let codebook = 0; codebook < meta.num_codebooks; codebook += 1) {
      codes[codebook * meta.frame_length + step] = symbols[codebook];
    }
    inputValues = lmInputFromCodes(symbols);
    states = lm.nextStates;
    offset = lm.nextOffset;
    lastOutputSummary = lm.outputSummary;
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
    arithmeticMs,
    lastOutputSummary,
  };
}

async function initWasm() {
  if (!wasmReady) {
    wasmReady = init(new URL("../pkg/encodec_rs_bg.wasm?v=lm-coding-v2", window.location.href).href).then(() => {
      initPanicHook();
    });
  }
  return wasmReady;
}

async function getSession(bundleName, modelPath, runtime) {
  const cacheKey = `${runtime.id}:${bundleName}:${modelPath}`;
  if (!sessionCache.has(cacheKey)) {
    sessionCache.set(
      cacheKey,
      loadModelForOrt(modelPath).then((model) =>
        ort.InferenceSession.create(model, {
          executionProviders: [...runtime.executionProviders],
          graphOptimizationLevel: "all",
        }),
      ),
    );
  }
  return sessionCache.get(cacheKey);
}

async function loadModelForOrt(modelPath) {
  const partsManifestUrl = new URL(`${modelPath}.parts.json`, window.location.href).href;
  const manifestResponse = await fetch(partsManifestUrl, { cache: "force-cache" });
  if (manifestResponse.status === 404) {
    return modelPath;
  }
  if (!manifestResponse.ok) {
    throw new Error(`Failed to fetch ${partsManifestUrl}: ${manifestResponse.status}`);
  }

  const manifest = await manifestResponse.json();
  if (!Array.isArray(manifest.parts) || !Number.isInteger(manifest.byteLength)) {
    throw new Error(`Invalid model parts manifest: ${partsManifestUrl}`);
  }

  const chunks = await Promise.all(
    manifest.parts.map(async (part) => new Uint8Array(await fetchArrayBuffer(new URL(part, partsManifestUrl).href))),
  );
  const model = concatUint8Chunks(chunks);
  if (model.byteLength !== manifest.byteLength) {
    throw new Error(
      `Model parts for ${modelPath} produced ${model.byteLength} bytes, expected ${manifest.byteLength}`,
    );
  }
  return model;
}

function selectedRuntime() {
  if (els.runtime.value === "webgpu") {
    if (!navigator.gpu) {
      throw new Error(webGpuUnavailableMessage());
    }
    return {
      id: "webgpu",
      label: "WebGPU (Metal-backed on macOS)",
      executionProviders: ["webgpu", "wasm"],
    };
  }

  return {
    id: "wasm",
    label: "WASM CPU",
    executionProviders: ["wasm"],
  };
}

function webGpuUnavailableMessage() {
  const context = `secureContext=${window.isSecureContext} userAgent=${navigator.userAgent}`;
  if (isSafariBrowser()) {
    return [
      "WebGPU is not exposed by this Safari context.",
      "To enable it: Safari > Settings > Advanced > enable Show features for web developers.",
      "Then open Develop > Feature Flags, search WebGPU, and enable WebGPU.",
      "If present, also enable GPU Process: DOM Rendering and GPU Process: Canvas Rendering.",
      "Quit Safari with Cmd+Q, reopen it, reload this page, and check navigator.gpu in the console.",
      "If navigator.gpu is still undefined, use Safari Technology Preview or Safari 26+.",
      "Apple Silicon support is not enough by itself; the browser must expose navigator.gpu.",
      context,
    ].join(" ");
  }

  return [
    "WebGPU is not exposed by this browser context.",
    "Select WASM CPU or use a WebGPU-capable browser.",
    "Apple Silicon support is not enough by itself; the page needs navigator.gpu.",
    context,
  ].join(" ");
}

function isSafariBrowser() {
  const ua = navigator.userAgent;
  return /\bSafari\//.test(ua) && /\bVersion\//.test(ua) && !/\b(Chrome|Chromium|CriOS|FxiOS|Edg|OPR)\//.test(ua);
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

function initialLmStates(meta) {
  return Array.from({ length: meta.lm_num_layers }, () => ({
    data: new Float32Array(meta.lm_dim),
    dims: [1, 1, meta.lm_dim],
  }));
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
  const logits = outputs.logits ?? findLmLogitsOutput(outputs, meta);
  const offsetTensor = outputs.offset_out ?? findLmOffsetOutput(outputs);
  const nextStates = states.map((_, index) => {
    const tensor = outputs[`next_state_${index}`];
    if (!tensor) {
      throw new Error(`LM output next_state_${index} was not returned`);
    }
    return {
      data: tensor.data,
      dims: tensor.dims,
    };
  });
  return {
    logits,
    nextOffset: Number(offsetTensor.data[0]),
    nextStates,
    outputSummary: summarizeOutputs(outputs),
  };
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
    throw new Error(`Unexpected LM outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return tensor;
}

function findLmOffsetOutput(outputs) {
  const tensor = Object.values(outputs).find((candidate) => candidate.type === "int64" && candidate.data.length === 1);
  if (!tensor) {
    throw new Error(`Unexpected LM outputs: ${JSON.stringify(summarizeOutputs(outputs))}`);
  }
  return tensor;
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
    const base = batchIndex * valuesPerSegment;
    frames.push(
      buildRawFrame(
        codes.subarray(base, base + valuesPerSegment),
        scales.subarray(batchIndex, batchIndex + 1),
        {
          offset: segments.starts[batchIndex],
          samples: Math.min(segments.audioLength - segments.starts[batchIndex], meta.segment_samples),
          frameLength: segments.frameLengths[batchIndex],
        },
        meta,
        batchIndex,
      ),
    );
  }
  return frames;
}

function buildRawFrame(codes, scales, segment, meta, segmentIndex) {
  const valuesPerSegment = meta.num_codebooks * meta.frame_length;
  if (codes.length !== valuesPerSegment) {
    throw new Error(`Segment ${segmentIndex} codes length ${codes.length} does not match ${valuesPerSegment}`);
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

function nextAnimationFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

function clearMetrics() {
  for (const key of [
    "model",
    "input",
    "codes",
    "scale",
    "ecdc",
    "ecdcBytes",
    "runtimeUsed",
    "encodeTime",
    "decodeTime",
    "elapsed",
  ]) {
    els[key].textContent = "-";
  }
  writeLog("");
}

function formatMs(ms) {
  return `${ms.toFixed(1)} ms`;
}

function formatInteger(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function roundMs(ms) {
  return Number(ms.toFixed(1));
}

function setStatus(message, kind = "") {
  els.status.textContent = message;
  els.status.className = `status ${kind}`.trim();
}

function writeLog(message) {
  els.log.textContent = message;
}
