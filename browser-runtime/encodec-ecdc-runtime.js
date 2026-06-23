"use strict";

function requiredFunction(value, name) {
  if (typeof value !== "function") {
    throw new Error(`encodec-rs ECDC runtime requires ${name}.`);
  }
  return value;
}

function normalizeBaseUrl(value) {
  return String(value || "").replace(/\/?$/, "/");
}

function normalizeBytes(value, name) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);

  if (
    typeof SharedArrayBuffer !== "undefined" &&
    value instanceof SharedArrayBuffer
  ) {
    return new Uint8Array(value);
  }

  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(
      value.buffer,
      value.byteOffset,
      value.byteLength,
    );
  }

  throw new Error(`${name} must be an array buffer-like value.`);
}

function normalizeFloat32(value, name) {
  if (value instanceof Float32Array) return value;
  if (value instanceof ArrayBuffer) return new Float32Array(value);

  if (
    typeof SharedArrayBuffer !== "undefined" &&
    value instanceof SharedArrayBuffer
  ) {
    return new Float32Array(value);
  }

  if (ArrayBuffer.isView(value)) {
    if (value.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error(
        `${name} byte length must be divisible by ${Float32Array.BYTES_PER_ELEMENT}.`,
      );
    }

    return new Float32Array(
      value.buffer,
      value.byteOffset,
      value.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
  }

  throw new Error(`${name} must be Float32Array-like.`);
}

function disposeTensor(tensor) {
  try {
    tensor?.dispose?.();
  } catch (_error) {
    // Tensor disposal is best-effort.
  }
}

function disposeTensorMap(tensors) {
  Object.values(tensors || {}).forEach(disposeTensor);
}

function summarizeOrtOutputs(outputs) {
  return Object.fromEntries(
    Object.entries(outputs || {}).map(([name, tensor]) => [
      name,
      {
        type: tensor.type,
        dims: tensor.dims,
        length: tensor.data?.length || 0,
      },
    ]),
  );
}

function findEncodeOutputs(outputs) {
  const tensors = Object.values(outputs || {});

  const codesTensor = tensors.find(
    (tensor) => tensor.type === "int64",
  );

  const scaleTensor = tensors.find(
    (tensor) =>
      tensor.type === "float32" &&
      tensor.dims.length === 2,
  );

  if (!codesTensor || !scaleTensor) {
    throw new Error(
      `Unexpected EnCodec encoder outputs: ${JSON.stringify(
        summarizeOrtOutputs(outputs),
      )}`,
    );
  }

  return {
    codesTensor,
    scaleTensor,
  };
}

function toU16Codes(codes, segmentIndex) {
  const frameCodes = new Uint16Array(codes.length);

  for (let index = 0; index < codes.length; index += 1) {
    const value = Number(codes[index]);

    if (
      !Number.isInteger(value) ||
      value < 0 ||
      value > 65535
    ) {
      throw new Error(
        `Invalid EnCodec code at ${segmentIndex}:${index}: ${String(
          codes[index],
        )}`,
      );
    }

    frameCodes[index] = value;
  }

  return frameCodes;
}

function positiveInteger(value, name) {
  const number = Math.floor(Number(value));

  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }

  return number;
}

function normalizeEncodeMessage(payload = {}) {
  return {
    sessionKey: String(payload.sessionKey || "default"),
    bundleRoot: String(payload.bundleRoot || ""),
    segment: payload.segment,
    segmentIndex: Math.max(
      0,
      Math.floor(Number(payload.segmentIndex) || 0),
    ),
    segmentSamples: Math.max(
      0,
      Math.floor(Number(payload.segmentSamples) || 0),
    ),
    segmentFrameLength: Math.max(
      0,
      Math.floor(Number(payload.segmentFrameLength) || 0),
    ),
    lmWeights: payload.lmWeights,
    bundleJson:
      typeof payload.bundleJson === "string"
        ? payload.bundleJson
        : "",
  };
}

export function createEncodecEcdcRuntime({
  encodecWasmBaseUrl = "./wasm/encodec-rs/",
  onnxRuntimeBaseUrl = "./wasm/onnxruntime-web/",
  globalScope = globalThis,
  versionedAssetUrl = (path) => path,
  isCancelled = () => false,
  createCancelledError = () => new Error("Upload cancelled"),
} = {}) {
  const versionAsset = requiredFunction(
    versionedAssetUrl,
    "versionedAssetUrl",
  );

  const checkCancelled = requiredFunction(
    isCancelled,
    "isCancelled",
  );

  const makeCancelledError = requiredFunction(
    createCancelledError,
    "createCancelledError",
  );

  const encodecBase = normalizeBaseUrl(encodecWasmBaseUrl);
  const ortBase = normalizeBaseUrl(onnxRuntimeBaseUrl);

  let encodecWasmModulePromise = null;
  let onnxRuntimeModulePromise = null;

  const encodeSessionCache = new Map();
  const encodeJobState = new Map();

  function throwIfCancelled(requestId) {
    if (checkCancelled(requestId)) {
      throw makeCancelledError();
    }
  }

  function versionedUrl(path) {
    return String(versionAsset(path));
  }

  function versionedRelativeUrl(path, baseUrl) {
    return versionedUrl(
      new URL(path, baseUrl).toString(),
    );
  }

  async function ensureEncodecWasmModule() {
    if (!encodecWasmModulePromise) {
      encodecWasmModulePromise = (async () => {
        const module = await import(
          versionedUrl(`${encodecBase}pkg/encodec_rs.js`)
        );

        await module.default({
          module_or_path: versionedUrl(
            `${encodecBase}pkg/encodec_rs_bg.wasm`,
          ),
        });

        module.initPanicHook?.();

        if (
          typeof module.ecdcEncodeModelInput !== "function"
        ) {
          throw new Error(
            "encodec-rs WASM is missing ecdcEncodeModelInput; rebuild encodec-rs fixed bundles.",
          );
        }

        if (
          typeof module.lmEcdcChunkFromFrame !== "function"
        ) {
          throw new Error(
            "encodec-rs WASM is missing lmEcdcChunkFromFrame; rebuild encodec-rs fixed bundles.",
          );
        }

        return module;
      })();
    }

    return encodecWasmModulePromise;
  }

  async function ensureOnnxRuntimeModule() {
    if (!onnxRuntimeModulePromise) {
      onnxRuntimeModulePromise = (async () => {
        const ort = await import(
          versionedUrl(`${ortBase}ort.wasm.min.mjs`)
        );

        ort.env.wasm.wasmPaths = new URL(
          ortBase,
          globalScope.location.href,
        ).href;

        ort.env.wasm.numThreads =
          globalScope.crossOriginIsolated
            ? Math.max(
              1,
              Math.min(
                4,
                Math.floor(
                  globalScope.navigator?.hardwareConcurrency ||
                  2,
                ),
              ),
            )
            : 1;

        return ort;
      })();
    }

    return onnxRuntimeModulePromise;
  }

  async function fetchArrayBuffer(
    url,
    { cache = "forc[118;1:3ue-cache" } = {},
  ) {
    const response = await globalScope.fetch(url, {
      cache,
    });

    if (!response.ok) {
      const detail = await response
        .text()
        .catch(() => "");

      throw new Error(
        `Failed to fetch ${url}: ${response.status}` +
        (detail ? ` ${detail.slice(0, 180)}` : ""),
      );
    }

    return response.arrayBuffer();
  }

  function concatenateUint8Chunks(
    chunks,
    totalLength = chunks.reduce(
      (sum, chunk) => sum + chunk.length,
      0,
    ),
  ) {
    const output = new Uint8Array(totalLength);
    let cursor = 0;

    for (const chunk of chunks) {
      output.set(chunk, cursor);
      cursor += chunk.length;
    }

    return output;
  }

  async function fetchSplitArrayBuffer(assetPath) {
    const partsManifestUrl = versionedUrl(
      `${assetPath}.parts.json`,
    );

    const manifestResponse = await globalScope.fetch(
      partsManifestUrl,
      {
        cache: "force-cache",
      },
    );

    if (!manifestResponse.ok) {
      throw new Error(
        `Failed to fetch ${partsManifestUrl}: ${manifestResponse.status}`,
      );
    }

    const manifest = await manifestResponse.json();

    if (
      !Array.isArray(manifest.parts) ||
      !Number.isInteger(manifest.byteLength)
    ) {
      throw new Error(
        `Invalid asset parts manifest: ${partsManifestUrl}`,
      );
    }

    const chunks = await Promise.all(
      manifest.parts.map(async (part) => {
        const url = versionedRelativeUrl(
          part,
          partsManifestUrl,
        );

        return new Uint8Array(
          await fetchArrayBuffer(url),
        );
      }),
    );

    const bytes = concatenateUint8Chunks(chunks);

    if (bytes.byteLength !== manifest.byteLength) {
      throw new Error(
        `Asset parts for ${assetPath} produced ${bytes.byteLength} bytes, expected ${manifest.byteLength}`,
      );
    }

    return bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
  }

  async function loadModelForOrt(
    modelPath,
    assetMetadata = null,
  ) {
    if (assetMetadata?.split === false) {
      return versionedUrl(modelPath);
    }

    if (assetMetadata?.split === true) {
      return fetchSplitArrayBuffer(modelPath);
    }

    const partsManifestUrl = versionedUrl(
      `${modelPath}.parts.json`,
    );

    const manifestResponse = await globalScope.fetch(
      partsManifestUrl,
      {
        cache: "force-cache",
      },
    );

    if (manifestResponse.status === 404) {
      return versionedUrl(modelPath);
    }

    if (!manifestResponse.ok) {
      throw new Error(
        `Failed to fetch ${partsManifestUrl}: ${manifestResponse.status}`,
      );
    }

    const manifest = await manifestResponse.json();

    if (
      !Array.isArray(manifest.parts) ||
      !Number.isInteger(manifest.byteLength)
    ) {
      throw new Error(
        `Invalid model parts manifest: ${partsManifestUrl}`,
      );
    }

    const modelFilename = modelPath.split("/").pop();

    if (
      manifest.parts.length === 1 &&
      manifest.parts[0] === modelFilename
    ) {
      return versionedUrl(modelPath);
    }

    const chunks = await Promise.all(
      manifest.parts.map(async (part) => {
        const url = versionedRelativeUrl(
          part,
          partsManifestUrl,
        );

        return new Uint8Array(
          await fetchArrayBuffer(url),
        );
      }),
    );

    const model = concatenateUint8Chunks(chunks);

    if (model.byteLength !== manifest.byteLength) {
      throw new Error(
        `Model parts for ${modelPath} produced ${model.byteLength} bytes, expected ${manifest.byteLength}`,
      );
    }

    return model.buffer.slice(
      model.byteOffset,
      model.byteOffset + model.byteLength,
    );
  }

  function sessionCacheKey(modelPath, sessionKey) {
    return `${sessionKey || "default"}:${modelPath || ""}`;
  }

  function bundleAssetMetadata(meta, assetName) {
    const assets =
      meta.bitneedle_player_assets ||
      meta.bitneedlePlayerAssets ||
      {};

    return assets?.[assetName] || null;
  }

  function bundleBrowserAssetMetadata(bundleJson) {
    try {
      const meta = JSON.parse(bundleJson);

      const assets =
        meta.bitneedle_player_assets ||
        meta.bitneedlePlayerAssets;

      return assets && typeof assets === "object"
        ? assets
        : null;
    } catch (_error) {
      return null;
    }
  }

  function mergeBundleBrowserAssetMetadata(
    meta,
    bundleJson,
  ) {
    const assets =
      bundleBrowserAssetMetadata(bundleJson);

    return assets
      ? {
        ...meta,
        bitneedle_player_assets: assets,
      }
      : meta;
  }

  async function getEncodeSession(
    sessionKey,
    modelPath,
    assetMetadata = null,
  ) {
    const cacheKey = sessionCacheKey(
      modelPath,
      sessionKey,
    );

    if (!encodeSessionCache.has(cacheKey)) {
      const sessionPromise = (async () => {
        const ort = await ensureOnnxRuntimeModule();

        const model = await loadModelForOrt(
          modelPath,
          assetMetadata,
        );

        return ort.InferenceSession.create(model, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
      })().catch((error) => {
        encodeSessionCache.delete(cacheKey);
        throw error;
      });

      encodeSessionCache.set(
        cacheKey,
        sessionPromise,
      );
    }

    return encodeSessionCache.get(cacheKey);
  }

  function getJobState(sessionKey) {
    const key = String(sessionKey || "default");

    let state = encodeJobState.get(key);

    if (!state) {
      state = {
        bundleJson: "",
        bundleMeta: null,
        bundleRoot: "",
        lmWeights: null,
      };

      encodeJobState.set(key, state);
    }

    return state;
  }

  function updateJobState(
    state,
    data,
    encodecModule,
  ) {
    if (data.bundleRoot) {
      state.bundleRoot = data.bundleRoot.replace(
        /\/?$/,
        "",
      );
    }

    if (data.bundleJson) {
      state.bundleJson = data.bundleJson;

      state.bundleMeta =
        mergeBundleBrowserAssetMetadata(
          encodecModule.bundleMetadata(
            data.bundleJson,
          ),
          data.bundleJson,
        );
    }

    if (data.lmWeights) {
      state.lmWeights = normalizeBytes(
        data.lmWeights,
        "lmWeights",
      );
    }
  }

  async function encodeEcdcChunk(
    payload,
    { requestId = null } = {},
  ) {
    const data = normalizeEncodeMessage(payload);
    const encodecModule =
      await ensureEncodecWasmModule();

    const state = getJobState(data.sessionKey);

    updateJobState(
      state,
      data,
      encodecModule,
    );

    if (!state.bundleJson || !state.bundleMeta) {
      throw new Error(
        "bundleJson was not provided for this encode job.",
      );
    }

    if (!state.lmWeights?.byteLength) {
      throw new Error(
        "lmWeights were not provided for this encode job.",
      );
    }

    if (!state.bundleRoot) {
      throw new Error(
        "bundleRoot was not provided for this encode job.",
      );
    }

    throwIfCancelled(requestId);

    const meta = state.bundleMeta;

    const channels = positiveInteger(
      meta.channels,
      "bundle channels",
    );

    const modelSegmentSamples = positiveInteger(
      meta.segment_samples,
      "bundle segment_samples",
    );

    const frameLength = positiveInteger(
      meta.frame_length,
      "bundle frame_length",
    );

    const numCodebooks = positiveInteger(
      meta.num_codebooks,
      "bundle num_codebooks",
    );

    const encodeModelName = String(
      meta.encode_model || "encode_frame.onnx",
    );

    const encodeModelPath =
      `${state.bundleRoot}/${encodeModelName}`;

    const encodeModelAssetMetadata =
      bundleAssetMetadata(
        meta,
        encodeModelName,
      );

    const segmentSamples = data.segmentSamples;
    const segmentFrameLength =
      data.segmentFrameLength;

    if (segmentSamples <= 0) {
      return new Uint8Array();
    }

    if (segmentFrameLength <= 0) {
      throw new Error(
        "segmentFrameLength was not provided for this encode segment.",
      );
    }

    if (segmentFrameLength > frameLength) {
      throw new Error(
        `segmentFrameLength ${segmentFrameLength} exceeds bundle frame_length ${frameLength}.`,
      );
    }

    const segmentAudio = normalizeFloat32(
      data.segment,
      "segment",
    );

    const expectedSegmentSamples =
      channels * segmentSamples;

    if (
      segmentAudio.length !==
      expectedSegmentSamples
    ) {
      throw new Error(
        `segment contains ${segmentAudio.length} samples, expected ` +
        `${channels} channels * ${segmentSamples} samples = ` +
        `${expectedSegmentSamples}.`,
      );
    }

    const modelInput =
      encodecModule.ecdcEncodeModelInput(
        state.bundleJson,
        segmentAudio,
      );

    const inputAudio =
      modelInput instanceof Float32Array
        ? modelInput
        : new Float32Array(modelInput);

    const expectedInputSamples =
      channels * modelSegmentSamples;

    if (
      inputAudio.length !==
      expectedInputSamples
    ) {
      throw new Error(
        `encodec-rs produced ${inputAudio.length} model input samples, ` +
        `expected ${expectedInputSamples}.`,
      );
    }

    const [ort, encodeSession] =
      await Promise.all([
        ensureOnnxRuntimeModule(),
        getEncodeSession(
          data.sessionKey,
          encodeModelPath,
          encodeModelAssetMetadata,
        ),
      ]);

    throwIfCancelled(requestId);

    let feeds = null;
    let outputs = null;

    try {
      feeds = {
        [encodeSession.inputNames[0]]:
          new ort.Tensor(
            "float32",
            inputAudio,
            [
              1,
              channels,
              modelSegmentSamples,
            ],
          ),
      };

      outputs =
        await encodeSession.run(feeds);

      throwIfCancelled(requestId);

      const {
        codesTensor,
        scaleTensor,
      } = findEncodeOutputs(outputs);

      const expectedCodes =
        numCodebooks * frameLength;

      if (
        codesTensor.data.length !==
        expectedCodes
      ) {
        throw new Error(
          `EnCodec encoder returned ${codesTensor.data.length} codes, ` +
          `expected ${expectedCodes}.`,
        );
      }

      const frameCodes = toU16Codes(
        codesTensor.data,
        data.segmentIndex,
      );

      const rawScale = Number(
        scaleTensor.data?.[0] ?? 1,
      );

      const scale = Number.isFinite(rawScale)
        ? rawScale
        : 1;

      const chunk =
        encodecModule.lmEcdcChunkFromFrame(
          state.bundleJson,
          state.lmWeights,
          scale,
          frameCodes,
          segmentFrameLength,
        );

      return chunk instanceof Uint8Array
        ? chunk
        : new Uint8Array(chunk);
    } finally {
      disposeTensorMap(feeds);
      disposeTensorMap(outputs);
    }
  }

  return Object.freeze({
    clearSessionCache() {
      encodeSessionCache.clear();
    },

    encodeEcdcChunk,
  });
}
