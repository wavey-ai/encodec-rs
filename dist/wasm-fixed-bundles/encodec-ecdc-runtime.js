"use strict";

import { createCustomEncoder } from "./custom-encoder-runtime.js";

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
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
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
      throw new Error(`${name} byte length must be divisible by four.`);
    }
    return new Float32Array(
      value.buffer,
      value.byteOffset,
      value.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
  }
  throw new Error(`${name} must be Float32Array-like.`);
}

function positiveInteger(value, name) {
  const number = Math.floor(Number(value));
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return number;
}

function toU16Codes(codes, segmentIndex) {
  const frameCodes = new Uint16Array(codes.length);
  for (let index = 0; index < codes.length; index += 1) {
    const value = Number(codes[index]);
    if (!Number.isInteger(value) || value < 0 || value > 65535) {
      throw new Error(
        `Invalid EnCodec code at ${segmentIndex}:${index}: ${String(codes[index])}`,
      );
    }
    frameCodes[index] = value;
  }
  return frameCodes;
}

function normalizeEncodeMessage(payload = {}) {
  return {
    sessionKey: String(payload.sessionKey || "default"),
    bundleRoot: String(payload.bundleRoot || ""),
    segment: payload.segment,
    segmentIndex: Math.max(0, Math.floor(Number(payload.segmentIndex) || 0)),
    segmentSamples: Math.max(0, Math.floor(Number(payload.segmentSamples) || 0)),
    segmentFrameLength: Math.max(
      0,
      Math.floor(Number(payload.segmentFrameLength) || 0),
    ),
    lmWeights: payload.lmWeights,
    bundleJson: typeof payload.bundleJson === "string" ? payload.bundleJson : "",
  };
}

function encoderAssetConfig(bundleConfig) {
  const runtime = bundleConfig?.custom_wasm_runtime;
  const encoder = runtime?.encoder || {};
  return {
    assetRoot: String(encoder.asset_root || "encoder/"),
    kernel: String(encoder.kernel || "encodec-encoder-relaxed.mjs"),
    fallbackKernel: String(encoder.fallback_kernel || "encodec-encoder.mjs"),
  };
}

export function createEncodecEcdcRuntime({
  encodecWasmBaseUrl = "./wasm/encodec-rs/",
  globalScope = globalThis,
  versionedAssetUrl = (asset) => asset,
  isCancelled = () => false,
  createCancelledError = () => new Error("Upload cancelled"),
} = {}) {
  const versionAsset = requiredFunction(versionedAssetUrl, "versionedAssetUrl");
  const checkCancelled = requiredFunction(isCancelled, "isCancelled");
  const makeCancelledError = requiredFunction(
    createCancelledError,
    "createCancelledError",
  );
  const fetchImpl = requiredFunction(globalScope.fetch, "globalScope.fetch");
  const encodecBase = normalizeBaseUrl(encodecWasmBaseUrl);
  const encodeJobState = new Map();
  let encodecWasmModulePromise = null;
  let cachedEncoderKey = "";
  let cachedEncoderPromise = null;
  let encodeInferenceTail = Promise.resolve();

  function throwIfCancelled(requestId) {
    if (checkCancelled(requestId)) throw makeCancelledError();
  }

  function versionedUrl(asset) {
    return String(versionAsset(String(asset)));
  }

  async function versionedFetch(input, init) {
    const url = input instanceof URL ? input.href : String(input);
    return fetchImpl.call(globalScope, versionedUrl(url), init);
  }

  async function ensureEncodecWasmModule() {
    if (!encodecWasmModulePromise) {
      encodecWasmModulePromise = (async () => {
        const module = await import(
          versionedUrl(`${encodecBase}pkg/encodec_rs.js`)
        );
        await module.default({
          module_or_path: versionedUrl(`${encodecBase}pkg/encodec_rs_bg.wasm`),
        });
        module.initPanicHook?.();
        if (typeof module.lmEcdcChunkFromFrame !== "function") {
          throw new Error(
            "encodec-rs WASM is missing lmEcdcChunkFromFrame. Rebuild the fixed bundles.",
          );
        }
        return module;
      })();
    }
    return encodecWasmModulePromise;
  }

  async function releaseCachedEncoder() {
    const encoderPromise = cachedEncoderPromise;
    cachedEncoderKey = "";
    cachedEncoderPromise = null;
    if (!encoderPromise) return;
    const result = await Promise.allSettled([encoderPromise]);
    if (result[0].status === "fulfilled") result[0].value.release();
  }

  async function createEncoder(bundleRoot, bundleMeta, bundleConfig) {
    const config = encoderAssetConfig(bundleConfig);
    const root = new URL(config.assetRoot, normalizeBaseUrl(bundleRoot));
    const primary = new URL(config.kernel, root);
    const fallback = new URL(config.fallbackKernel, root);
    const options = {
      assetRoot: root,
      bundleMetadata: bundleMeta,
      fetchImpl: versionedFetch,
      versionedAssetUrl: versionedUrl,
    };
    try {
      return await createCustomEncoder({
        ...options,
        kernelModuleUrl: primary.href,
      });
    } catch (primaryError) {
      if (primary.href === fallback.href) throw primaryError;
      globalScope.console?.warn?.(
        "Relaxed SIMD is unavailable. The encoder will use standard SIMD.",
      );
      try {
        return await createCustomEncoder({
          ...options,
          kernelModuleUrl: fallback.href,
        });
      } catch (fallbackError) {
        throw new AggregateError(
          [primaryError, fallbackError],
          "The custom EnCodec encoder could not start.",
        );
      }
    }
  }

  async function getEncoder(bundleRoot, bundleMeta, bundleConfig) {
    const config = encoderAssetConfig(bundleConfig);
    const key = JSON.stringify([bundleRoot, config]);
    if (cachedEncoderKey !== key || !cachedEncoderPromise) {
      await releaseCachedEncoder();
      cachedEncoderKey = key;
      cachedEncoderPromise = createEncoder(
        bundleRoot,
        bundleMeta,
        bundleConfig,
      ).catch((error) => {
        if (cachedEncoderKey === key) {
          cachedEncoderKey = "";
          cachedEncoderPromise = null;
        }
        throw error;
      });
    }
    return cachedEncoderPromise;
  }

  function getJobState(sessionKey) {
    const key = String(sessionKey || "default");
    let state = encodeJobState.get(key);
    if (!state) {
      state = {
        bundleJson: "",
        bundleMeta: null,
        bundleConfig: null,
        bundleRoot: "",
        lmWeights: null,
      };
      encodeJobState.set(key, state);
    }
    return state;
  }

  function releaseJobState(sessionKey) {
    return encodeJobState.delete(String(sessionKey || "default"));
  }

  function updateJobState(state, data, encodecModule) {
    if (data.bundleRoot) {
      state.bundleRoot = data.bundleRoot.replace(/\/?$/, "");
    }
    if (data.bundleJson) {
      state.bundleJson = data.bundleJson;
      state.bundleConfig = JSON.parse(data.bundleJson);
      state.bundleMeta = encodecModule.bundleMetadata(data.bundleJson);
    }
    if (data.lmWeights) {
      state.lmWeights = normalizeBytes(data.lmWeights, "lmWeights");
    }
  }

  async function encodeEcdcChunkNow(payload, { requestId = null } = {}) {
    const data = normalizeEncodeMessage(payload);
    const encodecModule = await ensureEncodecWasmModule();
    const state = getJobState(data.sessionKey);
    updateJobState(state, data, encodecModule);
    if (!state.bundleJson || !state.bundleMeta || !state.bundleConfig) {
      throw new Error("bundleJson was not provided for this encode job.");
    }
    if (!state.lmWeights?.byteLength) {
      throw new Error("lmWeights were not provided for this encode job.");
    }
    if (!state.bundleRoot) {
      throw new Error("bundleRoot was not provided for this encode job.");
    }
    if (data.segmentFrameLength <= 0) return new Uint8Array();

    throwIfCancelled(requestId);
    const meta = state.bundleMeta;
    const channels = positiveInteger(meta.channels, "bundle channels");
    const modelSegmentSamples = positiveInteger(
      meta.segment_samples,
      "bundle segment_samples",
    );
    const frameLength = positiveInteger(meta.frame_length, "bundle frame_length");
    const numCodebooks = positiveInteger(
      meta.num_codebooks,
      "bundle num_codebooks",
    );
    const segmentAudio = normalizeFloat32(data.segment, "segment");
    const expectedAudioLength = channels * modelSegmentSamples;
    if (segmentAudio.length !== expectedAudioLength) {
      throw new Error(
        `segment contains ${segmentAudio.length} samples, expected ${expectedAudioLength}.`,
      );
    }

    const encoder = await getEncoder(
      state.bundleRoot,
      meta,
      state.bundleConfig,
    );
    throwIfCancelled(requestId);
    const encoded = encoder.encode(segmentAudio);
    throwIfCancelled(requestId);
    if (encoded.codes.length !== numCodebooks * frameLength) {
      throw new Error(
        `EnCodec returned ${encoded.codes.length} codes, expected ${numCodebooks * frameLength}.`,
      );
    }
    const scale = Number.isFinite(encoded.scale) ? encoded.scale : 1;
    const chunk = encodecModule.lmEcdcChunkFromFrame(
      state.bundleJson,
      state.lmWeights,
      scale,
      toU16Codes(encoded.codes, data.segmentIndex),
      data.segmentFrameLength,
    );
    return chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
  }

  function encodeEcdcChunk(payload, options = {}) {
    const operation = encodeInferenceTail.then(() =>
      encodeEcdcChunkNow(payload, options)
    );
    encodeInferenceTail = operation.catch(() => {});
    return operation;
  }

  return Object.freeze({
    async clearSessionCache() {
      await releaseCachedEncoder();
    },
    diagnostics() {
      return {
        encodeJobCount: encodeJobState.size,
        encodeSessionCount: cachedEncoderPromise ? 1 : 0,
      };
    },
    encodeEcdcChunk,
    releaseJobState,
  });
}
