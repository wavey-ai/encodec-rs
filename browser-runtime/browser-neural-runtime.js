"use strict";

const BACKEND_AUTO = "auto";
const BACKEND_WEBGPU = "webgpu";
const BACKEND_WASM_SIMD = "wasm-simd";
const VALID_BACKENDS = new Set([
  BACKEND_AUTO,
  BACKEND_WEBGPU,
  BACKEND_WASM_SIMD,
]);

function normalizeUrl(value) {
  const url = value instanceof URL
    ? value
    : new URL(String(value), globalThis.location?.href);
  return new URL(url.href.replace(/\/?$/, "/"));
}

function normalizeBackend(value) {
  const backend = String(value || BACKEND_AUTO).trim().toLowerCase();
  if (!VALID_BACKENDS.has(backend)) {
    throw new Error(
      `Unsupported EnCodec browser backend ${JSON.stringify(value)}. `
      + `Expected ${[...VALID_BACKENDS].join(", ")}.`,
    );
  }
  return backend;
}

function runtimeAssetConfig(bundleMetadata, kind) {
  const runtime = bundleMetadata?.custom_wasm_runtime;
  const model = runtime?.[kind];
  if (runtime?.kind !== "custom_wasm_simd" || !model) {
    throw new Error(`The EnCodec bundle has no ${kind} browser runtime.`);
  }
  const defaultKernel = kind === "encoder"
    ? "encodec-encoder-relaxed.mjs"
    : "encodec-convtranspose-relaxed.mjs";
  const defaultFallback = kind === "encoder"
    ? "encodec-encoder.mjs"
    : "encodec-convtranspose.mjs";
  return {
    assetRoot: String(model.asset_root || `${kind}/`),
    kernel: String(model.kernel || defaultKernel),
    fallbackKernel: String(model.fallback_kernel || defaultFallback),
  };
}

function webGpuGlobalsAvailable(scope) {
  return Boolean(
    scope?.navigator?.gpu
    && scope?.GPUBufferUsage
    && scope?.GPUMapMode,
  );
}

export function browserNeuralCapabilities(scope = globalThis) {
  return Object.freeze({
    webgpu: webGpuGlobalsAvailable(scope),
    wasm: typeof scope?.WebAssembly === "object",
  });
}

async function createWebGpuRuntime(kind, options) {
  if (!webGpuGlobalsAvailable(options.scope)) {
    throw new Error("WebGPU is unavailable in this browser worker.");
  }
  if (kind === "encoder") {
    const { createWebGpuEncoder } = await import("./webgpu-encoder-runtime.js");
    return createWebGpuEncoder(options);
  }
  const { createWebGpuDecoder } = await import("./webgpu-decoder-runtime.js");
  return createWebGpuDecoder(options);
}

async function createWasmRuntime(kind, {
  assetRoot,
  bundleMetadata,
  config,
  fetchImpl,
  versionedAssetUrl,
  logger,
}) {
  const primary = new URL(config.kernel, assetRoot);
  const fallback = new URL(config.fallbackKernel, assetRoot);
  const runtimeModule = kind === "encoder"
    ? await import("./custom-encoder-runtime.js")
    : await import("./custom-decoder-runtime.js");
  const create = kind === "encoder"
    ? runtimeModule.createCustomEncoder
    : runtimeModule.createCustomDecoder;
  const options = {
    assetRoot,
    bundleMetadata,
    fetchImpl,
    versionedAssetUrl,
  };
  try {
    return await create({ ...options, kernelModuleUrl: primary });
  } catch (primaryError) {
    if (primary.href === fallback.href) throw primaryError;
    logger?.warn?.(
      `Relaxed SIMD is unavailable. The EnCodec ${kind} will use standard SIMD.`,
    );
    try {
      return await create({ ...options, kernelModuleUrl: fallback });
    } catch (fallbackError) {
      throw new AggregateError(
        [primaryError, fallbackError],
        `The EnCodec ${kind} WASM runtime could not start.`,
      );
    }
  }
}

async function createBrowserRuntime(kind, {
  backend = BACKEND_AUTO,
  bundleRoot,
  bundleMetadata,
  bundleConfig = bundleMetadata,
  fetchImpl = globalThis.fetch.bind(globalThis),
  versionedAssetUrl = (asset) => asset,
  scope = globalThis,
  logger = globalThis.console,
} = {}) {
  const preference = normalizeBackend(backend);
  const config = runtimeAssetConfig(bundleConfig, kind);
  const root = normalizeUrl(bundleRoot);
  const assetRoot = new URL(config.assetRoot, root);
  const shared = {
    assetRoot,
    bundleMetadata,
    fetchImpl,
    scope,
  };

  if (preference !== BACKEND_WASM_SIMD && webGpuGlobalsAvailable(scope)) {
    try {
      const runtime = await createWebGpuRuntime(kind, shared);
      return Object.assign(runtime, {
        backend: BACKEND_WEBGPU,
        backendPreference: preference,
      });
    } catch (webGpuError) {
      if (preference === BACKEND_WEBGPU) throw webGpuError;
      logger?.warn?.(
        `WebGPU EnCodec ${kind} is unavailable. Falling back to SIMD WASM.`,
        webGpuError,
      );
    }
  } else if (preference === BACKEND_WEBGPU) {
    throw new Error("WebGPU is unavailable in this browser worker.");
  }

  const runtime = await createWasmRuntime(kind, {
    ...shared,
    config,
    versionedAssetUrl,
    logger,
  });
  return Object.assign(runtime, {
    backend: BACKEND_WASM_SIMD,
    backendPreference: preference,
  });
}

export function createBrowserEncoder(options) {
  return createBrowserRuntime("encoder", options);
}

export function createBrowserDecoder(options) {
  return createBrowserRuntime("decoder", options);
}
