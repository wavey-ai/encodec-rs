import { probeWebGpu } from "./webkit-webgpu-probe-core.js";

try {
  postMessage({ ok: true, result: await probeWebGpu("dedicated-worker") });
} catch (error) {
  postMessage({
    ok: false,
    error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
  });
}
