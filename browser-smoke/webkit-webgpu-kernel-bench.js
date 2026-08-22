const result = document.querySelector("#result");
const workerUrl = new URL("./webkit-webgpu-kernel-bench-worker.js", import.meta.url);
workerUrl.search = globalThis.location.search;

const ready = new Promise((resolve) => {
  const worker = new Worker(workerUrl, { type: "module" });
  worker.addEventListener("message", (event) => {
    worker.terminate();
    resolve(event.data);
  }, { once: true });
  worker.addEventListener("error", (event) => {
    worker.terminate();
    resolve({ ok: false, error: event.message || "worker failed" });
  }, { once: true });
});

globalThis.webkitWebGpuKernelBench = { ready };
ready.then((value) => {
  result.textContent = JSON.stringify(value, null, 2);
});
