import { probeWebGpu } from "./webkit-webgpu-probe-core.js";

const resultElement = document.querySelector("#result");

window.webkitWebGpuProbe = {
  ready: run(),
};

async function run() {
  const result = {
    window: await probeWebGpu("window"),
    worker: await probeWorker(),
  };
  resultElement.textContent = JSON.stringify(result, null, 2);
  return result;
}

function probeWorker() {
  return new Promise((resolve) => {
    const worker = new Worker("./webkit-webgpu-probe-worker.js", { type: "module" });
    worker.addEventListener("message", (event) => {
      resolve(event.data);
      worker.terminate();
    }, { once: true });
    worker.addEventListener("error", (event) => {
      resolve({ ok: false, error: event.message });
      worker.terminate();
    }, { once: true });
  });
}
