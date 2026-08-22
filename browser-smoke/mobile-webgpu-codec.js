import "./webgpu-matrix.js";

const params = new URL(location.href).searchParams;
const runButton = document.querySelector("#run");
const heading = document.querySelector("#heading");
const description = document.querySelector("#description");
const phase = document.querySelector("#phase");
const progress = document.querySelector("#progress");
const result = document.querySelector("#result");
const decodeOnly = params.get("mode") === "decode";

if (decodeOnly) {
  heading.textContent = "EnCodec WebGPU incremental decode";
  description.textContent = "This test decodes and delivers one playable ECDC chunk at a time.";
  runButton.textContent = "Run incremental decode";
}

window.addEventListener("encodec-webgpu-progress", ({ detail }) => {
  phase.textContent = `${detail.phase}: ${detail.completed}/${detail.total}`;
  progress.max = Math.max(1, detail.total);
  progress.value = detail.completed;
});

runButton.addEventListener("click", run);
globalThis.mobileWebGpuCodec = { run };

if (params.get("autorun") === "1") {
  run();
}

async function run() {
  runButton.disabled = true;
  result.textContent = "";
  phase.textContent = "Starting";
  progress.value = 0;
  try {
    const bundleName = params.get("bundle") || "encodec_48khz_12kbps_1333ms";
    const summary = decodeOnly
      ? await window.webgpuMatrix.webGpuDecodeEcdc({
        bundleName,
        ecdcUrl: params.get("ecdc") || new URL(
          "../target/performance/mlx/borrowed-logits-candidate/" +
            "westside_4s_48khz_stereo.encodec_48khz_12kbps_1333ms.lm.mlx.ecdc",
          location.href,
        ).href,
        referenceWavUrl: params.get("reference") || null,
        collectAudio: params.get("collect") === "1" || Boolean(params.get("reference")),
      })
      : await window.webgpuMatrix.webGpuRoundTrip({
        bundleName,
        inputWavUrl: params.get("input") ||
          new URL("../testdata/westside_4s_48khz_stereo.wav", location.href).href,
        expectedEcdcUrl: params.get("expected") || null,
        decodeEcdcUrl: params.get("decode") || null,
        referenceWavUrl: params.get("reference") || null,
        downloadPrefix: params.get("download") || null,
      });
    result.textContent = JSON.stringify(summary, null, 2);
    document.title = "PASS — EnCodec WebGPU";
    return summary;
  } catch (error) {
    phase.textContent = "Failed";
    result.textContent = `${error}\n${error?.stack || ""}`;
    document.title = "FAIL — EnCodec WebGPU";
    throw error;
  } finally {
    runButton.disabled = false;
  }
}
