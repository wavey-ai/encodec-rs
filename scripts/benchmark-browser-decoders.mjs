#!/usr/bin/env node

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "../browser-smoke/node_modules/playwright-core/index.mjs";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const origin = process.env.BROWSER_BENCH_ORIGIN ?? "http://127.0.0.1:8798";
const chromePath =
  process.env.CHROME_BIN ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const reportPath = path.resolve(
  process.env.BROWSER_BENCH_REPORT ??
    path.join(
      repoRoot,
      "target/performance/custom-kernel/decoder-browser-clean/browser-comparison.json",
    ),
);
const iterations = Number(process.env.BROWSER_BENCH_ITERATIONS ?? 5);
const warmupFrameCount = Number(process.env.BROWSER_BENCH_WARMUP_FRAMES ?? 3);
const inputEcdcPath =
  process.env.BROWSER_BENCH_INPUT ??
  "target/performance/custom-encoder/conv-kernel-v1/westside-onnx-control.ecdc";
const mode = process.env.BROWSER_BENCH_MODE ?? "compare";
const customEncoderRoot =
  process.env.BROWSER_CUSTOM_ENCODER_ROOT ??
  "target/performance/custom-encoder/browser-clean/";
const customEncoderKernel =
  process.env.BROWSER_CUSTOM_ENCODER_KERNEL ??
  "target/performance/custom-encoder/browser-clean/encodec-encoder-relaxed.mjs";
const customDecoderRoot =
  process.env.BROWSER_CUSTOM_DECODER_ROOT ??
  "target/performance/custom-kernel/decoder-browser-clean/";
const customDecoderKernel =
  process.env.BROWSER_CUSTOM_DECODER_KERNEL ??
  "target/performance/custom-kernel/decoder-browser-clean/encodec-convtranspose-relaxed.mjs";

function browserUrl(relativePath) {
  return `${origin}/${relativePath.replace(/^\/+/, "")}`;
}

const browser = await chromium.launch({
  executablePath: chromePath,
  headless: true,
});

try {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("console", (message) => {
    process.stderr.write(`[browser:${message.type()}] ${message.text()}\n`);
  });
  page.on("pageerror", (error) => {
    process.stderr.write(`[browser:pageerror] ${error.stack ?? error.message}\n`);
  });
  page.on("requestfailed", (request) => {
    process.stderr.write(
      `[browser:requestfailed] ${request.url()} ${request.failure()?.errorText ?? ""}\n`,
    );
  });

  await page.goto(`${origin}/browser-smoke/webgpu-matrix.html?provider=wasm`, {
    waitUntil: "domcontentloaded",
  });
  await page.waitForFunction(() => window.webgpuMatrix, { timeout: 0 });
  const ready = await page.evaluate((selectedMode) =>
    selectedMode === "roundtrip" || selectedMode === "runtime-chunk"
      ? window.webgpuMatrix.readyCustom()
      : window.webgpuMatrix.ready(), mode);
  const result = mode === "runtime-chunk"
    ? await page.evaluate(
      async (options) => window.webgpuMatrix.customRuntimeChunk(options),
      {
        bundleRootUrl:
          `${origin}/dist/wasm-fixed-bundles/bundles/` +
          "encodec_48khz_12kbps_1333ms/",
        runtimeModuleUrl:
          `${origin}/dist/wasm-fixed-bundles/encodec-ecdc-runtime.js`,
        runtimeRootUrl: `${origin}/dist/wasm-fixed-bundles/`,
        inputWavUrl: `${origin}/testdata/westside_4s_48khz_stereo.wav`,
        expectedEcdcUrl:
          `${origin}/target/performance/custom-encoder/conv-kernel-v1/` +
          "westside-onnx-control.ecdc",
      },
    )
    : mode === "roundtrip"
    ? await page.evaluate(
      async (options) => window.webgpuMatrix.customRoundTrip(options),
      {
        bundleName: "encodec_48khz_12kbps_1333ms",
        inputWavUrl: `${origin}/testdata/westside_4s_48khz_stereo.wav`,
        expectedEcdcUrl:
          `${origin}/target/performance/custom-encoder/conv-kernel-v1/` +
          "westside-onnx-control.ecdc",
        referenceWavUrl:
          `${origin}/target/performance/custom-kernel/decoder-full-v2-fused/` +
          "westside-full-v2.f32.wav",
        customEncoderRootUrl: browserUrl(customEncoderRoot),
        customEncoderKernelUrl: browserUrl(customEncoderKernel),
        customDecoderRootUrl: browserUrl(customDecoderRoot),
        customDecoderKernelUrl: browserUrl(customDecoderKernel),
      },
    )
    : await page.evaluate(
      async (options) => window.webgpuMatrix.compareDecoders(options),
      {
        bundleName: "encodec_48khz_12kbps_1333ms",
        inputEcdcUrl: `${origin}/${inputEcdcPath}`,
        customDecoderRootUrl: browserUrl(customDecoderRoot),
        customKernelModuleUrl: browserUrl(customDecoderKernel),
        iterations,
        warmupFrameCount,
      },
    );
  const report = {
    generatedAt: new Date().toISOString(),
    mode,
    ready,
    result,
  };
  mkdirSync(path.dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
