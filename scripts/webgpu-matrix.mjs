#!/usr/bin/env node

import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "../browser-smoke/node_modules/playwright-core/index.mjs";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const serverOrigin = "http://127.0.0.1:8787";
const outputDir = path.join(repoRoot, "target/webgpu-matrix");
const westsideWav = "target/lori-asha-wasm-native/wav/02 - Lori Asha - Westside.48k-stereo.wav";
const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

mkdirSync(outputDir, { recursive: true });

const server = spawn("python3", ["browser-smoke/serve.py"], {
  cwd: repoRoot,
  stdio: ["ignore", "pipe", "pipe"],
});
server.stdout.on("data", (chunk) => process.stderr.write(chunk));
server.stderr.on("data", (chunk) => process.stderr.write(chunk));

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath: chromePath,
    headless: false,
    args: [
      "--enable-unsafe-webgpu",
      "--disable-dawn-features=disallow_unsafe_apis",
    ],
  });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  page.on("console", (message) => process.stderr.write(`[browser:${message.type()}] ${message.text()}\n`));
  await page.goto(`${serverOrigin}/browser-smoke/webgpu-matrix.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.webgpuMatrix);
  const ready = await page.evaluate(() => window.webgpuMatrix.ready());
  console.log(JSON.stringify({ event: "ready", ...ready }, null, 2));

  const encodeSummaries = [];
  for (const bundleName of ["encodec_48khz_6kbps", "encodec_48khz_12kbps"]) {
    const outputName = `westside.encode-browser-webgpu-macos-arm64.${bundleName}.q8.ecdc`;
    encodeSummaries.push(await runDownload(page, "encode", {
      bundleName,
      inputWavUrl: repoUrl(westsideWav),
      downloadName: outputName,
    }, path.join(outputDir, outputName)));
  }

  const decodeSummaries = [];
  for (const bundleName of ["encodec_48khz_6kbps", "encodec_48khz_12kbps"]) {
    const inputName = `westside.encode-native-rust-linux-x86_64-cpu.${bundleName}.q8.ecdc`;
    const outputName = `westside.encode-native-rust-linux-x86_64-cpu.decode-browser-webgpu-macos-arm64.${bundleName}.q8.wav`;
    decodeSummaries.push(await runDownload(page, "decode", {
      bundleName,
      inputEcdcUrl: repoUrl(`target/gpu-matrix/${inputName}`),
      downloadName: outputName,
    }, path.join(outputDir, outputName)));
  }

  console.log(JSON.stringify({
    event: "complete",
    outputDir: path.relative(repoRoot, outputDir),
    encodes: encodeSummaries,
    decodes: decodeSummaries,
  }, null, 2));
} finally {
  if (browser) {
    await browser.close();
  }
  server.kill();
}

async function runDownload(page, method, options, outputPath) {
  const [download, summary] = await Promise.all([
    page.waitForEvent("download", { timeout: 0 }),
    page.evaluate(([name, opts]) => window.webgpuMatrix[name](opts), [method, options]),
  ]);
  await download.saveAs(outputPath);
  return { ...summary, outputPath: path.relative(repoRoot, outputPath) };
}

async function waitForServer() {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${serverOrigin}/browser-smoke/webgpu-matrix.html`);
      if (response.ok) {
        return;
      }
    } catch {
      // retry
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("browser-smoke server did not start");
}

function repoUrl(relativePath) {
  return `${serverOrigin}/${relativePath.split("/").map(encodeURIComponent).join("/")}`;
}
