#!/usr/bin/env node

import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "../browser-smoke/node_modules/playwright-core/index.mjs";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const serverPort = Number(process.env.PORT || 8797);
const serverOrigin = `http://127.0.0.1:${serverPort}`;
const outputDir = path.join(repoRoot, "target/pray4me-wasm-roundtrip");
const sourceWav = "/Users/jamie/Downloads/PRAY4ME - JB MSTR 2.wav";
const servedWav = path.join(outputDir, "PRAY4ME - JB MSTR 2.wav");
const bundleName = "encodec_48khz_12kbps_1333ms";
const outputTag = process.env.OUTPUT_TAG ? `.${process.env.OUTPUT_TAG}` : "";
const ecdcName = `pray4me-jb-mstr-2.encodec-rs-wasm.12kbps.1333ms${outputTag}.ecdc`;
const wavName = `PRAY4ME - JB MSTR 2.encodec-rs-wasm.12kbps.1333ms${outputTag}.roundtrip.wav`;
const ecdcOutput = path.join(outputDir, ecdcName);
const wavOutput = path.join(outputDir, wavName);
const downloadsWav = path.join("/Users/jamie/Downloads", wavName);
const downloadsEcdc = path.join("/Users/jamie/Downloads", ecdcName);
const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const reuseEcdc = process.env.REUSE_ECDC === "1";

mkdirSync(outputDir, { recursive: true });
copyFileSync(sourceWav, servedWav);

const server = spawn("python3", ["browser-smoke/serve.py"], {
  cwd: repoRoot,
  env: { ...process.env, PORT: String(serverPort) },
  stdio: ["ignore", "pipe", "pipe"],
});
server.stdout.on("data", (chunk) => process.stderr.write(chunk));
server.stderr.on("data", (chunk) => process.stderr.write(chunk));

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    executablePath: chromePath,
    headless: true,
  });
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  await page.addInitScript(() => {
    globalThis.WEBGPU_MATRIX_EXECUTION_PROVIDERS = ["wasm"];
  });
  page.on("console", (message) => process.stderr.write(`[browser:${message.type()}] ${message.text()}\n`));
  page.on("pageerror", (error) => process.stderr.write(`[browser:pageerror] ${error.stack || error.message || error}\n`));

  await page.goto(`${serverOrigin}/browser-smoke/webgpu-matrix.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.webgpuMatrix, { timeout: 0 });
  const ready = await page.evaluate(() => window.webgpuMatrix.ready());
  console.log(JSON.stringify({ event: "ready", ...ready }, null, 2));

  const encodeSummary = reuseEcdc && existsSync(ecdcOutput)
    ? { reused: true, outputPath: ecdcOutput }
    : await runDownload(page, "encode", {
      bundleName,
      inputWavUrl: repoUrl(path.relative(repoRoot, servedWav)),
      downloadName: ecdcName,
    }, ecdcOutput);
  copyFileSync(ecdcOutput, downloadsEcdc);

  const decodeSummary = await runDownload(page, "decode", {
    bundleName,
    inputEcdcUrl: repoUrl(path.relative(repoRoot, ecdcOutput)),
    downloadName: wavName,
  }, wavOutput);
  copyFileSync(wavOutput, downloadsWav);

  console.log(JSON.stringify({
    event: "complete",
    bundleName,
    inputWav: sourceWav,
    ecdcOutput,
    wavOutput,
    downloadsWav,
    downloadsEcdc,
    encode: encodeSummary,
    decode: decodeSummary,
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
  return { ...summary, outputPath };
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
  return `${serverOrigin}/${relativePath.split(path.sep).map(encodeURIComponent).join("/")}`;
}
