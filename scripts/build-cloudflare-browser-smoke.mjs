import { copyFile, mkdir, rm, stat, writeFile } from "node:fs/promises";
import { createReadStream, createWriteStream } from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";

const root = process.cwd();
const out = path.join(root, "build", "cloudflare-pages");
const siteRoot = path.join(out, "code", "encodec-rs");
const maxPartBytes = 20 * 1024 * 1024;

const browserFiles = ["index.html", "style.css"];
const pkgFiles = ["encodec_rs.js", "encodec_rs_bg.wasm"];
const ortFiles = [
  "ort.webgpu.min.mjs",
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];
const bundles = ["encodec_48khz_6kbps", "encodec_48khz_12kbps"];
const bundleFiles = ["bundle.json", "encode_frame.onnx", "decode_frame.onnx", "lm_logits.onnx", "lm_weights.bin"];

await rm(out, { recursive: true, force: true });
await mkdir(path.join(siteRoot, "browser-smoke"), { recursive: true });
await mkdir(path.join(siteRoot, "pkg"), { recursive: true });
await mkdir(path.join(siteRoot, "mel-spec"), { recursive: true });

for (const file of browserFiles) {
  await copyFile(path.join(root, "browser-smoke", file), path.join(siteRoot, "browser-smoke", file));
}
await writeFile(
  path.join(siteRoot, "browser-smoke", "app.js"),
  (await readText(path.join(root, "browser-smoke", "app.js")))
    .replace(
      "./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs",
      "./vendor/onnxruntime-web/ort.webgpu.min.mjs",
    )
    .replace("./node_modules/onnxruntime-web/dist/", "./vendor/onnxruntime-web/"),
);

for (const file of pkgFiles) {
  await copyFile(path.join(root, "pkg", file), path.join(siteRoot, "pkg", file));
}

const ortOut = path.join(
  siteRoot,
  "browser-smoke",
  "vendor",
  "onnxruntime-web",
);
await mkdir(ortOut, { recursive: true });
for (const file of ortFiles) {
  await copyOrShard(
    path.join(root, "browser-smoke", "node_modules", "onnxruntime-web", "dist", file),
    path.join(ortOut, file),
  );
}

for (const bundle of bundles) {
  const bundleOut = path.join(siteRoot, "onnx-bundles", bundle);
  await mkdir(bundleOut, { recursive: true });
  for (const file of bundleFiles) {
    await copyOrShard(path.join(root, "onnx-bundles", bundle, file), path.join(bundleOut, file));
  }
}

await copyFile(
  path.join(root, "..", "mel-spec", "testdata", "jfk_f32le.wav"),
  path.join(siteRoot, "mel-spec", "jfk_f32le.wav"),
);

await writeFile(
  path.join(out, "_headers"),
  [
    "/code/encodec-rs/*",
    "  Cross-Origin-Opener-Policy: same-origin",
    "  Cross-Origin-Embedder-Policy: require-corp",
    "  Cross-Origin-Resource-Policy: same-origin",
    "  Cache-Control: public, max-age=3600",
    "",
    "/code/encodec-rs/pkg/*",
    "  Cache-Control: no-store",
    "",
    "/code/encodec-rs/browser-smoke/",
    "  Cache-Control: no-store",
    "",
    "/code/encodec-rs/browser-smoke/index.html",
    "  Cache-Control: no-store",
    "",
  ].join("\n"),
);

async function copyOrShard(source, target) {
  const info = await stat(source);
  if (info.size <= maxPartBytes) {
    await copyFile(source, target);
    return;
  }

  const parts = [];
  let offset = 0;
  let index = 0;
  while (offset < info.size) {
    const partName = `${path.basename(target)}.part${String(index).padStart(3, "0")}`;
    const partPath = path.join(path.dirname(target), partName);
    await copyRange(source, partPath, offset, Math.min(offset + maxPartBytes, info.size) - 1);
    parts.push(partName);
    offset += maxPartBytes;
    index += 1;
  }

  await writeFile(
    `${target}.parts.json`,
    `${JSON.stringify({ version: 1, byteLength: info.size, partBytes: maxPartBytes, parts }, null, 2)}\n`,
  );
}

async function copyRange(source, target, start, end) {
  await pipeline(createReadStream(source, { start, end }), createWriteStream(target));
}

async function readText(source) {
  const chunks = [];
  for await (const chunk of createReadStream(source, { encoding: "utf8" })) {
    chunks.push(chunk);
  }
  return chunks.join("");
}
