#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

if (!process.argv[2] || !process.argv[3] || !process.argv[4] || !process.argv[5]) {
  throw new Error(
    "usage: extract-ecdc-decoder-inputs.mjs <input.ecdc> <bundle> <wasm-root> <output-dir>",
  );
}

const inputEcdc = path.resolve(process.argv[2]);
const bundleDir = path.resolve(process.argv[3]);
const wasmRoot = path.resolve(process.argv[4]);
const outputDir = path.resolve(process.argv[5]);

const bundleJson = fs.readFileSync(path.join(bundleDir, "bundle.json"), "utf8");
const bundle = JSON.parse(bundleJson);
const ecdc = fs.readFileSync(inputEcdc);
const encodec = await import(
  pathToFileURL(path.join(wasmRoot, "encodec-rs/pkg/encodec_rs.js"))
);
encodec.initSync({
  module: fs.readFileSync(path.join(wasmRoot, "encodec-rs/pkg/encodec_rs_bg.wasm")),
});

const metadata = encodec.ecdcMetadata(ecdc);
const parsed = encodec.lmEcdcDecodeChunks(bundleJson, ecdc);
const weights = new Uint8Array(
  fs.readFileSync(path.join(bundleDir, bundle.lm_quant_weight_model)),
);
fs.mkdirSync(outputDir, { recursive: true });

const started = performance.now();
const frames = [];
for (let frameIndex = 0; frameIndex < parsed.chunks.length; ++frameIndex) {
  const chunk = parsed.chunks[frameIndex];
  const decoder = new encodec.QuantizedLmChunkDecoder(
    bundleJson,
    weights,
    Uint8Array.from(chunk.payload),
  );
  try {
    const symbols = decoder.pullAll(chunk.frameLength);
    const codes = new Uint16Array(bundle.num_codebooks * bundle.frame_length);
    for (let codebook = 0; codebook < bundle.num_codebooks; ++codebook) {
      const sourceBase = codebook * chunk.frameLength;
      const targetBase = codebook * bundle.frame_length;
      codes.set(symbols.subarray(sourceBase, sourceBase + chunk.frameLength), targetBase);
    }
    const file = `frame-${frameIndex}.u16le`;
    fs.writeFileSync(
      path.join(outputDir, file),
      Buffer.from(codes.buffer, codes.byteOffset, codes.byteLength),
    );
    frames.push({
      file,
      offset: chunk.offset,
      samples: chunk.samples,
      frameLength: chunk.frameLength,
      scale: decoder.scale(),
    });
  } finally {
    decoder.free();
  }
}

const report = {
  inputEcdc,
  audioLength: metadata.al ?? metadata.audio_length,
  sampleRate: bundle.sample_rate,
  channels: bundle.channels,
  frameLength: bundle.frame_length,
  numCodebooks: bundle.num_codebooks,
  entropyDecodeMs: performance.now() - started,
  frames,
};
fs.writeFileSync(path.join(outputDir, "metadata.json"), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
