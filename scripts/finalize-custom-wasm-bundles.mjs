#!/usr/bin/env node

import { createHash } from "node:crypto";
import {
  existsSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

if (process.argv.length < 4) {
  process.stderr.write(
    "usage: node scripts/finalize-custom-wasm-bundles.mjs <output-dir> <bundle>...\n",
  );
  process.exit(2);
}

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const outputRoot = path.resolve(process.argv[2]);
const bundleNames = process.argv.slice(3);
const summaries = [];

for (const name of bundleNames) {
  const sourceRoot = path.join(repoRoot, "onnx-bundles", name);
  const bundleRoot = path.join(outputRoot, "bundles", name);
  const sourceBundle = JSON.parse(
    readFileSync(path.join(sourceRoot, "bundle.json"), "utf8"),
  );
  const customRuntime = {
    schema_version: 1,
    kind: "custom_wasm_simd",
    encoder: {
      asset_root: "encoder/",
      metadata: "metadata.json",
      weights_manifest: "weights.json",
      weights: "weights.f32le",
      kernel: "encodec-encoder-relaxed.mjs",
      fallback_kernel: "encodec-encoder.mjs",
    },
    decoder: {
      asset_root: "decoder/",
      metadata: "metadata.json",
      weights_manifest: "weights.json",
      weights: "weights.f32le",
      kernel: "encodec-convtranspose-relaxed.mjs",
      fallback_kernel: "encodec-convtranspose.mjs",
    },
  };
  const requiredAssets = [
    "lm_weights_q8.bin",
    ...runtimeAssets(customRuntime.encoder).map((asset) => `encoder/${asset}`),
    ...runtimeAssets(customRuntime.decoder).map((asset) => `decoder/${asset}`),
  ];
  for (const asset of requiredAssets) {
    if (!existsSync(path.join(bundleRoot, asset))) {
      throw new Error(`${name} is missing ${asset}`);
    }
  }

  const bundle = {
    ...sourceBundle,
    encode_model: "encoder/metadata.json",
    decode_model: "decoder/metadata.json",
    custom_wasm_runtime: customRuntime,
    bitneedle_player_assets: Object.fromEntries(
      requiredAssets.map((asset) => [asset, { split: false }]),
    ),
  };
  writeJson(path.join(bundleRoot, "bundle.json"), bundle);

  const assetHashes = Object.fromEntries(
    ["bundle.json", ...requiredAssets].map((asset) => {
      const bytes = readFileSync(path.join(bundleRoot, asset));
      return [
        asset,
        {
          bytes: bytes.byteLength,
          sha256: createHash("sha256").update(bytes).digest("hex"),
        },
      ];
    }),
  );
  const summary = {
    name,
    bundleJson: "bundle.json",
    lmWeights: "lm_weights_q8.bin",
    runtime: customRuntime,
    modelName: bundle.model_name,
    bandwidthKbps: bundle.bandwidth_kbps,
    sampleRate: bundle.sample_rate,
    channels: bundle.channels,
    segmentSamples: bundle.segment_samples,
    segmentStride: bundle.segment_stride,
    frameLength: bundle.frame_length,
    numCodebooks: bundle.num_codebooks,
  };
  writeJson(path.join(bundleRoot, "manifest.json"), {
    schemaVersion: 2,
    ...summary,
    assets: assetHashes,
  });
  summaries.push(summary);
}

writeJson(path.join(outputRoot, "manifest.json"), {
  schemaVersion: 2,
  pkg: "pkg",
  runtimeModule: "encodec-ecdc-runtime.js",
  encoderRuntimeModule: "custom-encoder-runtime.js",
  decoderRuntimeModule: "custom-decoder-runtime.js",
  neuralRuntime: "custom_wasm_simd",
  bundles: summaries,
});

console.log(
  JSON.stringify({ outputRoot, bundles: summaries.map(({ name }) => name) }, null, 2),
);

function runtimeAssets(runtime) {
  const primaryWasm = runtime.kernel.replace(/\.mjs$/, ".wasm");
  const fallbackWasm = runtime.fallback_kernel.replace(/\.mjs$/, ".wasm");
  return [
    runtime.metadata,
    runtime.weights_manifest,
    runtime.weights,
    runtime.kernel,
    primaryWasm,
    runtime.fallback_kernel,
    fallbackWasm,
  ];
}

function writeJson(file, value) {
  writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}
