#!/usr/bin/env node

import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const profileMs = Number(process.argv[2] ?? 1333);
const packagePath = path.join(repoRoot, "target/paired-wasm-test/pkg/encodec_rs.js");
const require = createRequire(import.meta.url);
const wasm = require(packagePath);

function loadBundle(kbps) {
  const root = path.join(
    repoRoot,
    "onnx-bundles",
    `encodec_48khz_${kbps}kbps_${profileMs}ms`,
  );
  const bundleJson = readFileSync(path.join(root, "bundle.json"), "utf8");
  return {
    json: bundleJson,
    metadata: JSON.parse(bundleJson),
    weights: readFileSync(path.join(root, "lm_weights_q8.bin")),
  };
}

const primary = loadBundle(12);
const derived = loadBundle(6);
const frameLength = primary.metadata.frame_length;
assert.equal(frameLength, derived.metadata.frame_length);
const scale = Math.fround(0.17320508);
const steps = Array.from({ length: frameLength }, (_, step) =>
  Uint16Array.from({ length: primary.metadata.num_codebooks }, (_, codebook) =>
    (step * 17 + codebook * 31 + 7) % primary.metadata.lm_cardinality,
  ),
);

function runSeparate() {
  const primaryEncoder = new wasm.QuantizedLmChunkEncoder(
    primary.json,
    primary.weights,
    scale,
  );
  const derivedEncoder = new wasm.QuantizedLmChunkEncoder(
    derived.json,
    derived.weights,
    scale,
  );
  const started = performance.now();
  for (const codes of steps) primaryEncoder.push(codes);
  for (const codes of steps) {
    derivedEncoder.push(codes.subarray(0, derived.metadata.num_codebooks));
  }
  const primaryPayload = primaryEncoder.finish();
  const derivedPayload = derivedEncoder.finish();
  return { elapsedMs: performance.now() - started, primaryPayload, derivedPayload };
}

function runPaired() {
  const encoder = new wasm.QuantizedLmPairedChunkEncoder(
    primary.json,
    derived.json,
    primary.weights,
    derived.weights,
    scale,
  );
  const started = performance.now();
  for (const codes of steps) encoder.push(codes);
  const payloads = encoder.finish();
  return {
    elapsedMs: performance.now() - started,
    primaryPayload: Uint8Array.from(payloads.primary),
    derivedPayload: Uint8Array.from(payloads.derived),
  };
}

runSeparate();
runPaired();
const rounds = [];
for (let round = 0; round < 5; round += 1) {
  const first = round % 2 === 0 ? runSeparate() : runPaired();
  const second = round % 2 === 0 ? runPaired() : runSeparate();
  const separate = round % 2 === 0 ? first : second;
  const paired = round % 2 === 0 ? second : first;
  assert.deepEqual(Buffer.from(paired.primaryPayload), Buffer.from(separate.primaryPayload));
  assert.deepEqual(Buffer.from(paired.derivedPayload), Buffer.from(separate.derivedPayload));
  rounds.push({
    round,
    separateMs: separate.elapsedMs,
    pairedMs: paired.elapsedMs,
    speedup: separate.elapsedMs / paired.elapsedMs,
  });
}

const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const separateMedianMs = median(rounds.map(({ separateMs }) => separateMs));
const pairedMedianMs = median(rounds.map(({ pairedMs }) => pairedMs));
process.stdout.write(`${JSON.stringify({
  profileMs,
  frameLength,
  exactPrimary: true,
  exactDerived: true,
  separateMedianMs,
  pairedMedianMs,
  speedup: separateMedianMs / pairedMedianMs,
  rounds,
}, null, 2)}\n`);
