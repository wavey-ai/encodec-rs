#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { pathToFileURL } from "node:url";

if (process.argv.length !== 7) {
  throw new Error(
    "usage: verify-lm-wasm-parity.mjs <input.ecdc> <bundle> <reference-wasm-root> <candidate-wasm-root> <output.json>",
  );
}

const [, , inputArg, bundleArg, referenceArg, candidateArg, outputArg] = process.argv;
const inputPath = path.resolve(inputArg);
const bundleDir = path.resolve(bundleArg);
const referenceRoot = path.resolve(referenceArg);
const candidateRoot = path.resolve(candidateArg);
const outputPath = path.resolve(outputArg);
const bundleJson = fs.readFileSync(path.join(bundleDir, "bundle.json"), "utf8");
const bundle = JSON.parse(bundleJson);
const weights = new Uint8Array(
  fs.readFileSync(path.join(bundleDir, bundle.lm_quant_weight_model)),
);
const ecdc = new Uint8Array(fs.readFileSync(inputPath));

const reference = await loadRuntime(referenceRoot);
const candidate = await loadRuntime(candidateRoot);
const parsed = reference.lmEcdcDecodeChunks(bundleJson, ecdc);

const referenceCodes = [];
const candidateCodes = [];
const originalPayloads = [];
const referencePayloads = [];
const candidatePayloads = [];
const chunks = [];
let referenceDecodeMs = 0;
let candidateDecodeMs = 0;
let referenceEncodeMs = 0;
let candidateEncodeMs = 0;

for (const [index, chunk] of parsed.chunks.entries()) {
  const originalPayload = Uint8Array.from(chunk.payload);
  const referenceDecode = decodeChunk(
    reference,
    bundleJson,
    weights,
    originalPayload,
    chunk.frameLength,
  );
  referenceDecodeMs += referenceDecode.elapsedMs;
  const candidateDecode = decodeChunk(
    candidate,
    bundleJson,
    weights,
    originalPayload,
    chunk.frameLength,
  );
  candidateDecodeMs += candidateDecode.elapsedMs;
  assertBytesEqual(referenceDecode.codes, candidateDecode.codes, `chunk ${index} codes`);
  if (referenceDecode.scaleBits !== candidateDecode.scaleBits) {
    throw new Error(`chunk ${index} scale differs`);
  }

  const referenceEncode = encodeChunk(
    reference,
    bundleJson,
    weights,
    referenceDecode.scale,
    referenceDecode.codes,
    chunk.frameLength,
    bundle.num_codebooks,
  );
  referenceEncodeMs += referenceEncode.elapsedMs;
  const candidateEncode = encodeChunk(
    candidate,
    bundleJson,
    weights,
    referenceDecode.scale,
    referenceDecode.codes,
    chunk.frameLength,
    bundle.num_codebooks,
  );
  candidateEncodeMs += candidateEncode.elapsedMs;
  assertBytesEqual(originalPayload, referenceEncode.payload, `chunk ${index} reference payload`);
  assertBytesEqual(originalPayload, candidateEncode.payload, `chunk ${index} candidate payload`);

  referenceCodes.push(referenceDecode.codes);
  candidateCodes.push(candidateDecode.codes);
  originalPayloads.push(originalPayload);
  referencePayloads.push(referenceEncode.payload);
  candidatePayloads.push(candidateEncode.payload);
  chunks.push({
    index,
    frameLength: chunk.frameLength,
    payloadBytes: originalPayload.length,
    codesSha256: sha256(referenceDecode.codes),
    payloadSha256: sha256(originalPayload),
  });
  if ((index + 1) % 10 === 0 || index + 1 === parsed.chunks.length) {
    process.stderr.write(`verified ${index + 1}/${parsed.chunks.length} chunks\n`);
  }
}

const report = {
  schema: "encodec-rs.lm-wasm-parity",
  input: inputPath,
  bundle: bundleDir,
  referenceRoot,
  candidateRoot,
  chunkCount: chunks.length,
  exact: {
    decodedCodes: sha256Many(referenceCodes) === sha256Many(candidateCodes),
    referenceReencode: sha256Many(originalPayloads) === sha256Many(referencePayloads),
    candidateReencode: sha256Many(originalPayloads) === sha256Many(candidatePayloads),
    codesSha256: sha256Many(referenceCodes),
    payloadsSha256: sha256Many(originalPayloads),
  },
  timingMs: {
    referenceDecode: referenceDecodeMs,
    candidateDecode: candidateDecodeMs,
    referenceEncode: referenceEncodeMs,
    candidateEncode: candidateEncodeMs,
  },
  chunks,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));

async function loadRuntime(root) {
  const pkgDir = path.join(root, "encodec-rs", "pkg");
  const runtime = await import(pathToFileURL(path.join(pkgDir, "encodec_rs.js")));
  runtime.initSync({
    module: fs.readFileSync(path.join(pkgDir, "encodec_rs_bg.wasm")),
  });
  return runtime;
}

function decodeChunk(runtime, bundleText, lmWeights, payload, frameLength) {
  const decoder = new runtime.QuantizedLmChunkDecoder(bundleText, lmWeights, payload);
  const started = performance.now();
  try {
    const codes = decoder.pullAll(frameLength);
    const scale = decoder.scale();
    return {
      codes,
      scale,
      scaleBits: f32Bits(scale),
      elapsedMs: performance.now() - started,
    };
  } finally {
    decoder.free();
  }
}

function encodeChunk(
  runtime,
  bundleText,
  lmWeights,
  scale,
  codes,
  frameLength,
  codebooks,
) {
  const encoder = new runtime.QuantizedLmChunkEncoder(bundleText, lmWeights, scale);
  const stepCodes = new Uint16Array(codebooks);
  const started = performance.now();
  try {
    for (let step = 0; step < frameLength; ++step) {
      for (let codebook = 0; codebook < codebooks; ++codebook) {
        stepCodes[codebook] = codes[codebook * frameLength + step];
      }
      encoder.push(stepCodes);
    }
    return {
      payload: encoder.finish(),
      elapsedMs: performance.now() - started,
    };
  } finally {
    encoder.free();
  }
}

function f32Bits(value) {
  const bytes = new ArrayBuffer(4);
  new Float32Array(bytes)[0] = value;
  return new Uint32Array(bytes)[0];
}

function assertBytesEqual(left, right, label) {
  if (left.length !== right.length) {
    throw new Error(`${label} length differs: ${left.length} != ${right.length}`);
  }
  for (let index = 0; index < left.length; ++index) {
    if (left[index] !== right[index]) {
      throw new Error(`${label} differs at byte ${index}`);
    }
  }
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function sha256Many(parts) {
  const hash = createHash("sha256");
  for (const part of parts) hash.update(part);
  return hash.digest("hex");
}
