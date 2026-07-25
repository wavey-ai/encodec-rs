#!/usr/bin/env node

/* Run the mandatory macOS Core ML CPU/GPU qualification rows. */

import { appendFileSync, mkdirSync, readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import { acquireQualificationRunLock } from "./run-lock.mjs";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
await acquireQualificationRunLock("run-coreml");
const runRoot = path.resolve(process.argv[2]);
const bundleRoot = path.resolve(process.argv[3] ?? path.join(repoRoot, "onnx-bundles"));
const binary = path.resolve(process.argv[4] ?? path.join(repoRoot, "target/release/encodec-rs"));
const candidates = (process.env.COREML_CANDIDATES ?? "fixed-1333,fixed-1800").split(",").filter(Boolean);
const rates = (process.env.COREML_RATES ?? "6,12").split(",").map(Number);
const corpusIds = (process.env.COREML_CORPUS_IDS ?? "").split(",").filter(Boolean);

const geometry = {
  "fixed-1333": { suffix: "1333ms", stride: 64000 },
  "fixed-1800": { suffix: "1800ms", stride: 86400 },
};

function sha256File(filePath) {
  return crypto.createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function appendJsonl(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  appendFileSync(filePath, `${JSON.stringify(value)}\n`);
}

function run(command, args, environment) {
  const started = Date.now();
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    env: { ...process.env, ...environment },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  const record = {
    command: [command, ...args],
    elapsed_ms: Date.now() - started,
    exit_status: result.status,
    signal: result.signal,
    stdout_sha256: crypto.createHash("sha256").update(result.stdout ?? "").digest("hex"),
    stderr_sha256: crypto.createHash("sha256").update(result.stderr ?? "").digest("hex"),
    error: result.error ? String(result.error) : null,
  };
  appendJsonl(path.join(runRoot, "logs/coreml-commands.jsonl"), record);
  return record;
}

function wavFrames(filePath) {
  const bytes = readFileSync(filePath);
  if (bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`not a WAV file: ${filePath}`);
  }
  let offset = 12;
  let channels = 0;
  let bits = 0;
  let dataBytes = 0;
  while (offset + 8 <= bytes.length) {
    const id = bytes.toString("ascii", offset, offset + 4);
    const size = bytes.readUInt32LE(offset + 4);
    if (id === "fmt ") {
      channels = bytes.readUInt16LE(offset + 10);
      bits = bytes.readUInt16LE(offset + 22);
    }
    if (id === "data") dataBytes = size;
    offset += 8 + size + (size & 1);
  }
  if (channels !== 2 || bits !== 16 || dataBytes % (channels * bits / 8) !== 0) {
    throw new Error(`unexpected decoded WAV format channels=${channels} bits=${bits}`);
  }
  return dataBytes / (channels * bits / 8);
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

const manifest = readJson(path.join(runRoot, "corpus-manifest.json"));
const entries = manifest.entries;
const resultPath = path.join(runRoot, "metrics/results.jsonl");
const priorAttempts = new Map();
if (existsSync(resultPath)) {
  for (const line of readFileSync(resultPath, "utf8").split("\n").filter(Boolean)) {
    const row = JSON.parse(line);
    if (!row.case_id) continue;
    priorAttempts.set(row.case_id, Math.max(priorAttempts.get(row.case_id) ?? 0, Number(row.attempt ?? 0)));
  }
}
const environment = {
  EMSDK_QUIET: "1",
  ENCODEC_RS_ORT_THREADS: "4",
  ORT_DYLIB_PATH: process.env.ORT_DYLIB_PATH ?? "/opt/homebrew/lib/libonnxruntime.dylib",
};
const cacheDir = path.resolve(
  process.env.COREML_CACHE_DIR ?? path.join(runRoot, "providers/coreml-gpu/coreml-cache"),
);
const rows = [];

for (const rate of rates) {
  for (const candidate of candidates) {
    const bundleDir = path.join(bundleRoot, `encodec_48khz_${rate}kbps_${geometry[candidate].suffix}`);
    for (const corpus of entries.filter((entry) => !corpusIds.length || corpusIds.includes(entry.id))) {
      const caseId = `provider-coreml-gpu-${candidate}-${rate}-${corpus.id}`;
      const priorAttempt = priorAttempts.get(caseId) ?? 0;
      const priorPass = priorAttempt > 0 && readFileSync(resultPath, "utf8").split("\n").filter(Boolean).some((line) => {
        const row = JSON.parse(line);
        return row.case_id === caseId && row.status === "pass";
      });
      if (priorPass) continue;
      const objectPath = path.join(runRoot, "providers/coreml-gpu/encodes", corpus.id, `${candidate}-${rate}kbps.ecdc`);
      const wavPath = path.join(runRoot, "providers/coreml-gpu/decodes", corpus.id, `${candidate}-${rate}kbps.wav`);
      mkdirSync(path.dirname(objectPath), { recursive: true });
      mkdirSync(path.dirname(wavPath), { recursive: true });
      const flags = ["--coreml", "--coreml-compute-units", "cpu-and-gpu", "--coreml-cache-dir", cacheDir];
      const encode = run(binary, ["onnx-encode", bundleDir, path.join(runRoot, corpus.canonical_wav_path), objectPath, "--batch-size", "8", ...flags], environment);
      let decode = null;
      let decodedFrames = null;
      if (encode.exit_status === 0) {
        decode = run(binary, ["onnx-decode", bundleDir, objectPath, wavPath, ...flags], environment);
        if (decode.exit_status === 0) decodedFrames = wavFrames(wavPath);
      }
      const status = encode.exit_status === 0 && decode?.exit_status === 0 && decodedFrames === corpus.samples ? "pass" : "fail";
      const row = {
        schema: "wavey.encodec.qualification",
        schema_version: 1,
        run_id: path.basename(runRoot),
        case_id: caseId,
        attempt: priorAttempt + 1,
        status,
        phase: "encode",
        candidate,
        bandwidth_kbps: rate,
        corpus_id: corpus.id,
        source_sha256: corpus.source_sha256,
        canonical_pcm_sha256: corpus.canonical_pcm_sha256,
        model_input_sha256: null,
        codes_sha256: null,
        scale_bits_sha256: null,
        entropy_sha256: null,
        container_sha256: encode.exit_status === 0 ? sha256File(objectPath) : null,
        container_bytes: encode.exit_status === 0 ? readFileSync(objectPath).length : null,
        environment_id: "mac-arm64-coreml-gpu",
        artifact_lock_sha256: null,
        elapsed_ms: encode.elapsed_ms + (decode?.elapsed_ms ?? 0),
        metrics: { decoded_frames: decodedFrames, expected_frames: corpus.samples, length_match: decodedFrames === corpus.samples },
        damage: null,
        error: status === "pass" ? null : "Core ML encode or decode failed, or decoded sample count differed",
        provider: { execution_target: "coreml", compute_units: "cpu-and-gpu", cache_dir: path.relative(runRoot, cacheDir) },
        commands: { encode, decode },
        output_path: path.relative(runRoot, objectPath),
      };
      appendJsonl(resultPath, row);
      rows.push(row);
      console.log(JSON.stringify({ case_id: caseId, status, elapsed_ms: row.elapsed_ms }));
    }
  }
}

console.log(JSON.stringify({ rows: rows.length, pass: rows.filter((row) => row.status === "pass").length, fail: rows.filter((row) => row.status === "fail").length }, null, 2));
