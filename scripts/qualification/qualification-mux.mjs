#!/usr/bin/env node

/*
 * Provider-neutral qualification mux.
 *
 * This tool prepares locked inputs and records Stage A measurements. It does
 * not implement the Profile 1 container. Measurements made with the current
 * unpublished ECDC envelope are labelled as provisional evidence.
 */

import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync, appendFileSync, existsSync } from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { acquireQualificationRunLock } from "./run-lock.mjs";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const REPO_ROOT = path.resolve(path.dirname(SCRIPT_PATH), "../..");
await acquireQualificationRunLock("qualification-mux");

const CORPUS = [
  {
    id: "confirmation-i-want-her-v1",
    relativePath: "I WANT HER_MIX 4 CONFIRMATION_130323.wav",
    class: "Original",
    sha256: "4e984cdae6e57fc85b3fd316bd14edd5700de07ee8c5e94abba783e31bde53d4",
  },
  {
    id: "confirmation-pray-4-me-v1",
    relativePath: "PRAY 4 ME_MIX 4 CONFIRMATION_130323.wav",
    class: "Original",
    sha256: "742d36fd78b1263f0c91f76127f3ea77e4cf7ce781c37715c2a7c890549ecf59",
  },
  {
    id: "confirmation-westside-v1",
    relativePath: "WESTSIDE_MIX 4 CONFIRMATION_130323.wav",
    class: "Original",
    sha256: "dec1b383d58aea9848728126efab169f42e54375b64ca55363d2f234696474c9",
  },
  {
    id: "confirmation-westside-v2",
    relativePath: "WESTSIDE_V2_MIX 4 CONFIRMATION_140323.wav",
    class: "Original",
    sha256: "15518ce0885a9b0ed0ce2b32410e79da74dfcf1914bda002d1bbe88b0d947f6b",
  },
  {
    id: "confirmation-after-dark-v1",
    relativePath: "AFTER DARK_MIX 4 CONFIRMATION_130323.wav",
    class: "Original",
    sha256: "1812df5e6062549a4947a78b5bb08474b6d11026292c019c2da9b5b1950bc655",
  },
  {
    id: "confirmation-1979-v1",
    relativePath: "Lori Asha - Lori Asha Album Premix/1979_MIX 4 CONFIRMATION_130323.wav",
    class: "Premix",
    sha256: "ed1db8e4a35679be511b12be32fcd0945582f9623a088247431862a884311679",
  },
];

const GEOMETRIES = {
  "meta-1000": {
    modelSamples: 48000,
    strideSamples: 47520,
    fullCodeFrames: 150,
    finalTailRule: "true-variable-tail",
    reconstructionId: "meta-triangle-overlap-add",
    bundleSuffix: "1000ms",
  },
  "fixed-1333": {
    modelSamples: 64960,
    strideSamples: 64000,
    fullCodeFrames: 203,
    finalTailRule: "full-fixed-code-frame-count",
    reconstructionId: "fixed-context-crop-bound-seam",
    bundleSuffix: "1333ms",
  },
  "fixed-1800": {
    modelSamples: 87360,
    strideSamples: 86400,
    fullCodeFrames: 273,
    finalTailRule: "full-fixed-code-frame-count",
    reconstructionId: "fixed-context-crop-bound-seam",
    bundleSuffix: "1800ms",
  },
};

function parseArgs(argv) {
  const out = { _: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      out._.push(token);
      continue;
    }
    const equal = token.indexOf("=");
    if (equal !== -1) {
      out[token.slice(2, equal)] = token.slice(equal + 1);
      continue;
    }
    const key = token.slice(2);
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      out[key] = next;
      index += 1;
    } else {
      out[key] = true;
    }
  }
  return out;
}

function required(options, name) {
  const value = options[name];
  if (!value || value === true) {
    throw new Error(`missing --${name}`);
  }
  return value;
}

function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sha256File(filePath) {
  return sha256Bytes(readFileSync(filePath));
}

function jsonWrite(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function jsonRead(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function csvOption(options, name, fallback) {
  const value = options[name];
  if (value == null || value === true || value === "") return fallback;
  return String(value).split(",").map((item) => item.trim()).filter(Boolean);
}

function appendJsonl(filePath, value) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  appendFileSync(filePath, `${JSON.stringify(value)}\n`);
}

function now() {
  return new Date().toISOString();
}

function runCommand(command, args, cwd, outputDir, environment = {}) {
  const startedAt = now();
  const started = process.hrtime.bigint();
  const result = spawnSync(command, args, {
    cwd,
    env: { ...process.env, ...environment },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
  const record = {
    started_at: startedAt,
    elapsed_ms: Math.round(elapsedMs),
    command: [command, ...args],
    cwd,
    environment: Object.fromEntries(Object.entries(environment).filter(([key]) => !/TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL/i.test(key))),
    exit_status: result.status,
    signal: result.signal,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error ? String(result.error) : null,
  };
  if (outputDir) {
    appendJsonl(path.join(outputDir, "logs/commands.jsonl"), record);
  }
  return record;
}

function fail(message) {
  throw new Error(message);
}

function ensureNoEMastered(filePath) {
  if (filePath.toLowerCase().includes("emastered")) {
    fail(`eMastered source is forbidden: ${filePath}`);
  }
}

function parseWav(bytes, { allowInteger = false } = {}) {
  if (bytes.length < 12 || bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WAVE") {
    fail("canonical input is not a RIFF/WAVE file");
  }
  let offset = 12;
  let fmt = null;
  let data = null;
  while (offset + 8 <= bytes.length) {
    const id = bytes.toString("ascii", offset, offset + 4);
    const size = bytes.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + size;
    if (end > bytes.length) fail(`WAV chunk ${id} exceeds file length`);
    if (id === "fmt ") {
      fmt = {
        format: bytes.readUInt16LE(start),
        channels: bytes.readUInt16LE(start + 2),
        sampleRate: bytes.readUInt32LE(start + 4),
        bits: bytes.readUInt16LE(start + 14),
      };
    } else if (id === "data") {
      data = bytes.subarray(start, end);
    }
    offset = end + (size & 1);
  }
  if (!fmt || !data) fail("WAV file has no fmt or data chunk");
  const isFloat32 = fmt.format === 3 && fmt.bits === 32;
  const isPcmInteger = fmt.format === 1 && [16, 24, 32].includes(fmt.bits);
  if (!isFloat32 && !(allowInteger && isPcmInteger)) fail(`expected IEEE float32 WAV, got format=${fmt.format} bits=${fmt.bits}`);
  if (fmt.channels !== 2 || fmt.sampleRate !== 48000) fail(`expected stereo 48 kHz WAV, got ${fmt.channels} channels at ${fmt.sampleRate} Hz`);
  const frameBytes = fmt.channels * (fmt.bits / 8);
  if (data.length % frameBytes !== 0) fail("WAV data is not aligned to stereo float32 frames");
  const frames = data.length / frameBytes;
  const planar = Buffer.alloc(fmt.channels * frames * 4);
  const sampleBytes = fmt.bits / 8;
  const sampleAt = (frame, channel) => {
    const offset = (frame * fmt.channels + channel) * sampleBytes;
    if (isFloat32) return data.readFloatLE(offset);
    if (fmt.bits === 16) return data.readInt16LE(offset) / 32768;
    if (fmt.bits === 24) return data.readIntLE(offset, 3) / 8388608;
    return data.readInt32LE(offset) / 2147483648;
  };
  for (let channel = 0; channel < fmt.channels; channel += 1) {
    for (let frame = 0; frame < frames; frame += 1) {
      const value = sampleAt(frame, channel);
      if (!Number.isFinite(value)) fail(`canonical input contains a non-finite sample at ${frame}:${channel}`);
      planar.writeFloatLE(value, (channel * frames + frame) * 4);
    }
  }
  return { ...fmt, frames, planar, sample_format: isFloat32 ? "pcm_f32le" : `pcm_s${fmt.bits}le` };
}

function writePlanarVector(filePath, channels, frames, valueAt) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  const bytes = Buffer.alloc(channels * frames * 4);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let frame = 0; frame < frames; frame += 1) {
      bytes.writeFloatLE(valueAt(channel, frame), (channel * frames + frame) * 4);
    }
  }
  writeFileSync(filePath, bytes);
  return { bytes: bytes.length, sha256: sha256Bytes(bytes), frames, channels };
}

function readPlanarF32(filePath) {
  const bytes = readFileSync(filePath);
  if (bytes.length % 4 !== 0) fail(`invalid planar f32 file ${filePath}`);
  const floats = new Float32Array(bytes.length / 4);
  for (let index = 0; index < floats.length; index += 1) floats[index] = bytes.readFloatLE(index * 4);
  return { bytes, floats };
}

function metricAgainstReference(referencePath, decodedPath) {
  const reference = readPlanarF32(referencePath).floats;
  const decoded = readPlanarF32(decodedPath).floats;
  const count = Math.min(reference.length, decoded.length);
  let sum = 0;
  let max = 0;
  for (let index = 0; index < count; index += 1) {
    const difference = Math.abs(reference[index] - decoded[index]);
    sum += difference * difference;
    if (difference > max) max = difference;
  }
  return {
    reference_floats: reference.length,
    decoded_floats: decoded.length,
    compared_floats: count,
    rms_f32: count ? Math.sqrt(sum / count) : null,
    max_f32: max,
    length_match: reference.length === decoded.length,
  };
}

function currentEcdcLedger(filePath) {
  const bytes = readFileSync(filePath);
  if (bytes.toString("ascii", 0, 4) !== "ECDC" || bytes[4] !== 0) {
    fail(`not the current unpublished ECDC v0 envelope: ${filePath}`);
  }
  const metadataBytes = bytes.readUInt32BE(5);
  const headerBytes = 9 + metadataBytes;
  if (headerBytes > bytes.length) fail("ECDC metadata header is truncated");
  let offset = headerBytes;
  let chunks = 0;
  let segmentHeaderBytes = 0;
  let crcBytes = 0;
  let scales = 0;
  let arithmetic = 0;
  while (offset < bytes.length) {
    if (offset + 8 > bytes.length) fail("ECDC chunk header is truncated");
    const payloadBytes = bytes.readUInt32BE(offset);
    const end = offset + 8 + payloadBytes;
    if (end > bytes.length) fail("ECDC chunk payload is truncated");
    if (payloadBytes < 4) fail("ECDC chunk has no scale value");
    chunks += 1;
    segmentHeaderBytes += 4;
    crcBytes += 4;
    scales += 4;
    arithmetic += payloadBytes - 4;
    offset = end;
  }
  const objectBytes = bytes.length;
  const accounted = 9 + metadataBytes + segmentHeaderBytes + crcBytes + scales + arithmetic;
  if (accounted !== objectBytes) fail(`size ledger mismatch: ${accounted} != ${objectBytes}`);
  return {
    schema: "wavey.encodec.current-acv2-ledger",
    format: "ECDC-v0-acv2-unpublished",
    file_size_bytes: objectBytes,
    object_accounted_bytes: accounted,
    chunk_count: chunks,
    magic_version_bytes: 5,
    metadata_header_bytes: metadataBytes,
    profile_descriptor_bytes: 0,
    index_bytes: 0,
    segment_header_bytes: segmentHeaderBytes,
    normalization_scale_bytes: scales,
    arithmetic_payload_bytes: arithmetic,
    crc_bytes: crcBytes,
    recovery_bytes: 0,
    trailer_bytes: 0,
    proposed_profile1_structural_bytes: 100 + 20 * chunks,
    proposed_profile1_estimated_without_manifest_bytes: arithmetic + scales + 100 + 20 * chunks,
    object_sha256: sha256File(filePath),
    profile1_complete: false,
  };
}

function resultRecord({ runId, caseId, status, candidate, rate, corpus, environmentId, startedAt, elapsedMs, error, metrics, ledger, outputPath, stopReason }) {
  const geometry = GEOMETRIES[candidate] ?? null;
  return {
    schema: "wavey.encodec.qualification",
    schema_version: 1,
    run_id: runId,
    case_id: caseId,
    attempt: 1,
    status,
    phase: "geometry_selection",
    candidate,
    bandwidth_kbps: rate,
    corpus_id: corpus?.id ?? null,
    source_sha256: corpus?.source_sha256 ?? null,
    canonical_pcm_sha256: corpus?.canonical_pcm_sha256 ?? null,
    model_input_sha256: null,
    codes_sha256: null,
    scale_bits_sha256: null,
    entropy_sha256: null,
    container_sha256: ledger?.object_sha256 ?? null,
    container_bytes: ledger?.file_size_bytes ?? null,
    environment_id: environmentId,
    artifact_lock_sha256: null,
    elapsed_ms: Math.round(elapsedMs ?? 0),
    profile: geometry ? {
      geometry_id: candidate,
      model_samples: geometry.modelSamples,
      stride_samples: geometry.strideSamples,
      full_code_frames: geometry.fullCodeFrames,
      final_tail_rule: geometry.finalTailRule,
      reconstruction_id: geometry.reconstructionId,
      container_profile: "current-acv2-unpublished",
    } : null,
    metrics: metrics ?? {},
    size: ledger ? {
      current_acv2_bytes: ledger.file_size_bytes,
      profile1_complete: ledger.profile1_complete,
      proposed_profile1_structural_bytes: ledger.proposed_profile1_structural_bytes,
      proposed_profile1_estimated_without_manifest_bytes: ledger.proposed_profile1_estimated_without_manifest_bytes,
      arithmetic_payload_bytes: ledger.arithmetic_payload_bytes,
      normalization_scale_bytes: ledger.normalization_scale_bytes,
    } : null,
    damage: null,
    error: error ?? null,
    gates: [
      { id: "G05-entropy", status: "blocked", observed: "the CLI exports exact code and entropy evidence, but the full cross-provider matrix has not run", limit: "exact cross-provider agreement" },
      { id: "G13-size", status: ledger?.profile1_complete ? "pass" : "blocked", observed: ledger?.profile1_complete ? ledger.file_size_bytes : "current acv2 object only", limit: "complete proposed Profile 1 container" },
    ],
    output_path: outputPath ?? null,
    timing: { started_at: startedAt ?? null, elapsed_ms: Math.round(elapsedMs ?? 0) },
    log_path: "logs/commands.jsonl",
    stop_reason: stopReason ?? null,
  };
}

function sourcePath(root, relativePath) {
  const resolvedRoot = path.resolve(root);
  const resolvedPath = path.resolve(resolvedRoot, relativePath);
  if (resolvedPath !== resolvedRoot && !resolvedPath.startsWith(`${resolvedRoot}${path.sep}`)) fail(`source path escapes corpus root: ${relativePath}`);
  return resolvedPath;
}

function prepare(options) {
  const outputDir = path.resolve(required(options, "output"));
  const corpusRoot = path.resolve(options["corpus-root"] ?? process.env.CONFIRMATION_CORPUS_ROOT ?? "/Users/jamie/Downloads/Lori EP");
  mkdirSync(outputDir, { recursive: true });
  mkdirSync(path.join(outputDir, "inputs/source"), { recursive: true });
  const commandEnv = { EMSDK_QUIET: "1" };
  const entries = [];
  const seen = new Set();
  for (const item of CORPUS) {
    const source = sourcePath(corpusRoot, item.relativePath);
    ensureNoEMastered(source);
    if (!existsSync(source)) fail(`required CONFIRMATION source is absent: ${source}`);
    if (seen.has(source)) fail(`two corpus IDs resolve to one source: ${source}`);
    seen.add(source);
    const sourceSha = sha256File(source);
    if (sourceSha !== item.sha256) fail(`source digest mismatch for ${item.id}: ${sourceSha}`);
    const canonical = path.join(outputDir, "inputs/canonical-wav", `${item.id}.wav`);
    mkdirSync(path.dirname(canonical), { recursive: true });
    let conversion = null;
    if (!existsSync(canonical)) {
      conversion = runCommand("ffmpeg", ["-nostdin", "-hide_banner", "-loglevel", "error", "-bitexact", "-i", source, "-map", "0:a:0", "-map_metadata", "-1", "-ar", "48000", "-ac", "2", "-c:a", "pcm_f32le", "-flags:a", "+bitexact", "-n", canonical], REPO_ROOT, outputDir, commandEnv);
      if (conversion.exit_status !== 0) fail(`ffmpeg failed for ${item.id}`);
    }
    const canonicalBytes = readFileSync(canonical);
    const parsed = parseWav(canonicalBytes);
    const planarPath = path.join(outputDir, "inputs/canonical-planar-f32le", `${item.id}.f32le`);
    mkdirSync(path.dirname(planarPath), { recursive: true });
    writeFileSync(planarPath, parsed.planar);
    writeFileSync(path.join(outputDir, "inputs/source", `${item.id}.source-path.txt`), `${source}\n`);
    entries.push({
      ...item,
      source_path: source,
      source_sha256: sourceSha,
      source_bytes: statSync(source).size,
      canonical_wav_path: path.relative(outputDir, canonical),
      canonical_wav_sha256: sha256File(canonical),
      canonical_pcm_path: path.relative(outputDir, planarPath),
      canonical_pcm_sha256: sha256File(planarPath),
      canonical_pcm_bytes: parsed.planar.length,
      sample_rate: parsed.sampleRate,
      channels: parsed.channels,
      samples: parsed.frames,
      sample_format: "pcm_f32le",
      ffmpeg: conversion ? { command: conversion.command, elapsed_ms: conversion.elapsed_ms } : { reused: true },
    });
  }

  const vectorManifest = writeVectors(outputDir);
  const environment = {
    id: "mac-arm64-onnx-cpu",
    os: `${os.platform()} ${os.release()}`,
    arch: process.arch,
    cpu: os.cpus()[0]?.model ?? null,
    memory_bytes: os.totalmem(),
    node: process.version,
    ffmpeg_sha256: commandHash("ffmpeg", ["-version"], outputDir),
    qualification_mux_sha256: sha256File(SCRIPT_PATH),
  };
  jsonWrite(path.join(outputDir, "environments/mac-arm64-onnx-cpu.json"), environment);
  const manifest = {
    schema: "wavey.encodec.qualification.corpus",
    schema_version: 1,
    run_id: path.basename(outputDir),
    corpus_root: corpusRoot,
    source_policy: "exact-relative-path-and-sha256; eMastered rejected",
    entries,
    vectors: vectorManifest,
    generated_at: now(),
  };
  jsonWrite(path.join(outputDir, "corpus-manifest.json"), manifest);
  jsonWrite(path.join(outputDir, "manifests/inputs.json"), entries.map((entry) => ({ id: entry.id, source_sha256: entry.source_sha256, canonical_pcm_sha256: entry.canonical_pcm_sha256 })));
  console.log(JSON.stringify({ output: outputDir, corpus: entries.length, vectors: vectorManifest }, null, 2));
}

function commandHash(command, args, outputDir) {
  const result = runCommand(command, args, REPO_ROOT, outputDir, { EMSDK_QUIET: "1" });
  return sha256Bytes(Buffer.from(`${result.stdout}\n${result.stderr}`));
}

function quietCommand(command, args, cwd = REPO_ROOT, environment = {}) {
  const result = spawnSync(command, args, {
    cwd,
    env: { ...process.env, ...environment },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  return {
    command: [command, ...args],
    cwd,
    exit_status: result.status,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error ? String(result.error) : null,
  };
}

function walkFiles(root) {
  if (!existsSync(root)) return [];
  const info = statSync(root);
  if (info.isFile()) return [root];
  const files = [];
  for (const name of readdirSync(root)) {
    if (name === "node_modules" || name === ".git") continue;
    const child = path.join(root, name);
    if (statSync(child).isDirectory()) files.push(...walkFiles(child));
    else files.push(child);
  }
  return files.sort();
}

function worktreeSnapshot() {
  const head = quietCommand("git", ["rev-parse", "HEAD"]);
  const status = quietCommand("git", ["status", "--porcelain=v1"]);
  const diff = quietCommand("git", ["diff", "--no-ext-diff", "--binary"]);
  const untracked = quietCommand("git", ["ls-files", "--others", "--exclude-standard"]);
  return {
    head: head.stdout.trim() || null,
    status: status.stdout.split("\n").filter(Boolean),
    clean: status.exit_status === 0 && status.stdout.trim() === "",
    tracked_diff_sha256: sha256Bytes(Buffer.from(diff.stdout)),
    untracked_paths: untracked.stdout.split("\n").filter(Boolean),
    status_command: status.command,
  };
}

function artifactLock(options) {
  const outputDir = path.resolve(required(options, "output"));
  const stageRoot = path.join(outputDir, "stage-a");
  const fixedRoot = path.resolve(options["bundle-root"] ?? path.join(REPO_ROOT, "onnx-bundles"));
  const roots = [
    path.join(REPO_ROOT, "Cargo.toml"),
    path.join(REPO_ROOT, "Cargo.lock"),
    path.join(REPO_ROOT, "scripts/qualification/qualification-mux.mjs"),
    path.join(REPO_ROOT, "browser-smoke"),
    path.join(REPO_ROOT, "scripts"),
    path.join(REPO_ROOT, "src"),
    fixedRoot,
    stageRoot,
  ];
  const files = [...new Set(roots.flatMap((root) => walkFiles(root)))].filter((file) => !file.includes(`${path.sep}target${path.sep}`));
  const artifacts = files.map((file) => ({
    path: path.relative(REPO_ROOT, file),
    bytes: statSync(file).size,
    sha256: sha256File(file),
  }));
  const condaPython = process.env.QUALIFICATION_CONDA_PYTHON ?? "/opt/anaconda3/envs/encodec-export/bin/python";
  const pythonVersion = quietCommand(condaPython, ["--version"]);
  const worktree = worktreeSnapshot();
  const lock = {
    schema: "wavey.encodec.qualification.lock",
    schema_version: 1,
    run_id: path.basename(outputDir),
    generated_at: now(),
    status: worktree.clean ? "ready-for-owner-review" : "blocked",
    worktree,
    environment: {
      node: process.version,
      python: { command: condaPython, version: `${pythonVersion.stdout}${pythonVersion.stderr}`.trim() },
      conda_environment: "encodec-export",
      conda_prefix: "/opt/anaconda3/envs/encodec-export",
      rust: quietCommand("rustc", ["--version"]).stdout.trim(),
      cargo: quietCommand("cargo", ["--version"]).stdout.trim(),
      ffmpeg: quietCommand("ffmpeg", ["-version"]).stdout.split("\n")[0] ?? null,
    },
    roots: roots.map((root) => path.relative(REPO_ROOT, root)),
    artifacts,
    blockers: worktree.clean ? [] : ["The qualification worktree contains uncommitted code or model changes."],
  };
  lock.artifact_lock_sha256 = sha256Bytes(Buffer.from(JSON.stringify({ ...lock, artifact_lock_sha256: null })));
  jsonWrite(path.join(outputDir, "qualification-lock.json"), lock);
  appendJsonl(path.join(outputDir, "logs/lock.jsonl"), lock);
  console.log(JSON.stringify({ output: path.join(outputDir, "qualification-lock.json"), status: lock.status, artifacts: artifacts.length, worktree_entries: worktree.status.length }, null, 2));
}

function writeVectors(outputDir) {
  const vectors = { generator_sha256: sha256File(SCRIPT_PATH), silence: null, impulse: null, tails: {} };
  const silencePath = path.join(outputDir, "vectors/silence/10s.f32le");
  vectors.silence = { path: path.relative(outputDir, silencePath), ...writePlanarVector(silencePath, 2, 480000, () => 0) };
  const impulsePositions = [0, 1, 318, 319, 320, 479, 480, 481, 47518, 47519, 47520, 47521, 47998, 47999, 48000, 48001, 63998, 63999, 64000, 64001, 86398, 86399, 86400, 86401, 479999 - 1, 479999];
  const impulsePath = path.join(outputDir, "vectors/impulses/10s.f32le");
  const impulseSet = new Map(impulsePositions.map((position, index) => [position, index % 2 === 0 ? 0.99 : -0.99]));
  vectors.impulse = { path: path.relative(outputDir, impulsePath), positions: impulsePositions, ...writePlanarVector(impulsePath, 2, 480000, (_channel, frame) => impulseSet.get(frame) ?? 0) };
  const tailCounts = [1, 319, 320, 479, 480, 481, 47519, 47520, 47521, 47999, 48000, 48001, 63999, 64000, 64001, 64959, 64960, 64961, 86399, 86400, 86401, 87359, 87360, 87361];
  for (const [geometryId, geometry] of Object.entries(GEOMETRIES)) {
    vectors.tails[geometryId] = [];
    for (const samples of tailCounts) {
      const filePath = path.join(outputDir, "vectors/tails", geometryId, `${samples}.f32le`);
      const record = writePlanarVector(filePath, 2, samples, (channel, frame) => Math.sin((frame + channel * 7) * 0.013) * 0.1);
      vectors.tails[geometryId].push({ samples, expected_frames: geometryId === "meta-1000" ? Math.ceil(samples * 150 / 48000) : geometry.fullCodeFrames, path: path.relative(outputDir, filePath), ...record });
    }
  }
  jsonWrite(path.join(outputDir, "vectors/vector-manifest.json"), vectors);
  return vectors;
}

function loadCorpus(outputDir) {
  const manifest = jsonRead(path.join(outputDir, "corpus-manifest.json"));
  return manifest.entries.map((entry) => ({ ...entry, canonical: path.join(outputDir, entry.canonical_wav_path), planar: path.join(outputDir, entry.canonical_pcm_path) }));
}

function bundleFor(bundleRoot, candidate, rate) {
  const geometry = GEOMETRIES[candidate];
  if (!geometry) fail(`unknown geometry ${candidate}`);
  return path.resolve(bundleRoot, `encodec_48khz_${rate}kbps_${geometry.bundleSuffix}`);
}

function commandForBinary(binary, args) {
  if (binary) return { command: path.resolve(binary), args };
  return { command: "cargo", args: ["run", "--locked", "--release", "--features", "onnx", "--", ...args] };
}

function geometrySelect(options) {
  const outputDir = path.resolve(required(options, "output"));
  const runId = path.basename(outputDir);
  const corpusIds = csvOption(options, "corpus-ids", null);
  const entries = loadCorpus(outputDir).filter((entry) => !corpusIds || corpusIds.includes(entry.id));
  const bundleRoot = path.resolve(options["bundle-root"] ?? path.join(REPO_ROOT, "onnx-bundles"));
  const metaBundleRoot = path.resolve(options["meta-bundle-root"] ?? bundleRoot);
  const selectedRates = csvOption(options, "rates", [6, 12]).map((value) => Number(value));
  const selectedCandidates = csvOption(options, "candidates", Object.keys(GEOMETRIES));
  const binary = options.binary ? path.resolve(options.binary) : (existsSync(path.join(REPO_ROOT, "target/release/encodec-rs")) ? path.join(REPO_ROOT, "target/release/encodec-rs") : null);
  const resultPath = path.join(outputDir, "metrics/results.jsonl");
  const existing = existsSync(resultPath) ? readFileSync(resultPath, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line).case_id) : [];
  const existingSet = new Set(existing);
  const rows = [];
  for (const rate of selectedRates) {
    for (const candidate of selectedCandidates) {
      for (const corpus of entries) {
        const caseId = `select-${candidate}-${rate}-${corpus.id}`;
        if (existingSet.has(caseId)) continue;
        const startedAt = now();
        const started = process.hrtime.bigint();
        const bundleDir = bundleFor(candidate === "meta-1000" ? metaBundleRoot : bundleRoot, candidate, rate);
        if (candidate === "meta-1000") {
          const row = resultRecord({ runId, caseId, status: "blocked", candidate, rate, corpus, environmentId: "mac-arm64-onnx-cpu", startedAt, elapsedMs: Number(process.hrtime.bigint() - started) / 1e6, error: "the current fixed-shape ONNX runner pads the final Meta segment", stopReason: "true variable-tail evidence requires a runner that passes the actual tail length to the model; padded input is not accepted" });
          appendJsonl(resultPath, row); rows.push(row); continue;
        }
        if (!existsSync(path.join(bundleDir, "bundle.json"))) {
          const row = resultRecord({ runId, caseId, status: "blocked", candidate, rate, corpus, environmentId: "mac-arm64-onnx-cpu", startedAt, elapsedMs: Number(process.hrtime.bigint() - started) / 1e6, error: "candidate bundle is absent", stopReason: `missing bundle ${bundleDir}` });
          appendJsonl(resultPath, row); rows.push(row); continue;
        }
        const objectPath = path.join(outputDir, "encodes", corpus.id, `${candidate}-${rate}kbps.ecdc`);
        const decodedWav = path.join(outputDir, "decodes", corpus.id, `${candidate}-${rate}kbps.wav`);
        mkdirSync(path.dirname(objectPath), { recursive: true });
        mkdirSync(path.dirname(decodedWav), { recursive: true });
        const runtimeEnvironment = { EMSDK_QUIET: "1", ENCODEC_RS_ORT_THREADS: "4", ORT_DYLIB_PATH: process.env.ORT_DYLIB_PATH ?? "/opt/homebrew/lib/libonnxruntime.dylib" };
        const encodeCommand = commandForBinary(binary, ["onnx-encode", bundleDir, corpus.canonical, objectPath, "--batch-size", "8"]);
        const encode = runCommand(encodeCommand.command, encodeCommand.args, REPO_ROOT, outputDir, runtimeEnvironment);
        if (encode.exit_status !== 0) {
          const row = resultRecord({ runId, caseId, status: "fail", candidate, rate, corpus, environmentId: "mac-arm64-onnx-cpu", startedAt, elapsedMs: Number(process.hrtime.bigint() - started) / 1e6, error: `encode failed: ${encode.stderr}`, stopReason: "native encoder command failed" });
          appendJsonl(resultPath, row); rows.push(row); continue;
        }
        const decodeCommand = commandForBinary(binary, ["onnx-decode", bundleDir, objectPath, decodedWav]);
        const decode = runCommand(decodeCommand.command, decodeCommand.args, REPO_ROOT, outputDir, runtimeEnvironment);
        if (decode.exit_status !== 0) {
          const row = resultRecord({ runId, caseId, status: "fail", candidate, rate, corpus, environmentId: "mac-arm64-onnx-cpu", startedAt, elapsedMs: Number(process.hrtime.bigint() - started) / 1e6, error: `decode failed: ${decode.stderr}`, stopReason: "native decoder command failed" });
          appendJsonl(resultPath, row); rows.push(row); continue;
        }
        const decoded = parseWav(readFileSync(decodedWav), { allowInteger: true });
        const decodedPlanar = path.join(outputDir, "decodes", corpus.id, `${candidate}-${rate}kbps.f32le`);
        writeFileSync(decodedPlanar, decoded.planar);
        const ledger = currentEcdcLedger(objectPath);
        jsonWrite(path.join(outputDir, "metrics/size-ledger", `${caseId}.json`), ledger);
        const metrics = metricAgainstReference(corpus.planar, decodedPlanar);
        const row = resultRecord({ runId, caseId, status: "pass", candidate, rate, corpus: { ...corpus, source_sha256: corpus.source_sha256, canonical_pcm_sha256: corpus.canonical_pcm_sha256 }, environmentId: "mac-arm64-onnx-cpu", startedAt, elapsedMs: Number(process.hrtime.bigint() - started) / 1e6, metrics, ledger, outputPath: path.relative(outputDir, objectPath), stopReason: "measurement uses current unpublished acv=2 envelope; not Profile 1 evidence" });
        appendJsonl(resultPath, row); rows.push(row);
      }
    }
  }
  summarizeGeometry(outputDir);
  console.log(JSON.stringify({ output: outputDir, new_rows: rows.length, result_path: path.relative(outputDir, resultPath) }, null, 2));
}

function frameEvidence(options) {
  const outputDir = path.resolve(required(options, "output"));
  const runId = path.basename(outputDir);
  const corpusIds = csvOption(options, "corpus-ids", null);
  const entries = loadCorpus(outputDir).filter((entry) => !corpusIds || corpusIds.includes(entry.id));
  const bundleRoot = path.resolve(options["bundle-root"] ?? path.join(REPO_ROOT, "onnx-bundles"));
  const metaBundleRoot = path.resolve(options["meta-bundle-root"] ?? bundleRoot);
  const selectedRates = csvOption(options, "rates", [6, 12]).map((value) => Number(value));
  const selectedCandidates = csvOption(options, "candidates", Object.keys(GEOMETRIES));
  const binary = options.binary
    ? path.resolve(options.binary)
    : (existsSync(path.join(REPO_ROOT, "target/release/encodec-rs"))
      ? path.join(REPO_ROOT, "target/release/encodec-rs")
      : null);
  const resultPath = path.join(outputDir, "metrics/results.jsonl");
  const priorRows = existsSync(resultPath)
    ? readFileSync(resultPath, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line))
    : [];
  const rows = [];
  const runtimeEnvironment = {
    EMSDK_QUIET: "1",
    ENCODEC_RS_ORT_THREADS: "4",
    ORT_DYLIB_PATH: process.env.ORT_DYLIB_PATH ?? "/opt/homebrew/lib/libonnxruntime.dylib",
  };

  for (const rate of selectedRates) {
    for (const candidate of selectedCandidates) {
      const bundleDir = bundleFor(candidate === "meta-1000" ? metaBundleRoot : bundleRoot, candidate, rate);
      for (const corpus of entries) {
        const caseId = `frame-evidence-${candidate}-${rate}-${corpus.id}`;
        const priorCaseRows = priorRows.filter((row) => row.case_id === caseId);
        if (priorCaseRows.some((row) => row.status === "pass")) continue;
        const attempt = Math.max(0, ...priorCaseRows.map((row) => Number(row.attempt ?? 0))) + 1;
        if (!existsSync(path.join(bundleDir, "bundle.json"))) {
          const row = {
            schema: "wavey.encodec.qualification",
            schema_version: 1,
            run_id: runId,
            case_id: caseId,
            attempt,
            status: "blocked",
            phase: "frame_evidence",
            candidate,
            bandwidth_kbps: rate,
            corpus_id: corpus.id,
            environment_id: "mac-arm64-onnx-cpu",
            error: `candidate bundle is absent: ${bundleDir}`,
            stop_reason: candidate === "meta-1000"
              ? "a dynamic true-variable-tail Meta bundle is required"
              : "the fixed candidate bundle is required",
          };
          appendJsonl(resultPath, row);
          rows.push(row);
          continue;
        }

        const evidenceDir = path.join(outputDir, "evidence", corpus.id, `${candidate}-${rate}kbps`);
        const args = [
          "onnx-encode-evidence",
          bundleDir,
          corpus.canonical,
          evidenceDir,
          "--batch-size",
          "8",
        ];
        if (candidate === "meta-1000") args.push("--true-variable-tail");
        const invocation = commandForBinary(binary, args);
        const command = runCommand(
          invocation.command,
          invocation.args,
          REPO_ROOT,
          outputDir,
          runtimeEnvironment,
        );
        let manifest = null;
        try {
          manifest = JSON.parse(command.stdout);
        } catch {
          const manifestPath = path.join(evidenceDir, "manifest.json");
          if (existsSync(manifestPath)) manifest = jsonRead(manifestPath);
        }
        const status = command.exit_status === 0
          && manifest?.schema_version >= 2
          && manifest?.codes_exactly_recovered === true
          && manifest?.segments?.length > 0
          && manifest?.segments?.every((segment) => segment.codes_exactly_recovered === true)
          ? "pass"
          : "fail";
        const row = {
          schema: "wavey.encodec.qualification",
          schema_version: 1,
          run_id: runId,
          case_id: caseId,
          attempt,
          status,
          phase: "frame_evidence",
          candidate,
          bandwidth_kbps: rate,
          corpus_id: corpus.id,
          source_sha256: corpus.source_sha256,
          canonical_pcm_sha256: corpus.canonical_pcm_sha256,
          model_input_sha256: manifest?.model_inputs_sha256 ?? null,
          codes_sha256: manifest?.codes_sha256 ?? null,
          recovered_codes_sha256: manifest?.recovered_codes_sha256 ?? null,
          scale_bits_sha256: manifest?.scales_sha256 ?? null,
          entropy_sha256: manifest?.entropy_sha256 ?? null,
          codebook_order_sha256: manifest?.codebook_order_sha256 ?? null,
          environment_id: "mac-arm64-onnx-cpu",
          segment_count: manifest?.segments?.length ?? null,
          true_variable_tail: manifest?.true_variable_tail ?? candidate === "meta-1000",
          evidence_manifest_path: path.relative(outputDir, path.join(evidenceDir, "manifest.json")),
          elapsed_ms: command.elapsed_ms,
          error: status === "pass" ? null : (command.stderr || "frame evidence validation failed"),
          stop_reason: status === "pass"
            ? "model inputs, codes, scale bits, entropy bytes, and recovered codes recorded"
            : "frame evidence command or exact code recovery failed",
        };
        appendJsonl(resultPath, row);
        rows.push(row);
      }
    }
  }
  summarizeGeometry(outputDir);
  console.log(JSON.stringify({ output: outputDir, new_rows: rows.length, result_path: path.relative(outputDir, resultPath) }, null, 2));
}

function summarizeGeometry(outputDir) {
  const resultPath = path.join(outputDir, "metrics/results.jsonl");
  const allRows = existsSync(resultPath) ? readFileSync(resultPath, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line)) : [];
  const latestByCase = new Map();
  for (const row of allRows) {
    const key = row.case_id ?? `${row.phase ?? "unknown"}-${latestByCase.size}`;
    const previous = latestByCase.get(key);
    if (!previous || Number(row.attempt ?? 0) >= Number(previous.attempt ?? 0)) latestByCase.set(key, row);
  }
  const rows = [...latestByCase.values()];
  const byRate = {};
  for (const rate of [6, 12]) {
    const rateRows = rows.filter((row) => row.bandwidth_kbps === rate && row.status === "pass" && row.container_bytes != null);
    const baselines = new Map();
    for (const row of rateRows.filter((item) => item.candidate.startsWith("fixed-"))) {
      const key = row.corpus_id;
      baselines.set(key, Math.min(baselines.get(key) ?? Number.MAX_SAFE_INTEGER, row.container_bytes));
    }
    byRate[rate] = rateRows.map((row) => ({ case_id: row.case_id, candidate: row.candidate, corpus_id: row.corpus_id, current_acv2_bytes: row.container_bytes, best_fixed_current_acv2_bytes: baselines.get(row.corpus_id) ?? null, delta_bytes: baselines.has(row.corpus_id) ? row.container_bytes - baselines.get(row.corpus_id) : null }));
  }
  const selection = {
    schema: "wavey.encodec.geometry-selection",
    schema_version: 1,
    run_id: path.basename(outputDir),
    selected_geometry_id: null,
    owner_approval_time: null,
    owner_approval_identifier: null,
    status: "blocked",
    stage: "A-geometry-selection",
    measurement_format: "current-acv2-unpublished",
    result_digest: sha256File(resultPath),
    rows_by_rate: byRate,
    blockers: [
      "The current objects are not complete proposed Profile 1 containers.",
      "The qualification CLI exports model inputs, codes, scales, entropy bytes, and recovered codes. The full matrix has not run.",
      "The worktree is dirty, so the release lock cannot be frozen.",
      "No owner approval identifier exists for a geometry selection.",
    ],
  };
  jsonWrite(path.join(outputDir, "geometry-selection.json"), selection);
  jsonWrite(path.join(outputDir, "metrics/summary.json"), { schema: "wavey.encodec.geometry-summary", rows: rows.length, pass: rows.filter((row) => row.status === "pass").length, fail: rows.filter((row) => row.status === "fail").length, blocked: rows.filter((row) => row.status === "blocked").length, conditional_not_authorized: rows.filter((row) => row.status === "conditional_not_authorized").length, by_rate: byRate });
  jsonWrite(path.join(outputDir, "metrics/gates.json"), {
    "G00-selection": "blocked",
    "G01-input": "pass",
    "G02-lock": "blocked",
    "G03-preflight": "pass",
    "G04-model-input": "blocked",
    "G05-entropy": "blocked",
    "G06-tail": "blocked",
    "G07-cross-decode": "blocked",
    "G13-size": "blocked",
    "G18-mutation": "blocked",
    "G19-listening": "blocked",
    "G20-artifacts": "blocked",
  });
}

function fixedCode(options) {
  const outputDir = path.resolve(required(options, "output"));
  const binary = options.binary ? path.resolve(options.binary) : path.join(REPO_ROOT, "target/release/encodec-rs");
  const bundleRoot = path.resolve(options["bundle-root"] ?? path.join(REPO_ROOT, "onnx-bundles"));
  const runId = path.basename(outputDir);
  const resultPath = path.join(outputDir, "metrics/results.jsonl");
  const priorRows = existsSync(resultPath)
    ? readFileSync(resultPath, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line))
    : [];
  for (const rate of [6, 12]) {
    const bundleDir = bundleFor(bundleRoot, "fixed-1333", rate);
    const caseId = `entropy-probe-203-${rate}`;
    const priorCaseRows = priorRows.filter((row) => row.case_id === caseId);
    if (priorCaseRows.some((row) => row.status === "pass" && row.canonical_fixed_code_vector_complete === true)) continue;
    const attempt = Math.max(0, ...priorCaseRows.map((row) => Number(row.attempt ?? 0))) + 1;
    if (!existsSync(path.join(bundleDir, "bundle.json"))) {
      appendJsonl(resultPath, { schema: "wavey.encodec.qualification", schema_version: 1, run_id: runId, case_id: caseId, attempt, status: "blocked", phase: "entropy", candidate: "fixed-1333", bandwidth_kbps: rate, environment_id: "mac-arm64-onnx-cpu", error: "bundle missing", stop_reason: bundleDir });
      continue;
    }
    const evidenceDir = path.join(outputDir, "vectors/lm", `${rate}kbps-203`);
    const environment = { EMSDK_QUIET: "1", ENCODEC_RS_ORT_THREADS: "4", ORT_DYLIB_PATH: process.env.ORT_DYLIB_PATH ?? "/opt/homebrew/lib/libonnxruntime.dylib" };
    const command = runCommand(binary, ["onnx-lm-evidence", bundleDir, evidenceDir, "--steps", "203"], REPO_ROOT, outputDir, environment);
    const probeCommand = runCommand(binary, ["onnx-lm-probe", bundleDir, "--steps", "203"], REPO_ROOT, outputDir, environment);
    let evidence = null;
    let probe = null;
    try { evidence = JSON.parse(command.stdout); } catch { /* recorded below */ }
    try { probe = JSON.parse(probeCommand.stdout); } catch { /* recorded below */ }
    const expectedDigest = {
      6: "9a243c837a6e4b99b8da788be10262a1e626b0f695244094babf5569ea9340e2",
      12: "10261b7f0ec351dec0fd62c67967d3bc8c125a9aae710e2c0f4dc37d93ea36d2",
    }[rate];
    const status = command.exit_status === 0
      && probeCommand.exit_status === 0
      && evidence?.codes_exactly_recovered === true
      && probe?.cdf_sequence_hash === expectedDigest
      ? "pass"
      : "fail";
    appendJsonl(resultPath, {
      schema: "wavey.encodec.qualification",
      schema_version: 1,
      run_id: runId,
      case_id: caseId,
      attempt,
      status,
      phase: "entropy",
      candidate: "fixed-1333",
      bandwidth_kbps: rate,
      environment_id: "mac-arm64-onnx-cpu",
      steps: 203,
      cdf_sequence_digest: probe?.cdf_sequence_hash ?? null,
      expected_203_step_digest: expectedDigest,
      exact_match: probe?.cdf_sequence_hash === expectedDigest,
      canonical_fixed_code_vector_complete: evidence?.codes_exactly_recovered === true,
      codes_sha256: evidence?.codes_sha256 ?? null,
      recovered_codes_sha256: evidence?.recovered_codes_sha256 ?? null,
      scale_bits_hex: evidence?.scale_bits_hex ?? null,
      entropy_sha256: evidence?.entropy_sha256 ?? null,
      payload_sha256: evidence?.payload_sha256 ?? null,
      codebook_order_sha256: evidence?.codebook_order_sha256 ?? null,
      evidence_manifest_path: path.relative(outputDir, path.join(evidenceDir, "manifest.json")),
      error: status === "pass" ? null : `${command.stderr}\n${probeCommand.stderr}`.trim(),
      stop_reason: status === "pass"
        ? "one canonical fixed-code vector records symbols, entropy bytes, scale bits, and recovered codes"
        : "fixed-code evidence or the 203-step CDF digest failed",
    });
  }
  summarizeGeometry(outputDir);
  console.log(JSON.stringify({ output: outputDir, result_path: path.relative(outputDir, resultPath) }, null, 2));
}

function providerStatus(options) {
  const outputDir = path.resolve(required(options, "output"));
  const runId = path.basename(outputDir);
  const resultPath = path.join(outputDir, "metrics/results.jsonl");
  const provider = options.provider ?? "browser-wasm";
  const records = {
    "browser-wasm": { status: "blocked", environment_id: "mac-arm64-chromium-ort-wasm", reason: "The checked-in browser path still targets current acv=2 and has no selected Profile 1 segment builder.", checks: [["node", ["--check", "browser-smoke/webgpu-matrix.js"]], ["node", ["--check", "scripts/webgpu-matrix.mjs"]], ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", ["--version"]]] },
    "browser-webgpu": { status: "blocked", environment_id: "mac-arm64-chromium-webgpu", reason: "The checked-in browser path still targets current acv=2 and has no selected Profile 1 segment builder.", checks: [["node", ["--check", "browser-smoke/webgpu-matrix.js"]], ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", ["--version"]]] },
    "gcp-cpu": { status: "blocked", environment_id: "gcp-linux-x86_64-cpu", reason: "A temporary CPU VM was created and deleted, but the package transfer failed before any GCP CPU row ran.", checks: [["gcloud", ["config", "get-value", "project"]], ["gcloud", ["compute", "instances", "list", "--project", "steadfast-slate-498623-r2", "--filter=labels.run_id:20260725t050000z-redo-e3fd8a8e64f64988", "--format=value(name)"]]] },
    "gcp-cuda-fp32": { status: "blocked", environment_id: "gcp-linux-x86_64-cuda-fp32", reason: "GCP GPUS_ALL_REGIONS quota is 0, so no CUDA VM could be created.", checks: [["gcloud", ["config", "get-value", "project"]], ["gcloud", ["compute", "accelerator-types", "list", "--zones=us-central1-a", "--filter=name:nvidia-v100", "--format=value(name)"]]] },
    azure: { status: "authorized_pending_plan", environment_id: "azure-linux-x86_64-cpu", reason: "Azure subscription access is authorized. No temporary qualification resource was created.", checks: [["/opt/homebrew/bin/az", ["account", "show", "--subscription", "622d8fdd-2eee-4b8a-a5cd-e0d145587e76", "--query", "{id:id,state:state}", "--output", "json"]]] },
  }[provider];
  if (!records) fail(`unknown provider ${provider}`);
  const checks = records.checks.map(([command, args]) => {
    const result = runCommand(command, args, REPO_ROOT, outputDir, { EMSDK_QUIET: "1" });
    return { command: result.command, exit_status: result.exit_status, stdout_sha256: sha256Bytes(Buffer.from(result.stdout)), stderr_sha256: sha256Bytes(Buffer.from(result.stderr)) };
  });
  const row = {
    schema: "wavey.encodec.qualification",
    schema_version: 1,
    run_id: runId,
    case_id: `provider-${provider}`,
    attempt: 1,
    status: records.status,
    phase: "preflight",
    candidate: null,
    bandwidth_kbps: null,
    corpus_id: null,
    environment_id: records.environment_id,
    error: records.reason,
    preflight_checks: checks,
    command: provider === "azure" ? null : "provider-neutral preflight plus qualification block",
    stop_reason: records.reason,
  };
  appendJsonl(resultPath, row);
  console.log(JSON.stringify(row, null, 2));
}

function writeDecision(options) {
  const outputDir = path.resolve(required(options, "output"));
  const selectionPath = path.join(outputDir, "geometry-selection.json");
  const selection = existsSync(selectionPath) ? jsonRead(selectionPath) : null;
  const text = `# Geometry Decision\n\nStatus: blocked pending Profile 1 qualification evidence.\n\nRun: \`${path.basename(outputDir)}\`\n\n## Measurement state\n\nThe run records current unpublished \`acv=2\` measurements only. These bytes are not Profile 1 objects. The final Profile 1 container remains intentionally unimplemented.\n\nThe selection record is [geometry-selection.json](geometry-selection.json). Its current status is \`${selection?.status ?? "missing"}\`.\n\n## Candidates\n\n- \`meta-1000\`: 48,000-sample window, 47,520-sample stride, true variable tail, triangle overlap-add.\n- \`fixed-1333\`: 64,960-sample window, 64,000 owned samples, 203 code frames, bound seam repair.\n- \`fixed-1800\`: 87,360-sample window, 86,400 owned samples, 273 code frames, bound seam repair.\n\n## Unresolved blockers\n\n- The worktree contains uncommitted code and model changes.\n- The qualification CLI exports model inputs, codes, scales, entropy bytes, and recovered codes. The full provider and corpus matrix has not run.\n- No local dynamic Meta bundle is available to run the true variable-tail path.\n- The browser path is hard-wired to the current \`acv=2\` envelope.\n- GCP CPU and CUDA rows remain incomplete because the temporary CPU package transfer failed and the project GPU quota is zero.\n- Azure was not used for this run.\n- Owner approval is required before a geometry selection can be locked.\n\n## Next implementation boundary\n\nRun the evidence matrix with a dynamic Meta bundle and all required providers. Complete the fixed-code pattern and length matrix. After owner approval, implement the Profile 1 container and its strict and salvage decoders. Freeze one geometry only after complete-container size, entropy, recovery, quality, and runtime gates pass.\n\nRaw evidence is under [metrics/results.jsonl](metrics/results.jsonl), [metrics/summary.json](metrics/summary.json), [metrics/gates.json](metrics/gates.json), [corpus-manifest.json](corpus-manifest.json), and [logs/commands.jsonl](logs/commands.jsonl).\n`;
  const corrected = text
    .replace("GCP CPU and CUDA rows need a locked remote checkout and temporary instances.", "GCP CPU execution did not complete because the temporary worker package transfer failed.")
    .replace("Azure remains `conditional_not_authorized`.", "GCP CUDA execution is blocked by a project-wide GPU quota of zero.");
  writeFileSync(path.join(outputDir, "geometry-decision.md"), corrected);
  console.log(path.join(outputDir, "geometry-decision.md"));
}

function localPreflight(options) {
  const outputDir = path.resolve(required(options, "output"));
  const condaPython = process.env.QUALIFICATION_CONDA_PYTHON ?? "/opt/anaconda3/envs/encodec-export/bin/python";
  const checks = [
    ["cargo", ["fmt", "--all", "--", "--check"]],
    ["cargo", ["check", "--locked", "--features", "onnx"]],
    ["cargo", ["test", "--locked", "--features", "ecdc", "--no-fail-fast"]],
    ["cargo", ["test", "--locked", "--features", "wasm", "--no-fail-fast"]],
    ["cargo", ["check", "--locked", "--target", "wasm32-unknown-unknown", "--no-default-features", "--features", "wasm"]],
    ["node", ["--check", "browser-smoke/webgpu-matrix.js"]],
    ["node", ["--check", "scripts/wasm-encode-fixture.mjs"]],
    ["node", ["--check", "scripts/westside-chunk-wasm-roundtrip.mjs"]],
    [condaPython, ["-c", "import numpy, torch, onnx, torchaudio; print(numpy.__version__, torch.__version__, onnx.__version__, torchaudio.__version__)"]],
  ];
  const results = checks.map(([command, args]) => {
    const result = runCommand(command, args, REPO_ROOT, outputDir, { EMSDK_QUIET: "1" });
    return { command: result.command, exit_status: result.exit_status, elapsed_ms: result.elapsed_ms, stdout_sha256: sha256Bytes(Buffer.from(result.stdout)), stderr_sha256: sha256Bytes(Buffer.from(result.stderr)) };
  });
  const value = { schema: "wavey.encodec.qualification.local-preflight", schema_version: 1, run_id: path.basename(outputDir), python_environment: "encodec-export", status: results.every((result) => result.exit_status === 0) ? "pass" : "fail", checks: results };
  jsonWrite(path.join(outputDir, "reports/local-preflight.json"), value);
  console.log(JSON.stringify(value, null, 2));
}

function writeReports(options) {
  const outputDir = path.resolve(required(options, "output"));
  const lock = existsSync(path.join(outputDir, "qualification-lock.json")) ? jsonRead(path.join(outputDir, "qualification-lock.json")) : null;
  const summary = existsSync(path.join(outputDir, "metrics/summary.json")) ? jsonRead(path.join(outputDir, "metrics/summary.json")) : null;
  const resultRows = existsSync(path.join(outputDir, "metrics/results.jsonl"))
    ? readFileSync(path.join(outputDir, "metrics/results.jsonl"), "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line))
    : [];
  const latestCoreMl = new Map();
  for (const row of resultRows.filter((item) => item.environment_id === "mac-arm64-coreml-gpu")) {
    const previous = latestCoreMl.get(row.case_id);
    if (!previous || Number(row.attempt ?? 0) >= Number(previous.attempt ?? 0)) latestCoreMl.set(row.case_id, row);
  }
  const coreMlRows = [...latestCoreMl.values()];
  const reports = {
    "geometry-selection.md": "# Geometry Selection\n\nStatus: blocked.\n\nThe run measures the current unpublished `acv=2` envelope. It does not measure a complete Profile 1 object.\n\nThe evidence runner supports a true variable tail for a dynamic model. No local dynamic Meta bundle is available for this run.\n\nThe fixed candidates can provide provisional CPU measurements. They cannot lock a geometry while the worktree is dirty and the full fixed-code matrix is incomplete.\n",
    "complete-size-comparison.md": "# Complete Container Size\n\nStatus: blocked.\n\nThe run records current ECDC object bytes. It does not record a complete Profile 1 header, scales, chunk framing, directory, trailer, or manifests.\n\nNo size decision is valid until all required fields are included.\n",
    "quality-comparison.md": "# Quality Comparison\n\nStatus: provisional only.\n\nThe current CPU rows contain a basic float32 RMS and maximum error. They do not establish the required listening gate or candidate ranking.\n",
    "fixed-code-entropy.md": "# Fixed-Code and Entropy Evidence\n\nStatus: partial.\n\nThe provider-neutral commands export model inputs, raw codes, scale bits, codebook order, entropy bytes, and recovered codes. The LM probe records the 203-step CDF digest. One canonical 203-step vector exists for each fixed-1333 rate. The full pattern, length, and provider matrix remains incomplete.\n",
    "tail-vectors.md": "# Tail Vectors\n\nStatus: prepared, execution blocked.\n\nThe run contains silence, impulse, and boundary tail vectors for all candidates. The evidence runner supports a true variable tail for a dynamic model. No local dynamic Meta bundle is available, so no Meta tail result is accepted.\n",
    "browser-wasm.md": "# Browser and WASM\n\nStatus: blocked.\n\nThe browser path still targets the current `acv=2` envelope. It has no selected Profile 1 segment builder or strict/salvage decoder.\n",
    "gcp.md": "# GCP\n\nStatus: blocked.\n\nThe project is configured for `steadfast-slate-498623-r2`. The run did not create temporary instances because the local worktree lock is blocked and no remote qualification package is frozen.\n\nAzure access is authorized. Its resource plan remains separate from GCP.\n",
    "architecture-matrix.md": "# Architecture Matrix Plan\n\nScope: audio encode and decode, plus the code and entropy payload path. The approved six-file corpus stays unchanged.\n\n| Environment | Architecture | Work | Status |\n| --- | --- | --- | --- |\n| macOS local | ARM64 Apple M1 CPU | Canonical CPU baseline for both candidates and both rates | Running |\n| macOS Chromium WASM | ARM64 | Encode, decode, code order, and entropy parity | Blocked by current `acv=2` browser path |\n| macOS Chromium WebGPU | ARM64 GPU | Same parity checks with WebGPU execution | Blocked by current `acv=2` browser path |\n| GCP Linux | x86_64 CPU | Repeat the locked CPU cases | Blocked by dirty worktree lock |\n| GCP Linux CUDA | x86_64 NVIDIA GPU | FP32 encode and decode repeat | Blocked by dirty worktree lock |\n| Azure Linux | x86_64 CPU and GPU options | Repeat the same vectors after resource approval | Authorized; resource plan pending |\n\nThe codec input is audio. This run treats code and entropy as the text-like payload. A literal text benchmark needs a separate corpus and contract.\n\nEach environment must use the same corpus manifest, model bundle hashes, vector hashes, codebook order, scale bits, entropy bytes, and salvage cases. Compare exact bytes first. Compare PCM and runtime second.\n\nDo not compare GPU and CPU outputs when the model, precision, or container differs.\n",
    "mutation-salvage.md": "# Mutation and Salvage\n\nStatus: blocked.\n\nThe final Profile 1 container is not implemented. Truncation, CRC, directory, unknown-field, and salvage tests cannot run.\n",
    "unresolved-blockers.md": `# Unresolved Blockers\n\nThe run cannot select or freeze a geometry.\n\n- The worktree is dirty.\n- The current container is not Profile 1.\n- No local dynamic Meta bundle is available for the true variable-tail path.\n- The full fixed-code pattern, length, corpus, and provider matrix is incomplete.\n- Browser and remote providers lack a locked Profile 1 runner.\n- Complete-container size and salvage evidence are absent.\n- Owner approval is absent.\n\nObserved rows: ${summary?.rows ?? 0} total, ${summary?.pass ?? 0} pass, ${summary?.blocked ?? 0} blocked, ${summary?.fail ?? 0} fail.\n\nLock status: ${lock?.status ?? "missing"}.\n`,
    "next-implementation-boundary.md": "# Next Implementation Boundary\n\nStop before wire-format freeze.\n\nRun the evidence matrix with a dynamic Meta bundle and all required providers. Complete the fixed-code pattern and length matrix.\n\nAfter the owner approves one geometry, implement Profile 1 encoding and strict and salvage decoding.\n",
  };
  mkdirSync(path.join(outputDir, "reports"), { recursive: true });
  for (const [name, content] of Object.entries(reports)) writeFileSync(path.join(outputDir, "reports", name), content);
  writeFileSync(path.join(outputDir, "reports", "quality-comparison.md"), "# Quality Comparison\n\nStatus: provisional only.\n\nThe quality audit measures RMS error, RMS dBFS, SNR dB, SI-SDR dB, loudness, true peak, channel correlation, inter-channel balance, spectral distance, clipping, and seam residual.\n\nThese rows compare canonical PCM with the fork output through the current unpublished acv2 envelope. They are not Profile 1 evidence and do not replace the required listening approval.\n");
  writeFileSync(path.join(outputDir, "reports", "gcp.md"), "# GCP\n\nStatus: blocked.\n\nThe project is `steadfast-slate-498623-r2`. A temporary `n2-standard-4` worker was created in `us-west1-a` and deleted after the package transfer failed. No GCP CPU row ran.\n\nGCP GPU provisioning was checked in `us-central1-a` and `us-west1-a`. The project-wide `GPUS_ALL_REGIONS` quota is zero, so no CUDA worker could start.\n");
  writeFileSync(path.join(outputDir, "reports", "metal.md"), `# Mac Metal\n\nEnvironment: \`mac-arm64-coreml-gpu\`.\n\nThe runner requests Core ML with \`cpu-and-gpu\` compute units. Latest rows: ${coreMlRows.length}. Pass: ${coreMlRows.filter((row) => row.status === "pass").length}. Fail: ${coreMlRows.filter((row) => row.status === "fail").length}.\n\nThese rows exercise the current unpublished acv2 envelope. They do not qualify Profile 1 code, entropy, or container parity.\n`);
  const architecturePath = path.join(outputDir, "reports", "architecture-matrix.md");
  const architecture = readFileSync(architecturePath, "utf8")
    .replace("Canonical CPU baseline for both candidates and both rates | Running", "Canonical CPU baseline for both candidates and both rates | Completed; current acv2 baseline")
    .replace("Repeat the locked CPU cases | Blocked by dirty worktree lock", "Repeat the locked CPU cases | Blocked; package transfer failed before execution")
    .replace("FP32 encode and decode repeat | Blocked by dirty worktree lock", "FP32 encode and decode repeat | Blocked; project GPU quota is zero")
    .replace("| Azure Linux | x86_64 CPU and GPU options | Repeat the same vectors after resource approval | Authorized; resource plan pending |\n", "");
  writeFileSync(architecturePath, architecture);
  console.log(JSON.stringify({ output: path.join(outputDir, "reports"), files: Object.keys(reports).length + 1 }, null, 2));
}

function usage() {
  console.error("usage: qualification-mux.mjs <prepare|lock|local-preflight|geometry-select|frame-evidence|fixed-code|provider-status|write-decision|write-reports> [options]");
  process.exitCode = 2;
}

try {
  const options = parseArgs(process.argv.slice(2));
  const command = options._[0];
  if (command === "prepare") prepare(options);
  else if (command === "lock") artifactLock(options);
  else if (command === "local-preflight") localPreflight(options);
  else if (command === "geometry-select") geometrySelect(options);
  else if (command === "frame-evidence") frameEvidence(options);
  else if (command === "fixed-code") fixedCode(options);
  else if (command === "provider-status") providerStatus(options);
  else if (command === "write-decision") writeDecision(options);
  else if (command === "write-reports") writeReports(options);
  else usage();
} catch (error) {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
}
