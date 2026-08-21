#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const options = parseArguments(process.argv.slice(2));
const baseline = readReport(options.baseline);
const candidate = readReport(options.candidate);
const comparison = compareReports(baseline, candidate, options);
const serialized = `${JSON.stringify(comparison, null, 2)}\n`;

if (options.output) {
  mkdirSync(path.dirname(options.output), { recursive: true });
  writeFileSync(options.output, serialized);
}
process.stdout.write(serialized);
if (!comparison.determinism.pass) process.exitCode = 1;

function parseArguments(args) {
  const positional = [];
  let output = null;
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--output") output = path.resolve(args[++index]);
    else positional.push(path.resolve(args[index]));
  }
  if (positional.length !== 2) {
    process.stderr.write(
      "usage: node scripts/compare-encodec-wasm-profiles.mjs "
        + "[--output comparison.json] baseline.json candidate.json\n",
    );
    process.exit(2);
  }
  return { baseline: positional[0], candidate: positional[1], output };
}

function readReport(filename) {
  const report = JSON.parse(readFileSync(filename, "utf8"));
  if (report.schema !== "yl.vin.encodec-wasm-session-profile") {
    throw new Error(`${filename} has unexpected schema ${String(report.schema)}`);
  }
  if (report.status !== "passed") {
    throw new Error(`${filename} did not pass: status=${String(report.status)}`);
  }
  return report;
}

function compareReports(baseline, candidate, options) {
  const baselineRuns = new Map(baseline.runs.map((run) => [run.label, run]));
  const candidateRuns = new Map(candidate.runs.map((run) => [run.label, run]));
  const runLabels = ["isolated-a", "isolated-b", "interleaved-a-b"];
  const runs = {};
  for (const label of runLabels) {
    const before = requiredRun(baselineRuns, label, options.baseline);
    const after = requiredRun(candidateRuns, label, options.candidate);
    runs[label] = compareRun(before, after);
  }

  const hashMismatches = [];
  let comparedChunks = 0;
  let candidateChunks = 0;
  for (const label of ["isolated-a", "isolated-b"]) {
    const before = requiredRun(baselineRuns, label, options.baseline);
    const after = requiredRun(candidateRuns, label, options.candidate);
    const expected = new Map(
      before.chunksDetail.map((chunk) => [chunkKey(chunk), chunk.sha256]),
    );
    const actual = new Map(
      after.chunksDetail.map((chunk) => [chunkKey(chunk), chunk.sha256]),
    );
    comparedChunks += expected.size;
    candidateChunks += actual.size;
    for (const [key, expectedHash] of expected) {
      const candidateHash = actual.get(key);
      if (expectedHash !== candidateHash) {
        const [trackId, chunkIndex] = key.split(":");
        hashMismatches.push({
          run: label,
          trackId,
          chunkIndex: Number(chunkIndex),
          baselineSha256: expectedHash,
          candidateSha256: candidateHash ?? null,
        });
      }
    }
    for (const [key, candidateHash] of actual) {
      if (expected.has(key)) continue;
      const [trackId, chunkIndex] = key.split(":");
      hashMismatches.push({
        run: label,
        trackId,
        chunkIndex: Number(chunkIndex),
        baselineSha256: null,
        candidateSha256: candidateHash,
      });
    }
  }

  const candidateInternal = candidate.determinism ?? {};
  const internalPass = candidateInternal.pass === true
    && candidateInternal.mismatchedChunks === 0
    && candidateInternal.comparedChunks === candidateChunks;
  return {
    schema: "encodec-rs.entropy-optimization-comparison",
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    baseline: options.baseline,
    candidate: options.candidate,
    configuration: candidate.configuration,
    sources: candidate.sources,
    determinism: {
      pass: hashMismatches.length === 0 && internalPass,
      baselineComparedChunks: comparedChunks,
      candidateComparedChunks: candidateChunks,
      baselineMismatchedChunks: hashMismatches.length,
      candidateInterleavePass: internalPass,
      candidateInterleaveComparedChunks: candidateInternal.comparedChunks ?? 0,
      candidateInterleaveMismatchedChunks: candidateInternal.mismatchedChunks ?? null,
      mismatches: hashMismatches,
    },
    runs,
  };
}

function requiredRun(runs, label, filename) {
  const run = runs.get(label);
  if (!run) throw new Error(`${filename} has no ${label} run`);
  return run;
}

function compareRun(baseline, candidate) {
  const before = summarizeRun(baseline);
  const after = summarizeRun(candidate);
  return {
    baseline: before,
    candidate: after,
    deltaPercent: {
      sessionCreateMs: percentDelta(before.sessionCreateMs, after.sessionCreateMs),
      onnxMs: percentDelta(before.timings.onnxMs, after.timings.onnxMs),
      lmWasmMs: percentDelta(before.timings.lmWasmMs, after.timings.lmWasmMs),
      totalEncodeMs: percentDelta(before.timings.totalEncodeMs, after.timings.totalEncodeMs),
      realtimeFactor: percentDelta(before.realtimeFactor, after.realtimeFactor),
      warmRealtimeFactor: percentDelta(before.warmRealtimeFactor, after.warmRealtimeFactor),
      medianOnnxMs: percentDelta(before.medianMs.onnx, after.medianMs.onnx),
      medianLmWasmMs: percentDelta(before.medianMs.lmWasm, after.medianMs.lmWasm),
      medianTotalMs: percentDelta(before.medianMs.total, after.medianMs.total),
      p90TotalMs: percentDelta(before.p90Ms.total, after.p90Ms.total),
    },
  };
}

function summarizeRun(run) {
  return {
    label: run.label,
    sessionCreateMs: run.sessionCreateMs,
    chunks: run.chunks,
    audioSeconds: run.audioSeconds,
    timings: run.timings,
    realtimeFactor: run.realtimeFactor,
    warmRealtimeFactor: run.warmRealtimeFactor,
    passesTwoTimesRealtimeWarmGate: run.passesTwoTimesRealtimeWarmGate,
    medianMs: distribution(run.chunksDetail, 0.5),
    p90Ms: distribution(run.chunksDetail, 0.9),
  };
}

function distribution(chunks, quantile) {
  return {
    onnx: round(selectQuantile(chunks.map((chunk) => chunk.onnxMs), quantile)),
    lmWasm: round(selectQuantile(chunks.map((chunk) => chunk.lmWasmMs), quantile)),
    total: round(selectQuantile(chunks.map((chunk) => chunk.totalMs), quantile)),
  };
}

function selectQuantile(values, quantile) {
  const sorted = values.map(Number).sort((left, right) => left - right);
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

function percentDelta(before, after) {
  if (before === 0) return null;
  return round((after / before - 1) * 100);
}

function chunkKey(chunk) {
  return `${chunk.trackId}:${chunk.chunkIndex}`;
}

function round(value, digits = 3) {
  const scale = 10 ** digits;
  return Math.round(Number(value) * scale) / scale;
}
