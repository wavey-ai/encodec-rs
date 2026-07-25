#!/usr/bin/env node

import crypto from "node:crypto";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { acquireQualificationRunLock } from "./run-lock.mjs";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "../..");
await acquireQualificationRunLock("export-results");
const sourceRoot = path.resolve(process.argv[2] ?? "");
const destinationRoot = path.resolve(
  process.argv[3] ?? path.join(repoRoot, "qualification-results", path.basename(sourceRoot)),
);

function fail(message) {
  throw new Error(message);
}

function isWithin(parent, child) {
  return child === parent || child.startsWith(`${parent}${path.sep}`);
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function walk(directory) {
  if (!existsSync(directory)) return [];
  const files = [];
  for (const name of readdirSync(directory)) {
    const entry = path.join(directory, name);
    if (statSync(entry).isDirectory()) files.push(...walk(entry));
    else files.push(entry);
  }
  return files;
}

function copy(relativePath) {
  const source = path.join(sourceRoot, relativePath);
  if (!existsSync(source)) return;
  const destination = path.join(destinationRoot, relativePath);
  mkdirSync(path.dirname(destination), { recursive: true });
  copyFileSync(source, destination);
}

function copyMatching(relativeDirectory, predicate) {
  const directory = path.join(sourceRoot, relativeDirectory);
  for (const source of walk(directory)) {
    const relativePath = path.relative(sourceRoot, source);
    if (predicate(relativePath, source)) copy(relativePath);
  }
}

function parseRows() {
  const resultPath = path.join(sourceRoot, "metrics/results.jsonl");
  return readFileSync(resultPath, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function latestRows(rows) {
  const latest = new Map();
  for (const row of rows) {
    const previous = latest.get(row.case_id);
    if (!previous || Number(row.attempt ?? 0) >= Number(previous.attempt ?? 0)) {
      latest.set(row.case_id, row);
    }
  }
  return [...latest.values()];
}

function countBy(rows, field) {
  const counts = {};
  for (const row of rows) {
    const value = String(row[field] ?? "unspecified");
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
}

function environmentSummary(rows) {
  const environments = {};
  for (const row of rows) {
    const environment = row.environment_id ?? "unspecified";
    const status = row.status ?? "unspecified";
    environments[environment] ??= { rows: 0, statuses: {} };
    environments[environment].rows += 1;
    environments[environment].statuses[status] =
      (environments[environment].statuses[status] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(environments).sort(([left], [right]) => left.localeCompare(right)));
}

function writeSummary(rows) {
  const latest = latestRows(rows);
  const directQuality = latest.filter((row) =>
    String(row.case_id).startsWith("quality-original-vs-fork-"));
  const summary = {
    schema: "wavey.encodec.qualification-results-snapshot",
    schema_version: 1,
    run_id: path.basename(sourceRoot),
    generated_at: new Date().toISOString(),
    recorded_rows: rows.length,
    latest_case_rows: latest.length,
    latest_statuses: countBy(latest, "status"),
    latest_environments: environmentSummary(latest),
    direct_meta_fork_quality: {
      rows: directQuality.length,
      statuses: countBy(directQuality, "status"),
    },
    gcp: {
      cpu_codec_rows_completed: 0,
      cuda_codec_rows_completed: 0,
      status: "blocked",
      reasons: [
        "The CPU package transfer failed before codec execution.",
        "The project-wide GPU quota was zero.",
      ],
    },
    scope: {
      includes: "Result records, reports, manifests, environment records, and command logs.",
      excludes: "Source audio, decoded audio, model files, compiled caches, and other large binary artifacts.",
    },
  };
  writeFileSync(
    path.join(destinationRoot, "latest-summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
  );
  return summary;
}

function writeReadme(summary) {
  const statuses = Object.entries(summary.latest_statuses)
    .map(([status, count]) => `- \`${status}\`: ${count}`)
    .join("\n");
  const text = `# Qualification results

This directory preserves the result data for run \`${summary.run_id}\`.

The snapshot contains ${summary.recorded_rows} recorded rows.
It contains ${summary.latest_case_rows} latest case rows.

## Latest status

${statuses}

## Quality evidence

The run contains ${summary.direct_meta_fork_quality.rows} direct Meta-versus-fork quality comparisons.
These rows include SNR, SI-SDR, RMS, peak, channel, clipping, and seam measurements.
The recorded environment for these rows is local macOS ARM64.

## GCP evidence

No GCP codec row completed.
The CPU package transfer failed before codec execution.
The project-wide GPU quota was zero.

## Snapshot scope

The snapshot includes result records, reports, manifests, environment records, and command logs.
It excludes source audio, decoded audio, model files, compiled caches, and other large binary artifacts.

The excluded artifacts remain in the ignored \`target/qualification\` run directory.
Do not use this run as Profile 1 release evidence.
`;
  writeFileSync(path.join(destinationRoot, "README.md"), text);
}

function writeFileManifest() {
  const files = walk(destinationRoot)
    .filter((filePath) => path.basename(filePath) !== "files.sha256")
    .sort();
  const lines = files.map((filePath) =>
    `${sha256File(filePath)}  ${path.relative(destinationRoot, filePath)}`);
  writeFileSync(path.join(destinationRoot, "files.sha256"), `${lines.join("\n")}\n`);
}

const qualificationRoot = path.join(repoRoot, "target", "qualification");
if (!isWithin(qualificationRoot, sourceRoot) || sourceRoot === qualificationRoot) {
  fail("the source must be one run below target/qualification");
}
if (!existsSync(path.join(sourceRoot, "metrics/results.jsonl"))) {
  fail("the source run has no metrics/results.jsonl file");
}
if (!isWithin(repoRoot, destinationRoot) || destinationRoot === repoRoot) {
  fail("the destination must be below the repository root");
}
if (existsSync(destinationRoot)) {
  fail(`the immutable destination already exists: ${destinationRoot}`);
}

mkdirSync(destinationRoot, { recursive: true });
for (const relativePath of [
  "corpus-manifest.json",
  "geometry-decision.md",
  "geometry-selection.json",
  "qualification-lock.json",
  "metrics/gates.json",
  "metrics/results.jsonl",
  "metrics/summary.json",
  "vectors/vector-manifest.json",
]) {
  copy(relativePath);
}
copyMatching("environments", (relativePath) => relativePath.endsWith(".json"));
copyMatching("manifests", (relativePath) =>
  relativePath.endsWith(".json") || relativePath.endsWith(".sha256"));
copyMatching("reports", (relativePath) =>
  relativePath.endsWith(".json") || relativePath.endsWith(".md"));
copyMatching("logs", (relativePath) => relativePath.endsWith(".jsonl"));
copyMatching("metrics/quality", (relativePath) => relativePath.endsWith(".json"));
copyMatching("metrics/size-ledger", (relativePath) => relativePath.endsWith(".json"));
copyMatching("metrics/original-fork", (relativePath, source) =>
  relativePath.endsWith(".json") && statSync(source).size <= 1024 * 1024);
copyMatching("metrics/frame-evidence", (relativePath) =>
  path.basename(relativePath) === "manifest.json");
copyMatching("evidence", (relativePath) =>
  path.basename(relativePath) === "manifest.json");

const summary = writeSummary(parseRows());
writeReadme(summary);
writeFileManifest();

console.log(JSON.stringify({
  destination: path.relative(repoRoot, destinationRoot),
  files: walk(destinationRoot).length,
  bytes: walk(destinationRoot).reduce((total, filePath) => total + statSync(filePath).size, 0),
}, null, 2));
