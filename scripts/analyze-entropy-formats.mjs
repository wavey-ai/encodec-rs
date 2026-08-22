#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";

if (process.argv.length !== 5) {
  throw new Error(
    "usage: analyze-entropy-formats.mjs <decoder-input-dir> <reference.ecdc> <output.json>",
  );
}

const inputDir = path.resolve(process.argv[2]);
const referencePath = path.resolve(process.argv[3]);
const outputPath = path.resolve(process.argv[4]);
const metadata = JSON.parse(fs.readFileSync(path.join(inputDir, "metadata.json"), "utf8"));
const codebooks = metadata.numCodebooks;
const frameLength = metadata.frameLength;
const frames = metadata.frames.map((frame) => {
  const bytes = fs.readFileSync(path.join(inputDir, frame.file));
  return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
});
const symbolCount = frames.length * codebooks * frameLength;

const layouts = new Map();
layouts.set("chunk-time-codebook", collectChunkTimeCodebook());
layouts.set("chunk-codebook-time", collectChunkCodebookTime());
layouts.set("codebook-chunk-time", collectCodebookChunkTime());
layouts.set("codebook-temporal-xor", temporalTransform((value, previous) => value ^ previous));
layouts.set(
  "codebook-temporal-delta",
  temporalTransform((value, previous) => (value - previous) & 1023),
);

const representations = new Map();
for (const [layout, values] of layouts) {
  representations.set(`${layout}:packed10`, packBits(values, 10));
  representations.set(`${layout}:split8+2`, splitLowHigh(values));
  representations.set(`${layout}:bitplanes`, bitplanes(values, 10));
}

const codecs = [
  { name: "zstd-1", command: "zstd", args: ["-q", "-c", "-1"] },
  { name: "zstd-3", command: "zstd", args: ["-q", "-c", "-3"] },
  { name: "zstd-9", command: "zstd", args: ["-q", "-c", "-9"] },
  { name: "zstd-19", command: "zstd", args: ["-q", "-c", "-19"] },
  { name: "brotli-1", command: "brotli", args: ["-q", "1", "-c"] },
  { name: "brotli-4", command: "brotli", args: ["-q", "4", "-c"] },
  { name: "brotli-9", command: "brotli", args: ["-q", "9", "-c"] },
  { name: "brotli-11", command: "brotli", args: ["-q", "11", "-c"] },
  { name: "gzip-1", command: "gzip", args: ["-1", "-c"] },
  { name: "gzip-9", command: "gzip", args: ["-9", "-c"] },
  { name: "bzip2-1", command: "bzip2", args: ["-1", "-c"] },
  { name: "bzip2-9", command: "bzip2", args: ["-9", "-c"] },
  { name: "xz-1", command: "xz", args: ["-1", "-c"] },
  { name: "xz-6", command: "xz", args: ["-6", "-c"] },
];

const results = [];
for (const [representation, bytes] of representations) {
  for (const codec of codecs) {
    const started = performance.now();
    const result = spawnSync(codec.command, codec.args, {
      input: bytes,
      maxBuffer: 16 * 1024 * 1024,
    });
    if (result.status !== 0) {
      throw new Error(
        `${codec.name} failed: ${result.stderr?.toString("utf8") ?? `status ${result.status}`}`,
      );
    }
    results.push({
      representation,
      codec: codec.name,
      inputBytes: bytes.length,
      outputBytes: result.stdout.length,
      encodeMs: performance.now() - started,
    });
  }
}
results.sort((left, right) => left.outputBytes - right.outputBytes);

const report = {
  schema: "encodec-rs.entropy-format-analysis",
  inputDir,
  referencePath,
  referenceBytes: fs.statSync(referencePath).size,
  frames: frames.length,
  codebooks,
  frameLength,
  symbolCount,
  rawPacked10Bytes: Math.ceil((symbolCount * 10) / 8),
  best: results.slice(0, 30),
  results,
};
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ ...report, results: undefined }, null, 2));

function collectChunkTimeCodebook() {
  const values = new Uint16Array(symbolCount);
  let output = 0;
  for (const frame of frames) {
    for (let step = 0; step < frameLength; ++step) {
      for (let codebook = 0; codebook < codebooks; ++codebook) {
        values[output++] = frame[codebook * frameLength + step];
      }
    }
  }
  return values;
}

function collectChunkCodebookTime() {
  const values = new Uint16Array(symbolCount);
  let output = 0;
  for (const frame of frames) {
    values.set(frame, output);
    output += frame.length;
  }
  return values;
}

function collectCodebookChunkTime() {
  const values = new Uint16Array(symbolCount);
  let output = 0;
  for (let codebook = 0; codebook < codebooks; ++codebook) {
    for (const frame of frames) {
      const start = codebook * frameLength;
      values.set(frame.subarray(start, start + frameLength), output);
      output += frameLength;
    }
  }
  return values;
}

function temporalTransform(transform) {
  const values = new Uint16Array(symbolCount);
  let output = 0;
  for (let codebook = 0; codebook < codebooks; ++codebook) {
    let previous = 0;
    for (const frame of frames) {
      const start = codebook * frameLength;
      for (let step = 0; step < frameLength; ++step) {
        const value = frame[start + step];
        values[output++] = transform(value, previous);
        previous = value;
      }
    }
  }
  return values;
}

function splitLowHigh(values) {
  const low = Buffer.allocUnsafe(values.length);
  const high = new Uint16Array(values.length);
  for (let index = 0; index < values.length; ++index) {
    low[index] = values[index] & 255;
    high[index] = values[index] >>> 8;
  }
  return Buffer.concat([low, packBits(high, 2)]);
}

function bitplanes(values, bits) {
  const planes = [];
  for (let bit = 0; bit < bits; ++bit) {
    const plane = new Uint16Array(values.length);
    for (let index = 0; index < values.length; ++index) {
      plane[index] = (values[index] >>> bit) & 1;
    }
    planes.push(packBits(plane, 1));
  }
  return Buffer.concat(planes);
}

function packBits(values, bits) {
  const output = Buffer.alloc(Math.ceil((values.length * bits) / 8));
  let accumulator = 0;
  let accumulatorBits = 0;
  let outputOffset = 0;
  for (const value of values) {
    accumulator |= value << accumulatorBits;
    accumulatorBits += bits;
    while (accumulatorBits >= 8) {
      output[outputOffset++] = accumulator & 255;
      accumulator >>>= 8;
      accumulatorBits -= 8;
    }
  }
  if (accumulatorBits > 0) output[outputOffset] = accumulator & 255;
  return output;
}
