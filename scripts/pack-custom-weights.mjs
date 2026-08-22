#!/usr/bin/env node

import {
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";

if (process.argv.length !== 4) {
  process.stderr.write(
    "usage: node scripts/pack-custom-weights.mjs <source-dir> <output-dir>\n",
  );
  process.exit(2);
}

const source = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
const names = readdirSync(source)
  .filter((name) => name.endsWith(".f32le"))
  .sort();
if (names.length === 0) {
  throw new Error(`no float32 weights found in ${source}`);
}

const alignment = 16;
const tensors = {};
let byteLength = 0;
for (const name of names) {
  byteLength = align(byteLength, alignment);
  const bytes = readFileSync(path.join(source, name));
  if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`${name} is not a float32 asset`);
  }
  tensors[name] = {
    offsetBytes: byteLength,
    byteLength: bytes.byteLength,
    length: bytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
  };
  byteLength += bytes.byteLength;
}

const packed = Buffer.alloc(byteLength);
for (const name of names) {
  readFileSync(path.join(source, name)).copy(packed, tensors[name].offsetBytes);
}
const hash = createHash("sha256").update(packed).digest("hex");
const manifest = {
  schemaVersion: 1,
  file: "weights.f32le",
  dataType: "float32",
  byteOrder: "little",
  alignment,
  byteLength: packed.byteLength,
  sha256: hash,
  tensors,
};

mkdirSync(output, { recursive: true });
const metadata = JSON.parse(readFileSync(path.join(source, "metadata.json"), "utf8"));
delete metadata.sourceModel;
metadata.onnxFree = true;
writeFileSync(
  path.join(output, "metadata.json"),
  `${JSON.stringify(metadata, null, 2)}\n`,
);
writeFileSync(path.join(output, manifest.file), packed);
writeFileSync(
  path.join(output, "weights.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
console.log(
  JSON.stringify({ output, tensors: names.length, byteLength, sha256: hash }),
);

function align(value, multiple) {
  return Math.ceil(value / multiple) * multiple;
}
