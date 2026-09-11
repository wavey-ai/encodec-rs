#!/usr/bin/env node
// What the seven-codebook profile actually costs, across genres.
//
// The recorded bitrate figures for this profile were one track long: the
// browser matrix measured PRAY 4 ME and nothing else, and the eight-track
// Lori study that looks like a corpus is the *eight*-codebook bundle. So
// every capacity estimate downstream — which record format a programme is
// offered, how much groove a side needs — rested on one song by one artist.
//
// This runs soundkit's `flac-packet-bench/diverse-v1` corpus through the
// real seven-codebook encoder: four groups of ten clips, drawn from four
// records that do not sound alike. Jazz trio, eighties studio pop, modern
// R&B and a film score cover a useful spread of density, and the point of
// the spread is that the minimum and the maximum are as interesting as the
// mean — a capacity number has to hold for the loudest clip, not the
// average one.
//
//   node scripts/corpus-7cb-bitrate.mjs [--corpus DIR] [--out DIR]
//
// Clips are cut to a whole number of encoder chunks so nothing is measuring
// the encoder's own zero padding: the profile strides 64,000 samples, so
// seven chunks is 448,000 samples, 9.3333 s at 48 kHz. The corpus slots are
// 10 s, so each clip is the head of its slot.

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));

const SAMPLE_RATE = 48_000;
const CHANNELS = 2;
/// The profile's own stride, from `encoder/metadata.json`.
const CHUNK_FRAMES = 64_000;
const CHUNKS_PER_CLIP = 7;
const CLIP_FRAMES = CHUNK_FRAMES * CHUNKS_PER_CLIP;
const SLOT_SECONDS = 10;
const SLOTS = 10;

/// The corpus is 24-bit audio kept right-justified in 32-bit words, so a
/// sample is its own value over 2^23 rather than over 2^31. Reading it the
/// other way is 48 dB of attenuation, and quiet audio codes small — it
/// would have produced a plausible, badly wrong answer.
const S24_FULL_SCALE = 8_388_608;

/// The ECDC container's fixed costs, from
/// `docs/browser-backend-parity-and-bitrate-matrix.md`: a 96-byte header
/// once, and 8 bytes of framing per chunk.
const HEADER_BYTES = 96;
const FRAMING_BYTES = 8;

const GROUPS = [
    { id: "bill-evans-secret-sessions", title: "Bill Evans, The Secret Sessions", genre: "jazz trio, live" },
    { id: "blue-nile-hats", title: "The Blue Nile, Hats", genre: "studio pop, 1989" },
    { id: "lori-asha", title: "Lori Asha", genre: "contemporary R&B" },
    { id: "nocturnal-animals", title: "Nocturnal Animals", genre: "orchestral score" },
];

const options = parseArgs(process.argv.slice(2));
const bundle = path.join(
    repoRoot, "dist/wasm-fixed-bundles/bundles/encodec_48khz_12kbps_7cb_1333ms",
);

mkdirSync(options.out, { recursive: true });
const work = path.join(options.out, "clips");
mkdirSync(work, { recursive: true });

const rows = [];
for (const group of GROUPS) {
    const source = path.join(options.corpus, `${group.id}-48k-s24.s32le`);
    const pcm = readFileSync(source);
    for (let slot = 0; slot < SLOTS; slot += 1) {
        const wav = path.join(work, `${group.id}-${String(slot + 1).padStart(2, "0")}.wav`);
        writeFileSync(wav, clipToWav(pcm, slot));
        const report = path.join(work, `${group.id}-${String(slot + 1).padStart(2, "0")}.json`);
        encode(wav, path.join(work, "out.ecdc"), report);
        const r = JSON.parse(readFileSync(report, "utf8"));
        if (r.ecdcMetadata?.nc !== 7) {
            throw new Error(`${wav}: encoded ${r.ecdcMetadata?.nc} codebooks, expected 7`);
        }
        rows.push({
            group: group.id,
            slot: slot + 1,
            seconds: r.audioSeconds,
            segments: r.segments,
            ecdcBytes: r.ecdcBytes,
            // The header is written once per record and the framing once per
            // chunk; a groove carries the chunks. So the number that drives a
            // capacity estimate is the payload, and the whole-file figure is
            // kept beside it because that is what a file on disk measures.
            payloadBytes: r.ecdcBytes - HEADER_BYTES - r.segments * FRAMING_BYTES,
            rmsDbfs: r.sourceSignal?.rmsDbfs ?? null,
            peakDbfs: r.sourceSignal?.peakDbfs ?? null,
        });
        process.stdout.write(
            `${group.id} ${slot + 1}/${SLOTS}  ${r.ecdcBytes} B  ${kbps(r.ecdcBytes, r.audioSeconds).toFixed(3)} kbps\n`,
        );
    }
}
rmSync(path.join(work, "out.ecdc"), { force: true });

const summary = {
    schema: "yl.vin.encodec-7cb-corpus-bitrate",
    measuredAt: new Date().toISOString(),
    bundle: path.relative(repoRoot, bundle),
    corpus: options.corpus,
    codebooks: 7,
    clipSeconds: CLIP_FRAMES / SAMPLE_RATE,
    chunksPerClip: CHUNKS_PER_CLIP,
    clips: rows.length,
    groups: Object.fromEntries(GROUPS.map((g) => [
        g.id, stats(rows.filter((r) => r.group === g.id)),
    ])),
    overall: stats(rows),
    rows,
};
writeFileSync(
    path.join(options.out, "corpus-7cb-bitrate.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
);
writeFileSync(path.join(options.out, "corpus-7cb-bitrate.md"), markdown(summary));
process.stdout.write(`\n${markdown(summary)}`);

function kbps(bytes, seconds) { return (bytes * 8) / seconds / 1000; }
function kBps(bytes, seconds) { return bytes / seconds / 1000; }

function stats(list) {
    const rates = list.map((r) => kbps(r.payloadBytes, r.seconds)).sort((a, b) => a - b);
    const mean = rates.reduce((t, v) => t + v, 0) / rates.length;
    const sd = Math.sqrt(rates.reduce((t, v) => t + (v - mean) ** 2, 0) / rates.length);
    const whole = list.map((r) => kbps(r.ecdcBytes, r.seconds));
    return {
        clips: list.length,
        payloadKbps: {
            min: rates[0],
            max: rates[rates.length - 1],
            mean,
            median: rates[Math.floor(rates.length / 2)],
            stdDev: sd,
        },
        payloadKBytesPerSecond: {
            min: rates[0] / 8,
            max: rates[rates.length - 1] / 8,
            mean: mean / 8,
        },
        wholeFileKbps: {
            min: Math.min(...whole),
            max: Math.max(...whole),
            mean: whole.reduce((t, v) => t + v, 0) / whole.length,
        },
    };
}

/// One slot's head, as a 32-bit float WAV.
///
/// Float rather than one of the integer widths because the reader in
/// `wasm-encode-fixture.mjs` takes float32 without a scale decision of its
/// own — the justification question above is settled here, once, where the
/// corpus's own format is known.
function clipToWav(pcm, slot) {
    const startFrame = slot * SLOT_SECONDS * SAMPLE_RATE;
    const samples = new Int32Array(
        pcm.buffer, pcm.byteOffset + startFrame * CHANNELS * 4, CLIP_FRAMES * CHANNELS,
    );
    const dataBytes = samples.length * 4;
    const out = Buffer.alloc(44 + dataBytes);
    out.write("RIFF", 0, "ascii");
    out.writeUInt32LE(36 + dataBytes, 4);
    out.write("WAVE", 8, "ascii");
    out.write("fmt ", 12, "ascii");
    out.writeUInt32LE(16, 16);
    out.writeUInt16LE(3, 20);                                   // IEEE float
    out.writeUInt16LE(CHANNELS, 22);
    out.writeUInt32LE(SAMPLE_RATE, 24);
    out.writeUInt32LE(SAMPLE_RATE * CHANNELS * 4, 28);
    out.writeUInt16LE(CHANNELS * 4, 32);
    out.writeUInt16LE(32, 34);
    out.write("data", 36, "ascii");
    out.writeUInt32LE(dataBytes, 40);
    for (let i = 0; i < samples.length; i += 1) {
        out.writeFloatLE(samples[i] / S24_FULL_SCALE, 44 + i * 4);
    }
    return out;
}

function encode(wav, outEcdc, report) {
    execFileSync("node", [
        path.join(repoRoot, "scripts/wasm-encode-fixture.mjs"), "encode",
        "--encodec-wasm-root", path.resolve(repoRoot, ".."),
        "--bundle", bundle,
        "--custom-encoder-root", path.join(bundle, "encoder"),
        "--custom-encoder-kernel-module", path.join(bundle, "encoder/encodec-encoder.mjs"),
        wav,
        "--output", outEcdc,
        "--report", report,
    ], { cwd: repoRoot, stdio: ["ignore", "ignore", "inherit"] });
}

function markdown(s) {
    const f = (v, d = 3) => v.toFixed(d);
    const lines = [];
    lines.push("| Group | Clips | Min | Mean | Max | Range | SD |");
    lines.push("|---|---:|---:|---:|---:|---:|---:|");
    for (const g of GROUPS) {
        const k = s.groups[g.id].payloadKbps;
        lines.push(
            `| ${g.title} | ${s.groups[g.id].clips} | ${f(k.min)} | ${f(k.mean)} `
            + `| ${f(k.max)} | ${f(k.max - k.min)} | ${f(k.stdDev)} |`,
        );
    }
    const o = s.overall.payloadKbps;
    lines.push(
        `| **All four** | **${s.overall.clips}** | **${f(o.min)}** | **${f(o.mean)}** `
        + `| **${f(o.max)}** | **${f(o.max - o.min)}** | **${f(o.stdDev)}** |`,
    );
    return `${lines.join("\n")}\n`;
}

function parseArgs(args) {
    const out = {
        corpus: path.resolve(repoRoot, "../soundkit/testdata/flac-packet-bench/diverse-v1"),
        out: path.join(repoRoot, "target/corpus-7cb-bitrate"),
    };
    for (let i = 0; i < args.length; i += 1) {
        if (args[i] === "--corpus") out.corpus = path.resolve(args[++i]);
        else if (args[i] === "--out") out.out = path.resolve(args[++i]);
    }
    return out;
}
