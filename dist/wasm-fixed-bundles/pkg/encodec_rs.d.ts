/* tslint:disable */
/* eslint-disable */

export class QuantizedLmChunkDecoder {
    free(): void;
    [Symbol.dispose](): void;
    bitstream_version(): number;
    lmWindowFrameLength(): number;
    constructor(bundle_json: string, weights: Uint8Array, payload: Uint8Array);
    pull(): Uint16Array;
    scale(): number;
}

export class QuantizedLmChunkEncoder {
    free(): void;
    [Symbol.dispose](): void;
    bitstream_version(): number;
    finish(): Uint8Array;
    finishPadded(frame_length: number): Uint8Array;
    lmWindowFrameLength(): number;
    constructor(bundle_json: string, weights: Uint8Array, scale: number);
    push(codes: Uint16Array): void;
}

export function bundleMetadata(bundle_json: string): any;

export function ecdcEncodeModelInput(bundle_json: string, audio: Float32Array): Float32Array;

export function ecdcMetadata(payload: Uint8Array): any;

export function ecdcOverlapAdd(bundle_json: string, audio_length: number, decoded_frames: Float32Array): Float32Array;

export function ecdcOverlapAddForMetadata(bundle_json: string, metadata_json: string, decoded_frames: Float32Array): Float32Array;

export function fixedEcdcBundleName(bandwidth_kbps: any, chunk_ms: any): string;

export function initPanicHook(): void;

export function lmEcdcChunk(payload: Uint8Array): Uint8Array;

export function lmEcdcChunkFromFrame(bundle_json: string, weights: Uint8Array, scale: number, codes: Uint16Array, frame_length: number): Uint8Array;

export function lmEcdcDecodeChunks(bundle_json: string, payload: Uint8Array): any;

export function lmEcdcFixedHeaderForWeights(bundle_json: string, audio_length: number, bitstream_version: number, weights: Uint8Array): Uint8Array;

export function lmEcdcHeaderForWeights(bundle_json: string, audio_length: number, bitstream_version: number, weights: Uint8Array): Uint8Array;

export function stableHashHex(bytes: Uint8Array): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly fixedEcdcBundleName: (a: any, b: any) => [number, number, number, number];
    readonly bundleMetadata: (a: number, b: number) => [number, number, number];
    readonly stableHashHex: (a: number, b: number) => [number, number];
    readonly ecdcMetadata: (a: number, b: number) => [number, number, number];
    readonly ecdcOverlapAdd: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly ecdcOverlapAddForMetadata: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly lmEcdcHeaderForWeights: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly lmEcdcFixedHeaderForWeights: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly lmEcdcChunk: (a: number, b: number) => [number, number, number, number];
    readonly ecdcEncodeModelInput: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly lmEcdcChunkFromFrame: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number, number];
    readonly lmEcdcDecodeChunks: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly __wbg_quantizedlmchunkencoder_free: (a: number, b: number) => void;
    readonly quantizedlmchunkencoder_new: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly quantizedlmchunkencoder_lmWindowFrameLength: (a: number) => number;
    readonly quantizedlmchunkencoder_push: (a: number, b: number, c: number) => [number, number];
    readonly quantizedlmchunkencoder_finishPadded: (a: number, b: number) => [number, number, number, number];
    readonly quantizedlmchunkencoder_finish: (a: number) => [number, number];
    readonly __wbg_quantizedlmchunkdecoder_free: (a: number, b: number) => void;
    readonly quantizedlmchunkdecoder_new: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly quantizedlmchunkdecoder_bitstream_version: (a: number) => number;
    readonly quantizedlmchunkdecoder_lmWindowFrameLength: (a: number) => number;
    readonly quantizedlmchunkdecoder_scale: (a: number) => number;
    readonly quantizedlmchunkdecoder_pull: (a: number) => [number, number, number, number];
    readonly initPanicHook: () => void;
    readonly quantizedlmchunkencoder_bitstream_version: (a: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
