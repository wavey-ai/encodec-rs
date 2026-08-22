import { createWebGpuDecoder } from "./webgpu-decoder-runtime.js";

function requiredFunction(value, name) {
  if (typeof value !== "function") {
    throw new Error(`WebGPU ECDC decoder requires ${name}`);
  }
  return value;
}

function normalizeBytes(value, name) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (Array.isArray(value)) return Uint8Array.from(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error(`${name} must be an array buffer-like value`);
}

function audioLengthFromMetadata(metadata) {
  const value = Number(metadata.al ?? metadata.audio_length);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error("ECDC metadata has an invalid audio length");
  }
  return value;
}

function decodeLmChunk(module, bundleJson, weights, metadata, chunk) {
  const decoder = new module.QuantizedLmChunkDecoder(
    bundleJson,
    weights,
    normalizeBytes(chunk.payload, "ECDC chunk payload"),
  );
  try {
    const codes = new Uint16Array(metadata.num_codebooks * metadata.frame_length);
    if (typeof decoder.pullAll === "function") {
      const pulled = decoder.pullAll(chunk.frameLength);
      const activeCodes = pulled instanceof Uint16Array
        ? pulled
        : Uint16Array.from(pulled);
      for (let codebook = 0; codebook < metadata.num_codebooks; codebook += 1) {
        const sourceOffset = codebook * chunk.frameLength;
        const targetOffset = codebook * metadata.frame_length;
        codes.set(
          activeCodes.subarray(sourceOffset, sourceOffset + chunk.frameLength),
          targetOffset,
        );
      }
    } else {
      for (let step = 0; step < chunk.frameLength; step += 1) {
        const symbols = decoder.pull();
        for (let codebook = 0; codebook < metadata.num_codebooks; codebook += 1) {
          codes[codebook * metadata.frame_length + step] = symbols[codebook];
        }
      }
    }
    return {
      offset: chunk.offset,
      samples: chunk.samples,
      frameLength: chunk.frameLength,
      scale: decoder.scale(),
      codes,
    };
  } finally {
    decoder.free();
  }
}

function cropOwnedPcm(modelWindow, metadata, chunk) {
  const privateSamples = metadata.segment_samples - metadata.segment_stride;
  if (privateSamples < 0 || privateSamples % 2 !== 0) {
    throw new Error("Incremental decode requires symmetric fixed context");
  }
  const context = privateSamples / 2;
  const expectedLength = metadata.channels * metadata.segment_samples;
  if (modelWindow.length !== expectedLength) {
    throw new Error(
      `Decoded model window has ${modelWindow.length} samples, expected ${expectedLength}`,
    );
  }
  if (context + chunk.samples > metadata.segment_samples) {
    throw new Error("Decoded model window cannot satisfy the owned chunk length");
  }

  const pcm = new Float32Array(metadata.channels * chunk.samples);
  for (let channel = 0; channel < metadata.channels; channel += 1) {
    const sourceOffset = channel * metadata.segment_samples + context;
    const targetOffset = channel * chunk.samples;
    pcm.set(
      modelWindow.subarray(sourceOffset, sourceOffset + chunk.samples),
      targetOffset,
    );
  }
  return pcm;
}

function copyPlanarChunk(target, pcm, channels, totalFrames, offset) {
  const samples = pcm.length / channels;
  if (!Number.isInteger(samples) || offset + samples > totalFrames) {
    throw new Error("Decoded PCM chunk does not fit the output timeline");
  }
  for (let channel = 0; channel < channels; channel += 1) {
    const sourceOffset = channel * samples;
    const targetOffset = channel * totalFrames + offset;
    target.set(pcm.subarray(sourceOffset, sourceOffset + samples), targetOffset);
  }
}

export async function createWebGpuEcdcDecoder({
  encodecModule,
  bundleJson,
  lmWeights,
  assetRoot,
  fetchImpl = globalThis.fetch.bind(globalThis),
  gpuContext = null,
}) {
  const module = encodecModule || {};
  requiredFunction(module.lmEcdcDecodeChunks, "lmEcdcDecodeChunks");
  requiredFunction(module.ecdcMetadata, "ecdcMetadata");
  requiredFunction(module.QuantizedLmChunkDecoder, "QuantizedLmChunkDecoder");
  const metadata = JSON.parse(bundleJson);
  const weights = normalizeBytes(lmWeights, "lmWeights");
  const neural = await createWebGpuDecoder({
    assetRoot,
    bundleMetadata: metadata,
    fetchImpl,
    gpuContext,
  });
  let released = false;

  return Object.freeze({
    adapter: neural.adapter,
    metadata,
    setupMs: neural.setupMs,

    async decode(ecdcPayload, {
      collectAudio = false,
      onChunk = null,
      onProgress = null,
      yieldControl = null,
    } = {}) {
      if (released) throw new Error("WebGPU ECDC decoder is released");
      const payload = normalizeBytes(ecdcPayload, "ecdcPayload");
      const containerStarted = performance.now();
      const parsed = module.lmEcdcDecodeChunks(bundleJson, payload);
      const ecdcMetadata = module.ecdcMetadata(payload);
      const audioLength = audioLengthFromMetadata(ecdcMetadata);
      const containerMs = performance.now() - containerStarted;
      const audio = collectAudio
        ? new Float32Array(metadata.channels * audioLength)
        : null;
      let entropyMs = 0;
      let modelMs = 0;
      let assemblyMs = 0;
      let deliveryMs = 0;
      let firstChunkMs = null;
      const pipelineStarted = performance.now();

      for (let index = 0; index < parsed.chunks.length; index += 1) {
        const chunk = parsed.chunks[index];
        const entropyStarted = performance.now();
        const frame = decodeLmChunk(module, bundleJson, weights, metadata, chunk);
        const chunkEntropyMs = performance.now() - entropyStarted;
        entropyMs += chunkEntropyMs;

        const decoded = await neural.decodeFrame(frame);
        modelMs += decoded.elapsedMs;

        const assemblyStarted = performance.now();
        const pcm = cropOwnedPcm(decoded.audio, metadata, chunk);
        if (audio) {
          copyPlanarChunk(audio, pcm, metadata.channels, audioLength, chunk.offset);
        }
        assemblyMs += performance.now() - assemblyStarted;
        if (firstChunkMs === null) firstChunkMs = performance.now() - pipelineStarted;

        if (onChunk) {
          const deliveryStarted = performance.now();
          await onChunk({
            index,
            total: parsed.chunks.length,
            offset: chunk.offset,
            samples: chunk.samples,
            pcm,
            modelWindow: decoded.audio,
            entropyMs: chunkEntropyMs,
            modelMs: decoded.elapsedMs,
          });
          deliveryMs += performance.now() - deliveryStarted;
        }
        onProgress?.({
          completed: index + 1,
          total: parsed.chunks.length,
          offset: chunk.offset,
          samples: chunk.samples,
        });
        if (yieldControl) await yieldControl();
      }

      return {
        audio,
        audioLength,
        chunkCount: parsed.chunks.length,
        metadata: ecdcMetadata,
        timings: {
          containerMs,
          entropyMs,
          modelMs,
          assemblyMs,
          deliveryMs,
          pipelineMs: performance.now() - pipelineStarted,
          firstChunkMs,
        },
      };
    },

    release() {
      if (released) return;
      released = true;
      neural.release();
    },
  });
}
