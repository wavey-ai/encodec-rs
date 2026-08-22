const GEOMETRIES = {
  encoder: {
    inputTime: 1632,
    inputChannels: 256,
    outputTime: 203,
    outputChannels: 512,
    kernel: 16,
    stride: 8,
  },
  decoder: {
    inputTime: 203,
    inputChannels: 512,
    outputTime: 1632,
    outputChannels: 256,
    kernel: 16,
    stride: 8,
  },
};

const mode = new URL(globalThis.location.href).searchParams.get("mode") || "encoder";
run(mode).then(
  (result) => postMessage({ ok: true, result }),
  (error) => postMessage({
    ok: false,
    error: String(error),
    stack: error?.stack || null,
  }),
);

async function run(selectedMode) {
  const geometry = GEOMETRIES[selectedMode];
  if (!geometry) throw new Error(`unsupported mode: ${selectedMode}`);
  if (!navigator.gpu) throw new Error("WebGPU is unavailable");

  const started = performance.now();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU did not return an adapter");
  const device = await adapter.requestDevice();

  const wasm = await prepareWasm(selectedMode, geometry);
  const gpu = await prepareGpu(device, selectedMode, geometry, wasm);
  try {
    wasm.run();
    const wasmReference = wasm.readOutput();

    await gpu.dispatch(2);
    const gpuIsolatedMs = [];
    const wasmIsolatedMs = [];
    for (let index = 0; index < 7; ++index) {
      if (index % 2 === 0) {
        gpuIsolatedMs.push(await gpu.dispatch(1));
        wasmIsolatedMs.push(timeSynchronous(wasm.run));
      } else {
        wasmIsolatedMs.push(timeSynchronous(wasm.run));
        gpuIsolatedMs.push(await gpu.dispatch(1));
      }
    }

    const gpuBatch = await gpu.dispatch(9);
    const readbackStarted = performance.now();
    const gpuOutput = await gpu.readOutput();
    const readbackMs = performance.now() - readbackStarted;
    const gpuMedianMs = median(gpuIsolatedMs);
    const wasmMedianMs = median(wasmIsolatedMs);

    return {
      mode: selectedMode,
      userAgent: navigator.userAgent,
      secureContext: globalThis.isSecureContext,
      adapter: adapterInfo(adapter),
      geometry,
      multiplyAccumulates: multiplyAccumulates(selectedMode, geometry),
      setupMs: {
        wasmModuleAndWeights: round(wasm.setupMs),
        gpuPipelineAndUpload: round(gpu.setupMs),
      },
      wasm: {
        isolatedMedianMs: round(wasmMedianMs),
        isolatedSamplesMs: wasmIsolatedMs.map(round),
      },
      webgpu: {
        isolatedMedianMs: round(gpuMedianMs),
        isolatedSamplesMs: gpuIsolatedMs.map(round),
        batchedDispatches: 9,
        batchedTotalMs: round(gpuBatch),
        batchedPerDispatchMs: round(gpuBatch / 9),
        dispatchAndReadbackMs: round(readbackMs),
      },
      speedup: {
        isolated: round(wasmMedianMs / gpuMedianMs),
        batched: round(wasmMedianMs / (gpuBatch / 9)),
      },
      parity: compareFloat32(wasmReference, gpuOutput),
      totalElapsedMs: round(performance.now() - started),
    };
  } finally {
    gpu.release();
    wasm.release();
    device.destroy();
  }
}

async function prepareWasm(selectedMode, geometry) {
  const started = performance.now();
  const moduleUrl = selectedMode === "encoder"
    ? new URL("./encoder/encodec-encoder.mjs", globalThis.location.href)
    : new URL("./decoder/encodec-convtranspose.mjs", globalThis.location.href);
  const createModule = (await import(moduleUrl)).default;
  const module = await createModule({
    locateFile: (file) => new URL(file, moduleUrl).href,
  });

  const inputLength = geometry.inputTime * geometry.inputChannels;
  const rawWeightLength = geometry.inputChannels * geometry.outputChannels * geometry.kernel;
  const outputLength = geometry.outputTime * geometry.outputChannels;
  const pointers = {
    input: allocate(module, inputLength),
    rawWeights: allocate(module, rawWeightLength),
    packedWeights: allocate(module, rawWeightLength),
    bias: allocate(module, geometry.outputChannels),
    output: allocate(module, outputLength),
  };
  const input = deterministicFloat32(inputLength, 0x1979, 0.25);
  const rawWeights = deterministicFloat32(rawWeightLength, 0x130323, 0.02);
  const bias = deterministicFloat32(geometry.outputChannels, 0x140323, 0.01);
  module.HEAPF32.set(input, pointers.input / 4);
  module.HEAPF32.set(rawWeights, pointers.rawWeights / 4);
  module.HEAPF32.set(bias, pointers.bias / 4);

  const packed = selectedMode === "encoder"
    ? module._pack_conv1d_nhwc_weights_8(
      pointers.rawWeights,
      pointers.packedWeights,
      geometry.inputChannels,
      geometry.outputChannels,
      geometry.kernel,
    )
    : module._pack_conv_transpose1d_weights(
      pointers.rawWeights,
      pointers.packedWeights,
      geometry.inputChannels,
      geometry.outputChannels,
      geometry.stride,
    );
  if (packed !== 1) throw new Error(`${selectedMode} WASM rejected weight packing`);

  const run = selectedMode === "encoder"
    ? () => {
      const ok = module._conv1d_nhwc_simd_8x8(
        pointers.input,
        pointers.packedWeights,
        pointers.bias,
        pointers.output,
        geometry.inputTime,
        geometry.inputChannels,
        geometry.outputChannels,
        geometry.kernel,
        geometry.stride,
      );
      if (ok !== 1) throw new Error("encoder WASM kernel failed");
    }
    : () => {
      const ok = module._conv_transpose1d_phase_simd_8x8_nhwc(
        pointers.input,
        pointers.packedWeights,
        pointers.bias,
        pointers.output,
        geometry.inputTime,
        geometry.inputChannels,
        geometry.outputChannels,
        geometry.stride,
      );
      if (ok !== 1) throw new Error("decoder WASM kernel failed");
    };

  return {
    setupMs: performance.now() - started,
    input,
    bias,
    packedWeights: module.HEAPF32.slice(
      pointers.packedWeights / 4,
      pointers.packedWeights / 4 + rawWeightLength,
    ),
    run,
    readOutput: () => module.HEAPF32.slice(
      pointers.output / 4,
      pointers.output / 4 + outputLength,
    ),
    release: () => {
      for (const pointer of Object.values(pointers)) module._free(pointer);
    },
  };
}

async function prepareGpu(device, selectedMode, geometry, wasm) {
  const started = performance.now();
  const outputLength = geometry.outputTime * geometry.outputChannels;
  const inputBuffer = createBuffer(device, wasm.input, GPUBufferUsage.STORAGE);
  const weightsBuffer = createBuffer(device, wasm.packedWeights, GPUBufferUsage.STORAGE);
  const biasBuffer = createBuffer(device, wasm.bias, GPUBufferUsage.STORAGE);
  const outputBuffer = device.createBuffer({
    size: outputLength * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    size: outputLength * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const reduction = geometry.inputChannels * geometry.kernel;
  const params = new Uint32Array([
    geometry.inputChannels,
    geometry.outputChannels,
    geometry.kernel,
    geometry.stride,
    geometry.outputTime,
    reduction,
    0,
    0,
  ]);
  const paramsBuffer = createBuffer(device, params, GPUBufferUsage.UNIFORM);
  const shader = device.createShaderModule({
    code: selectedMode === "encoder" ? ENCODER_SHADER : DECODER_SHADER,
  });
  const compilation = await shader.getCompilationInfo();
  const errors = Array.from(compilation.messages).filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(errors.map((message) => message.message).join("\n"));
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: weightsBuffer } },
      { binding: 2, resource: { buffer: biasBuffer } },
      { binding: 3, resource: { buffer: outputBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
    ],
  });
  const outputBlocks = geometry.outputChannels / 4;
  const dispatch = async (iterations) => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let index = 0; index < iterations; ++index) {
      pass.dispatchWorkgroups(Math.ceil(geometry.outputTime / 8), Math.ceil(outputBlocks / 8));
    }
    pass.end();
    const dispatchStarted = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - dispatchStarted;
  };
  const readOutput = async () => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(geometry.outputTime / 8), Math.ceil(outputBlocks / 8));
    pass.end();
    encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputLength * 4);
    device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(readbackBuffer.getMappedRange()).slice();
    readbackBuffer.unmap();
    return output;
  };

  return {
    setupMs: performance.now() - started,
    dispatch,
    readOutput,
    release: () => {
      inputBuffer.destroy();
      weightsBuffer.destroy();
      biasBuffer.destroy();
      outputBuffer.destroy();
      readbackBuffer.destroy();
      paramsBuffer.destroy();
    },
  };
}

function createBuffer(device, values, usage) {
  const buffer = device.createBuffer({
    size: values.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function allocate(module, length) {
  const pointer = module._malloc(length * 4);
  if (pointer === 0) throw new Error(`failed to allocate ${length} float32 values`);
  return pointer;
}

function deterministicFloat32(length, salt, amplitude) {
  const values = new Float32Array(length);
  let state = (0x9e3779b9 ^ salt) >>> 0;
  for (let index = 0; index < length; ++index) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values[index] = (state / 0xffffffff - 0.5) * 2 * amplitude;
  }
  return values;
}

function multiplyAccumulates(selectedMode, geometry) {
  const time = selectedMode === "encoder" ? geometry.outputTime : geometry.inputTime;
  return time * geometry.inputChannels *
    geometry.outputChannels * geometry.kernel;
}

function compareFloat32(reference, candidate) {
  if (reference.length !== candidate.length) {
    throw new Error(`output length mismatch: ${reference.length} != ${candidate.length}`);
  }
  const referenceBits = new Uint32Array(reference.buffer, reference.byteOffset, reference.length);
  const candidateBits = new Uint32Array(candidate.buffer, candidate.byteOffset, candidate.length);
  let exactMismatches = 0;
  let squaredSignal = 0;
  let squaredError = 0;
  let maxAbsError = 0;
  for (let index = 0; index < reference.length; ++index) {
    if (referenceBits[index] !== candidateBits[index]) exactMismatches += 1;
    const error = candidate[index] - reference[index];
    squaredSignal += reference[index] * reference[index];
    squaredError += error * error;
    maxAbsError = Math.max(maxAbsError, Math.abs(error));
  }
  return {
    exact: exactMismatches === 0,
    exactMismatches,
    maxAbsError,
    rmse: Math.sqrt(squaredError / reference.length),
    snrDb: squaredError === 0 ? null : 10 * Math.log10(squaredSignal / squaredError),
  };
}

function adapterInfo(adapter) {
  return {
    vendor: adapter.info?.vendor || null,
    architecture: adapter.info?.architecture || null,
    device: adapter.info?.device || null,
    description: adapter.info?.description || null,
  };
}

function timeSynchronous(callback) {
  const started = performance.now();
  callback();
  return performance.now() - started;
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

function round(value) {
  return Number(value.toFixed(3));
}

const ENCODER_SHADER = `
struct Params {
  inputChannels: u32,
  outputChannels: u32,
  kernel: u32,
  stride: u32,
  outputTime: u32,
  reduction: u32,
  padding0: u32,
  padding1: u32,
}

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> packedWeights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let time = id.x;
  let outputBlock = id.y;
  let outputBlocks = params.outputChannels / 4u;
  if (time >= params.outputTime || outputBlock >= outputBlocks) {
    return;
  }

  var value = biases[outputBlock];
  let outputPair = outputBlock / 2u;
  let outputHalf = outputBlock % 2u;
  for (var inputChannel = 0u; inputChannel < params.inputChannels; inputChannel++) {
    for (var tap = 0u; tap < params.kernel; tap++) {
      let reductionIndex = inputChannel * params.kernel + tap;
      let inputIndex = (time * params.stride + tap) * params.inputChannels + inputChannel;
      let weightIndex = (outputPair * params.reduction + reductionIndex) * 2u + outputHalf;
      value += inputValues[inputIndex] * packedWeights[weightIndex];
    }
  }
  outputValues[time * outputBlocks + outputBlock] = value;
}
`;

const DECODER_SHADER = `
struct Params {
  inputChannels: u32,
  outputChannels: u32,
  kernel: u32,
  stride: u32,
  outputTime: u32,
  reduction: u32,
  padding0: u32,
  padding1: u32,
}

@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> packedWeights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputIndex = id.x;
  let outputBlock = id.y;
  let outputBlocks = params.outputChannels / 4u;
  if (outputIndex >= params.outputTime || outputBlock >= outputBlocks) {
    return;
  }

  let frame = outputIndex / params.stride;
  let phase = outputIndex % params.stride;
  let inputTime = params.outputTime / params.stride - 1u;
  var value = biases[outputBlock];
  for (var inputChannel = 0u; inputChannel < params.inputChannels; inputChannel++) {
    if (frame < inputTime) {
      let inputIndex = frame * params.inputChannels + inputChannel;
      let weightIndex = ((phase * 2u) * params.inputChannels + inputChannel) *
        outputBlocks + outputBlock;
      value += inputValues[inputIndex] * packedWeights[weightIndex];
    }
    if (frame > 0u) {
      let inputIndex = (frame - 1u) * params.inputChannels + inputChannel;
      let weightIndex = ((phase * 2u + 1u) * params.inputChannels + inputChannel) *
        outputBlocks + outputBlock;
      value += inputValues[inputIndex] * packedWeights[weightIndex];
    }
  }
  outputValues[outputIndex * outputBlocks + outputBlock] = value;
}
`;
