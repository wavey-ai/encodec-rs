const STORAGE_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export async function requestEncodecWebGpuDevice() {
  if (!globalThis.navigator?.gpu) {
    throw new Error("WebGPU is unavailable");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("WebGPU did not return an adapter");
  }
  const device = await adapter.requestDevice();
  return { adapter, device };
}

export async function loadTensorViews(root, names, fetchImpl) {
  const manifestResponse = await fetchImpl(new URL("weights.json", root), {
    cache: "force-cache",
  });
  if (!manifestResponse.ok) {
    throw new Error(`fetch weight manifest failed: ${manifestResponse.status}`);
  }
  const manifest = await manifestResponse.json();
  const weightsResponse = await fetchImpl(new URL(manifest.file, root), {
    cache: "force-cache",
  });
  if (!weightsResponse.ok) {
    throw new Error(`fetch weights failed: ${weightsResponse.status}`);
  }
  const bytes = await weightsResponse.arrayBuffer();
  if (bytes.byteLength !== manifest.byteLength) {
    throw new Error("weight length does not match its manifest");
  }
  return new Map(names.map((name) => {
    const tensor = manifest.tensors?.[name];
    if (!tensor) throw new Error(`weight is missing: ${name}`);
    return [
      name,
      new Float32Array(bytes, tensor.offsetBytes, tensor.length),
    ];
  }));
}

export async function fetchJson(url, fetchImpl) {
  const response = await fetchImpl(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`fetch ${url} failed: ${response.status}`);
  return response.json();
}

export function normalizeAssetRoot(value) {
  const url = value instanceof URL
    ? value
    : new URL(String(value), globalThis.location?.href);
  return new URL(url.href.replace(/\/?$/, "/"));
}

export function adapterDescription(adapter) {
  return {
    vendor: adapter.info?.vendor || null,
    architecture: adapter.info?.architecture || null,
    device: adapter.info?.device || null,
    description: adapter.info?.description || null,
  };
}

export class WebGpuResources {
  constructor(device) {
    this.device = device;
    this.buffers = [];
  }

  storage(length, label, extraUsage = 0) {
    return this.bytes(length * Float32Array.BYTES_PER_ELEMENT, label,
      STORAGE_USAGE | extraUsage);
  }

  storageU32(length, label, extraUsage = 0) {
    return this.bytes(length * Uint32Array.BYTES_PER_ELEMENT, label,
      STORAGE_USAGE | extraUsage);
  }

  upload(values, label, usage = STORAGE_USAGE) {
    const byteLength = align4(values.byteLength);
    const buffer = this.device.createBuffer({
      label,
      size: byteLength,
      usage,
      mappedAtCreation: true,
    });
    const target = new Uint8Array(buffer.getMappedRange());
    target.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
    buffer.unmap();
    this.buffers.push(buffer);
    return buffer;
  }

  uniform(bytes, label) {
    return this.upload(
      new Uint8Array(bytes),
      label,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
  }

  readback(byteLength, label) {
    return this.bytes(
      byteLength,
      label,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
  }

  bytes(byteLength, label, usage) {
    const buffer = this.device.createBuffer({
      label,
      size: align4(byteLength),
      usage,
    });
    this.buffers.push(buffer);
    return buffer;
  }

  destroy() {
    for (const buffer of this.buffers.splice(0)) buffer.destroy();
  }
}

export async function createWebGpuKernels(device, resources) {
  const modules = new Map();
  const compile = async (name, code, entryPoint = "main", layout = "auto") => {
    let module = modules.get(code);
    if (!module) {
      module = device.createShaderModule({ label: `${name} shader`, code });
      modules.set(code, module);
    }
    return device.createComputePipelineAsync({
      label: `${name} pipeline`,
      layout,
      compute: { module, entryPoint },
    });
  };

  const [
    normalizeAudio,
    conv1d,
    groupNorm,
    add,
    copy,
    lstmSequence,
    rvqEncode,
    rvqDecode,
    convTranspose,
    crop,
    scaleToPlanar,
  ] = await Promise.all([
    compile("normalize audio", NORMALIZE_AUDIO_SHADER),
    compile("conv1d", CONV1D_SHADER),
    compile("group norm", GROUP_NORM_SHADER),
    compile("add", ADD_SHADER),
    compile("copy", COPY_SHADER),
    compile("LSTM sequence", LSTM_SEQUENCE_SHADER),
    compile("RVQ encode", RVQ_ENCODE_SHADER),
    compile("RVQ decode", RVQ_DECODE_SHADER),
    compile("conv transpose", CONV_TRANSPOSE_SHADER),
    compile("crop", CROP_SHADER),
    compile("scale planar", SCALE_TO_PLANAR_SHADER),
  ]);

  const kernels = {
    device,
    resources,
    pipelines: {
      normalizeAudio,
      conv1d,
      groupNorm,
      add,
      copy,
      lstmSequence,
      rvqEncode,
      rvqDecode,
      convTranspose,
      crop,
      scaleToPlanar,
    },
  };
  return kernels;
}

export function recordWebGpuOperations(commandEncoder, operations, label) {
  let pass = null;
  const endPass = () => {
    if (!pass) return;
    pass.end();
    pass = null;
  };
  for (const operation of operations) {
    if (typeof operation === "function") {
      if (!pass) pass = commandEncoder.beginComputePass({ label });
      operation(pass);
      continue;
    }
    endPass();
    operation.recordCommandEncoder(commandEncoder);
  }
  endPass();
}

export function createNormalizeAudioOperation(
  kernels,
  { input, output, scale, time, channels },
) {
  const params = uniformU32(kernels.resources, [time, channels, 0, 0],
    "normalize audio parameters");
  const bindGroup = kernels.device.createBindGroup({
    layout: kernels.pipelines.normalizeAudio.getBindGroupLayout(0),
    entries: [
      bufferEntry(0, input),
      bufferEntry(1, output),
      bufferEntry(2, scale),
      bufferEntry(3, params),
    ],
  });
  return (pass) => {
    pass.setPipeline(kernels.pipelines.normalizeAudio);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
  };
}

export function createConv1dOperation(kernels, options) {
  const {
    input,
    output,
    weights,
    bias,
    inputTime,
    inputChannels,
    outputChannels,
    kernel,
    stride,
    paddingLeft = 0,
    applyElu = false,
    label = "conv1d",
  } = options;
  const outputTime = Math.floor(
    (inputTime + paddingLeft + (options.paddingRight ?? paddingLeft) - kernel) /
      stride,
  ) + 1;
  const packed = packConvWeights4(weights, inputChannels, outputChannels, kernel);
  const paddedBias = padChannels4(bias, outputChannels);
  const weightBuffer = kernels.resources.upload(packed, `${label} weights`);
  const biasBuffer = kernels.resources.upload(paddedBias, `${label} bias`);
  const params = uniformU32(kernels.resources, [
    inputTime,
    inputChannels,
    outputChannels,
    kernel,
    stride,
    paddingLeft,
    outputTime,
    applyElu ? 1 : 0,
  ], `${label} parameters`);
  const bindGroup = kernels.device.createBindGroup({
    label: `${label} bind group`,
    layout: kernels.pipelines.conv1d.getBindGroupLayout(0),
    entries: [
      bufferEntry(0, input),
      bufferEntry(1, weightBuffer),
      bufferEntry(2, biasBuffer),
      bufferEntry(3, output),
      bufferEntry(4, params),
    ],
  });
  const outputBlocks = Math.ceil(outputChannels / 4);
  return {
    outputTime,
    outputChannels,
    storageChannels: outputBlocks * 4,
    record(pass) {
      pass.setPipeline(kernels.pipelines.conv1d);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(outputTime / 8), Math.ceil(outputBlocks / 8));
    },
  };
}

export function createGroupNormOperation(kernels, options) {
  const {
    values,
    scale,
    bias,
    time,
    channels,
    storageChannels = channels,
    label = "group norm",
  } = options;
  const scaleBuffer = kernels.resources.upload(scale, `${label} scale`);
  const biasBuffer = kernels.resources.upload(bias, `${label} bias`);
  const params = uniformU32(kernels.resources, [
    time,
    channels,
    storageChannels,
    time * channels,
  ], `${label} parameters`);
  const bindGroup = kernels.device.createBindGroup({
    label: `${label} bind group`,
    layout: kernels.pipelines.groupNorm.getBindGroupLayout(0),
    entries: [
      bufferEntry(0, values),
      bufferEntry(1, scaleBuffer),
      bufferEntry(2, biasBuffer),
      bufferEntry(3, params),
    ],
  });
  return (pass) => {
    pass.setPipeline(kernels.pipelines.groupNorm);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
  };
}

export function createAddOperation(
  kernels,
  { destination, source, length, applyElu = false, label = "add" },
) {
  const params = uniformU32(kernels.resources, [length / 4, applyElu ? 1 : 0, 0, 0],
    `${label} parameters`);
  const bindGroup = kernels.device.createBindGroup({
    label: `${label} bind group`,
    layout: kernels.pipelines.add.getBindGroupLayout(0),
    entries: [
      bufferEntry(0, destination),
      bufferEntry(1, source),
      bufferEntry(2, params),
    ],
  });
  return (pass) => {
    pass.setPipeline(kernels.pipelines.add);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 4 / 256));
  };
}

export function createCopyOperation(
  kernels,
  { input, output, length, label = "copy" },
) {
  const params = uniformU32(kernels.resources, [length / 4, 0, 0, 0],
    `${label} parameters`);
  const bindGroup = kernels.device.createBindGroup({
    label: `${label} bind group`,
    layout: kernels.pipelines.copy.getBindGroupLayout(0),
    entries: [bufferEntry(0, input), bufferEntry(1, output), bufferEntry(2, params)],
  });
  return (pass) => {
    pass.setPipeline(kernels.pipelines.copy);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(length / 4 / 256));
  };
}

export function createLstmLayerOperation(kernels, options) {
  const {
    input,
    output,
    gates,
    hidden,
    cell,
    inputWeights,
    recurrentWeights,
    bias,
    sequenceLength,
    hiddenSize,
    label,
  } = options;
  const gateSize = 4 * hiddenSize;
  const projection = createConv1dOperation(kernels, {
    input,
    output: gates,
    weights: inputWeights,
    bias,
    inputTime: sequenceLength,
    inputChannels: hiddenSize,
    outputChannels: gateSize,
    kernel: 1,
    stride: 1,
    label: `${label} input projection`,
  });
  const recurrentPacked = packConvWeights4(
    recurrentWeights,
    hiddenSize,
    gateSize,
    1,
  );
  const recurrentBuffer = kernels.resources.upload(
    recurrentPacked,
    `${label} recurrent weights`,
  );
  const params = uniformU32(kernels.resources, [
    sequenceLength,
    hiddenSize,
    gateSize,
    0,
  ], `${label} sequence parameters`);
  const sequenceBindGroup = kernels.device.createBindGroup({
    label: `${label} sequence bind group`,
    layout: kernels.pipelines.lstmSequence.getBindGroupLayout(0),
    entries: [
      bufferEntry(0, gates),
      bufferEntry(1, recurrentBuffer),
      bufferEntry(2, hidden),
      bufferEntry(3, cell),
      bufferEntry(4, output),
      bufferEntry(5, params),
    ],
  });
  return {
    recordCommandEncoder(commandEncoder) {
      let pass = commandEncoder.beginComputePass({ label: `${label} projection` });
      projection.record(pass);
      pass.end();
      pass = commandEncoder.beginComputePass({ label: `${label} recurrent sequence` });
      pass.setPipeline(kernels.pipelines.lstmSequence);
      pass.setBindGroup(0, sequenceBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    },
  };
}

export function createRvqEncodeOperation(kernels, options) {
  const {
    input,
    residual,
    embeddings,
    norms,
    codes,
    time,
    dimension,
    entries,
    codebooks,
  } = options;
  const copy = createCopyOperation(kernels, {
    input,
    output: residual,
    length: time * dimension,
    label: "RVQ residual copy",
  });
  const embeddingBuffer = kernels.resources.upload(embeddings, "RVQ embeddings");
  const normBuffer = kernels.resources.upload(norms, "RVQ norms");
  const books = [];
  for (let book = 0; book < codebooks; book += 1) {
    const params = uniformU32(kernels.resources, [time, dimension, entries, book],
      `RVQ book ${book} parameters`);
    const bindGroup = kernels.device.createBindGroup({
      label: `RVQ book ${book} bind group`,
      layout: kernels.pipelines.rvqEncode.getBindGroupLayout(0),
      entries: [
        bufferEntry(0, residual),
        bufferEntry(1, embeddingBuffer),
        bufferEntry(2, normBuffer),
        bufferEntry(3, codes),
        bufferEntry(4, params),
      ],
    });
    books.push(bindGroup);
  }
  return (pass) => {
    copy(pass);
    pass.setPipeline(kernels.pipelines.rvqEncode);
    for (const bindGroup of books) {
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(time);
    }
  };
}

export function createRvqDecodeOperation(kernels, options) {
  const {
    codes,
    embeddings,
    output,
    time,
    dimension,
    entries,
    codebooks,
  } = options;
  const embeddingBuffer = kernels.resources.upload(embeddings, "decoder RVQ embeddings");
  const params = uniformU32(kernels.resources, [time, dimension, entries, codebooks],
    "decoder RVQ parameters");
  const bindGroup = kernels.device.createBindGroup({
    layout: kernels.pipelines.rvqDecode.getBindGroupLayout(0),
    entries: [
      bufferEntry(0, codes),
      bufferEntry(1, embeddingBuffer),
      bufferEntry(2, output),
      bufferEntry(3, params),
    ],
  });
  return (pass) => {
    pass.setPipeline(kernels.pipelines.rvqDecode);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(time / 8), Math.ceil(dimension / 4 / 8));
  };
}

export function createConvTransposeOperation(kernels, options) {
  const {
    input,
    output,
    weights,
    bias,
    inputTime,
    inputChannels,
    outputChannels,
    stride,
    label,
  } = options;
  const rawOutputTime = (inputTime + 1) * stride;
  const packed = packConvTransposeWeights4(
    weights,
    inputChannels,
    outputChannels,
    stride,
  );
  const weightBuffer = kernels.resources.upload(packed, `${label} weights`);
  const biasBuffer = kernels.resources.upload(
    padChannels4(bias, outputChannels),
    `${label} bias`,
  );
  const params = uniformU32(kernels.resources, [
    inputTime,
    inputChannels,
    outputChannels,
    stride,
    rawOutputTime,
    0,
    0,
    0,
  ], `${label} parameters`);
  const bindGroup = kernels.device.createBindGroup({
    label: `${label} bind group`,
    layout: kernels.pipelines.convTranspose.getBindGroupLayout(0),
    entries: [
      bufferEntry(0, input),
      bufferEntry(1, weightBuffer),
      bufferEntry(2, biasBuffer),
      bufferEntry(3, output),
      bufferEntry(4, params),
    ],
  });
  const blocks = Math.ceil(outputChannels / 4);
  return {
    rawOutputTime,
    record(pass) {
      pass.setPipeline(kernels.pipelines.convTranspose);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(rawOutputTime / 8), Math.ceil(blocks / 8));
    },
  };
}

export function createCropOperation(kernels, options) {
  const {
    input,
    output,
    rawTime,
    channels,
    cropLeft,
    outputTime,
    label,
  } = options;
  const params = uniformU32(kernels.resources, [
    rawTime,
    channels,
    cropLeft,
    outputTime,
  ], `${label} parameters`);
  const bindGroup = kernels.device.createBindGroup({
    label: `${label} bind group`,
    layout: kernels.pipelines.crop.getBindGroupLayout(0),
    entries: [bufferEntry(0, input), bufferEntry(1, output), bufferEntry(2, params)],
  });
  return (pass) => {
    pass.setPipeline(kernels.pipelines.crop);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outputTime / 8), Math.ceil(channels / 4 / 8));
  };
}

export function createScaleToPlanarOperation(kernels, options) {
  const {
    input,
    output,
    time,
    channels,
    storageChannels = channels,
  } = options;
  const bytes = new ArrayBuffer(16);
  const view = new DataView(bytes);
  view.setUint32(0, time, true);
  view.setUint32(4, channels, true);
  view.setUint32(8, storageChannels, true);
  view.setFloat32(12, 1, true);
  const params = kernels.resources.uniform(bytes, "decoder output parameters");
  const bindGroup = kernels.device.createBindGroup({
    layout: kernels.pipelines.scaleToPlanar.getBindGroupLayout(0),
    entries: [bufferEntry(0, input), bufferEntry(1, output), bufferEntry(2, params)],
  });
  return {
    setScale(scale) {
      kernels.device.queue.writeBuffer(params, 12, new Float32Array([scale]));
    },
    record(pass) {
      pass.setPipeline(kernels.pipelines.scaleToPlanar);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(time / 16), channels);
    },
  };
}

export async function readFloat32Buffer(buffer, byteLength) {
  await buffer.mapAsync(GPUMapMode.READ, 0, byteLength);
  const output = new Float32Array(buffer.getMappedRange(0, byteLength)).slice();
  buffer.unmap();
  return output;
}

export async function readUint32Buffer(buffer, byteLength) {
  await buffer.mapAsync(GPUMapMode.READ, 0, byteLength);
  const output = new Uint32Array(buffer.getMappedRange(0, byteLength)).slice();
  buffer.unmap();
  return output;
}

function bufferEntry(binding, buffer) {
  return { binding, resource: { buffer } };
}

function uniformU32(resources, words, label) {
  return resources.upload(
    new Uint32Array(words),
    label,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  );
}

function align4(value) {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

function padChannels4(values, channels) {
  const output = new Float32Array(Math.ceil(channels / 4) * 4);
  output.set(values.subarray(0, channels));
  return output;
}

function packConvWeights4(weights, inputChannels, outputChannels, kernel) {
  const reduction = inputChannels * kernel;
  const outputBlocks = Math.ceil(outputChannels / 4);
  const packed = new Float32Array(outputBlocks * reduction * 4);
  for (let output = 0; output < outputChannels; output += 1) {
    const block = Math.floor(output / 4);
    const lane = output % 4;
    const source = output * reduction;
    const destination = block * reduction * 4 + lane;
    for (let index = 0; index < reduction; index += 1) {
      packed[destination + index * 4] = weights[source + index];
    }
  }
  return packed;
}

function packConvTransposeWeights4(
  weights,
  inputChannels,
  outputChannels,
  stride,
) {
  const kernel = stride * 2;
  const outputBlocks = Math.ceil(outputChannels / 4);
  const packed = new Float32Array(
    stride * 2 * inputChannels * outputBlocks * 4,
  );
  for (let phase = 0; phase < stride; phase += 1) {
    for (let tap = 0; tap < 2; tap += 1) {
      const kernelIndex = phase + tap * stride;
      for (let input = 0; input < inputChannels; input += 1) {
        for (let output = 0; output < outputChannels; output += 1) {
          const block = Math.floor(output / 4);
          const lane = output % 4;
          const source = ((input * outputChannels + output) * kernel) + kernelIndex;
          const destination = (
            (((phase * 2 + tap) * inputChannels + input) * outputBlocks + block) * 4
          ) + lane;
          packed[destination] = weights[source];
        }
      }
    }
  }
  return packed;
}

const NORMALIZE_AUDIO_SHADER = /* wgsl */ `
struct Params { time: u32, channels: u32, padding0: u32, padding1: u32 }
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputScale: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> sums: array<f32, 256>;
var<workgroup> sharedScale: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  var sum = 0.0;
  for (var time = lane; time < params.time; time += 256u) {
    var mono = 0.0;
    for (var channel = 0u; channel < params.channels; channel++) {
      mono += inputValues[channel * params.time + time];
    }
    mono /= f32(params.channels);
    sum += mono * mono;
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var width = 128u; width > 0u; width >>= 1u) {
    if (lane < width) { sums[lane] += sums[lane + width]; }
    workgroupBarrier();
  }
  if (lane == 0u) {
    sharedScale = sqrt(sums[0] / f32(params.time)) + 1.0e-8;
    outputScale[0] = sharedScale;
  }
  workgroupBarrier();
  let length = params.time * params.channels;
  for (var index = lane; index < length; index += 256u) {
    let channel = index / params.time;
    let time = index - channel * params.time;
    outputValues[time * params.channels + channel] = inputValues[index] / sharedScale;
  }
}
`;

const CONV1D_SHADER = /* wgsl */ `
struct Params {
  inputTime: u32,
  inputChannels: u32,
  outputChannels: u32,
  kernel: u32,
  stride: u32,
  paddingLeft: u32,
  outputTime: u32,
  applyElu: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

fn elu(value: f32) -> f32 {
  return select(exp(value) - 1.0, value, value >= 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let time = id.x;
  let block = id.y;
  let blocks = (params.outputChannels + 3u) / 4u;
  if (time >= params.outputTime || block >= blocks) { return; }
  var value = biases[block];
  let reduction = params.inputChannels * params.kernel;
  for (var channel = 0u; channel < params.inputChannels; channel++) {
    for (var tap = 0u; tap < params.kernel; tap++) {
      var sourceTime = i32(time * params.stride + tap) - i32(params.paddingLeft);
      if (sourceTime < 0) {
        sourceTime = -sourceTime;
      } else if (sourceTime >= i32(params.inputTime)) {
        sourceTime = 2 * i32(params.inputTime) - 2 - sourceTime;
      }
      var sample = inputValues[u32(sourceTime) * params.inputChannels + channel];
      if (params.applyElu != 0u) { sample = elu(sample); }
      let reductionIndex = channel * params.kernel + tap;
      value += sample * weights[block * reduction + reductionIndex];
    }
  }
  outputValues[time * blocks + block] = value;
}
`;

const GROUP_NORM_SHADER = /* wgsl */ `
struct Params { time: u32, channels: u32, storageChannels: u32, length: u32 }
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> sums: array<f32, 256>;
var<workgroup> mean: f32;
var<workgroup> inverseDeviation: f32;

fn storageIndex(index: u32) -> u32 {
  let time = index / params.channels;
  let channel = index - time * params.channels;
  return time * params.storageChannels + channel;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  var sum = 0.0;
  for (var index = lane; index < params.length; index += 256u) {
    sum += values[storageIndex(index)];
  }
  sums[lane] = sum;
  workgroupBarrier();
  for (var width = 128u; width > 0u; width >>= 1u) {
    if (lane < width) { sums[lane] += sums[lane + width]; }
    workgroupBarrier();
  }
  if (lane == 0u) { mean = sums[0] / f32(params.length); }
  workgroupBarrier();
  var squared = 0.0;
  for (var index = lane; index < params.length; index += 256u) {
    let difference = values[storageIndex(index)] - mean;
    squared += difference * difference;
  }
  sums[lane] = squared;
  workgroupBarrier();
  for (var width = 128u; width > 0u; width >>= 1u) {
    if (lane < width) { sums[lane] += sums[lane + width]; }
    workgroupBarrier();
  }
  if (lane == 0u) {
    inverseDeviation = inverseSqrt(sums[0] / f32(params.length) + 1.0e-5);
  }
  workgroupBarrier();
  for (var index = lane; index < params.length; index += 256u) {
    let channel = index % params.channels;
    let offset = storageIndex(index);
    values[offset] = (values[offset] - mean) * inverseDeviation * scales[channel] + biases[channel];
  }
}
`;

const ADD_SHADER = /* wgsl */ `
struct Params { vectors: u32, applyElu: u32, padding0: u32, padding1: u32 }
@group(0) @binding(0) var<storage, read_write> destination: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
fn elu(value: vec4<f32>) -> vec4<f32> {
  return select(exp(value) - vec4(1.0), value, value >= vec4(0.0));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.vectors) { return; }
  var value = destination[id.x] + source[id.x];
  if (params.applyElu != 0u) { value = elu(value); }
  destination[id.x] = value;
}
`;

const COPY_SHADER = /* wgsl */ `
struct Params { vectors: u32, padding0: u32, padding1: u32, padding2: u32 }
@group(0) @binding(0) var<storage, read> inputValues: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.vectors) { outputValues[id.x] = inputValues[id.x]; }
}
`;

const LSTM_SEQUENCE_SHADER = /* wgsl */ `
struct Params { sequenceLength: u32, hiddenSize: u32, gateSize: u32, padding: u32 }
@group(0) @binding(0) var<storage, read_write> gates: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(3) var<storage, read_write> cell: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
fn sigmoid(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
fn stableTanh(value: f32) -> f32 { return 2.0 * sigmoid(2.0 * value) - 1.0; }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let lane = local.x;
  let gateBlocks = params.gateSize / 4u;
  for (var time = 0u; time < params.sequenceLength; time++) {
    if (time > 0u) {
      for (var block = lane; block < gateBlocks; block += 256u) {
        let gateOffset = time * params.gateSize + block * 4u;
        var value = vec4(
          gates[gateOffset],
          gates[gateOffset + 1u],
          gates[gateOffset + 2u],
          gates[gateOffset + 3u],
        );
        for (var index = 0u; index < params.hiddenSize; index++) {
          value += hidden[index] * weights[block * params.hiddenSize + index];
        }
        gates[gateOffset] = value.x;
        gates[gateOffset + 1u] = value.y;
        gates[gateOffset + 2u] = value.z;
        gates[gateOffset + 3u] = value.w;
      }
    }
    workgroupBarrier();
    for (var index = lane; index < params.hiddenSize; index += 256u) {
      let base = time * params.gateSize;
      let inputGate = sigmoid(gates[base + index]);
      let outputGate = sigmoid(gates[base + params.hiddenSize + index]);
      let forgetGate = sigmoid(gates[base + 2u * params.hiddenSize + index]);
      let cellGate = stableTanh(gates[base + 3u * params.hiddenSize + index]);
      let previousCell = select(cell[index], 0.0, time == 0u);
      let nextCell = forgetGate * previousCell + inputGate * cellGate;
      let nextHidden = outputGate * stableTanh(nextCell);
      cell[index] = nextCell;
      hidden[index] = nextHidden;
      outputValues[time * params.hiddenSize + index] = nextHidden;
    }
    workgroupBarrier();
  }
}
`;

const RVQ_ENCODE_SHADER = /* wgsl */ `
struct Params { time: u32, dimension: u32, entries: u32, book: u32 }
@group(0) @binding(0) var<storage, read_write> residual: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> embeddings: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> norms: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> bestScores: array<f32, 256>;
var<workgroup> bestCodes: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(local_invocation_id) local: vec3<u32>,
) {
  let time = group.x;
  let lane = local.x;
  let blocks = params.dimension / 4u;
  var bestScore = -3.402823e38;
  var bestCode = 0u;
  for (var code = lane; code < params.entries; code += 256u) {
    var product = vec4(0.0);
    let vectorBase = time * blocks;
    let embeddingBase = (params.book * params.entries + code) * blocks;
    for (var block = 0u; block < blocks; block++) {
      product += residual[vectorBase + block] * embeddings[embeddingBase + block];
    }
    let score = 2.0 * (product.x + product.y + product.z + product.w) -
      norms[params.book * params.entries + code];
    if (score > bestScore || (score == bestScore && code < bestCode)) {
      bestScore = score;
      bestCode = code;
    }
  }
  bestScores[lane] = bestScore;
  bestCodes[lane] = bestCode;
  workgroupBarrier();
  for (var width = 128u; width > 0u; width >>= 1u) {
    if (lane < width) {
      let otherScore = bestScores[lane + width];
      let otherCode = bestCodes[lane + width];
      if (otherScore > bestScores[lane] ||
          (otherScore == bestScores[lane] && otherCode < bestCodes[lane])) {
        bestScores[lane] = otherScore;
        bestCodes[lane] = otherCode;
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    codes[params.book * params.time + time] = bestCodes[0];
  }
  workgroupBarrier();
  let selected = bestCodes[0];
  let vectorBase = time * blocks;
  let embeddingBase = (params.book * params.entries + selected) * blocks;
  for (var block = lane; block < blocks; block += 256u) {
    residual[vectorBase + block] -= embeddings[embeddingBase + block];
  }
}
`;

const RVQ_DECODE_SHADER = /* wgsl */ `
struct Params { time: u32, dimension: u32, entries: u32, codebooks: u32 }
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> embeddings: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let time = id.x;
  let block = id.y;
  let blocks = params.dimension / 4u;
  if (time >= params.time || block >= blocks) { return; }
  var value = vec4(0.0);
  for (var book = 0u; book < params.codebooks; book++) {
    let code = codes[book * params.time + time];
    value += embeddings[(book * params.entries + code) * blocks + block];
  }
  outputValues[time * blocks + block] = value;
}
`;

const CONV_TRANSPOSE_SHADER = /* wgsl */ `
struct Params {
  inputTime: u32,
  inputChannels: u32,
  outputChannels: u32,
  stride: u32,
  outputTime: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
}
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputIndex = id.x;
  let block = id.y;
  let blocks = (params.outputChannels + 3u) / 4u;
  if (outputIndex >= params.outputTime || block >= blocks) { return; }
  let frame = outputIndex / params.stride;
  let phase = outputIndex % params.stride;
  var value = biases[block];
  for (var channel = 0u; channel < params.inputChannels; channel++) {
    if (frame < params.inputTime) {
      let weight = ((phase * 2u * params.inputChannels + channel) * blocks) + block;
      value += inputValues[frame * params.inputChannels + channel] * weights[weight];
    }
    if (frame > 0u) {
      let weight = ((((phase * 2u + 1u) * params.inputChannels) + channel) * blocks) + block;
      value += inputValues[(frame - 1u) * params.inputChannels + channel] * weights[weight];
    }
  }
  outputValues[outputIndex * blocks + block] = value;
}
`;

const CROP_SHADER = /* wgsl */ `
struct Params { rawTime: u32, channels: u32, cropLeft: u32, outputTime: u32 }
@group(0) @binding(0) var<storage, read> inputValues: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let time = id.x;
  let block = id.y;
  let blocks = params.channels / 4u;
  if (time >= params.outputTime || block >= blocks) { return; }
  outputValues[time * blocks + block] =
    inputValues[(time + params.cropLeft) * blocks + block];
}
`;

const SCALE_TO_PLANAR_SHADER = /* wgsl */ `
struct Params { time: u32, channels: u32, storageChannels: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> inputValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let time = id.x;
  let channel = id.y;
  if (time >= params.time || channel >= params.channels) { return; }
  outputValues[channel * params.time + time] =
    inputValues[time * params.storageChannels + channel] * params.scale;
}
`;
