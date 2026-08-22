const LIMIT_NAMES = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxStorageBuffersPerShaderStage",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupsPerDimension",
];

export async function probeWebGpu(scope) {
  const gpu = globalThis.navigator?.gpu;
  const base = {
    scope,
    userAgent: globalThis.navigator?.userAgent ?? null,
    secureContext: globalThis.isSecureContext,
    hasNavigatorGpu: Boolean(gpu),
  };
  if (!gpu) {
    return base;
  }

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    return { ...base, adapter: null };
  }

  const adapterFeatures = Array.from(adapter.features).sort();
  const languageFeatures = Array.from(gpu.wgslLanguageFeatures ?? []).sort();
  const requestedFeatures = ["shader-f16", "subgroups"].filter((feature) =>
    adapter.features.has(feature),
  );
  const device = await adapter.requestDevice({ requiredFeatures: requestedFeatures });
  try {
    return {
      ...base,
      adapter: adapterInfo(adapter),
      adapterFeatures,
      languageFeatures,
      limits: selectedLimits(adapter.limits),
      requestedFeatures,
      basicCompute: await runBasicCompute(device),
      shaderF16: await probeShaderF16(device, requestedFeatures),
      packedIntegerDot: await probePackedIntegerDot(device, languageFeatures),
      subgroups: await probeSubgroups(device, requestedFeatures),
    };
  } finally {
    device.destroy();
  }
}

function adapterInfo(adapter) {
  const info = adapter.info;
  if (!info) {
    return null;
  }
  return {
    vendor: info.vendor ?? null,
    architecture: info.architecture ?? null,
    device: info.device ?? null,
    description: info.description ?? null,
    subgroupMinSize: info.subgroupMinSize ?? null,
    subgroupMaxSize: info.subgroupMaxSize ?? null,
    isFallbackAdapter: info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
  };
}

function selectedLimits(limits) {
  return Object.fromEntries(LIMIT_NAMES.map((name) => [name, limits[name] ?? null]));
}

async function runBasicCompute(device) {
  const input = new Float32Array([1, 2, 3, 4]);
  const storage = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: input.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  device.queue.writeBuffer(storage, 0, input);

  const shader = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> values: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        if (id.x < 4u) {
          values[id.x] = values[id.x] * 2.0 + 1.0;
        }
      }
    `,
  });
  const compilation = await compilationMessages(shader);
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: storage } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(storage, 0, readback, 0, input.byteLength);

  const started = performance.now();
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const elapsedMs = performance.now() - started;
  const values = Array.from(new Float32Array(readback.getMappedRange().slice(0)));
  readback.unmap();
  storage.destroy();
  readback.destroy();
  return {
    ok: values.join(",") === "3,5,7,9",
    values,
    elapsedMs,
    compilation,
  };
}

async function probeShaderF16(device, requestedFeatures) {
  if (!requestedFeatures.includes("shader-f16")) {
    return { supported: false };
  }
  return compileOnly(
    device,
    `
      enable f16;
      @compute @workgroup_size(1)
      fn main() {
        let value = f16(1.0);
        _ = value;
      }
    `,
  );
}

async function probePackedIntegerDot(device, languageFeatures) {
  if (!languageFeatures.includes("packed_4x8_integer_dot_product")) {
    return { supported: false };
  }
  return compileOnly(
    device,
    `
      requires packed_4x8_integer_dot_product;
      @compute @workgroup_size(1)
      fn main() {
        let value = dot4I8Packed(0x01010101u, 0x02020202u);
        _ = value;
      }
    `,
  );
}

async function probeSubgroups(device, requestedFeatures) {
  if (!requestedFeatures.includes("subgroups")) {
    return { supported: false };
  }
  return compileOnly(
    device,
    `
      enable subgroups;
      @compute @workgroup_size(1)
      fn main(@builtin(subgroup_size) size: u32) {
        _ = size;
      }
    `,
  );
}

async function compileOnly(device, code) {
  const shader = device.createShaderModule({ code });
  const messages = await compilationMessages(shader);
  const errors = messages.filter((message) => message.type === "error");
  if (errors.length === 0) {
    device.createComputePipeline({
      layout: "auto",
      compute: { module: shader, entryPoint: "main" },
    });
  }
  return { supported: errors.length === 0, messages };
}

async function compilationMessages(shader) {
  if (typeof shader.getCompilationInfo !== "function") {
    return [];
  }
  const info = await shader.getCompilationInfo();
  return Array.from(info.messages, (message) => ({
    type: message.type,
    message: message.message,
    lineNum: message.lineNum,
    linePos: message.linePos,
  }));
}
