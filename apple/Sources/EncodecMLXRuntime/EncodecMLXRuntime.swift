import Foundation
import MLX
import MLXNN
import CEncodecMLXBridge

public enum EncodecMLXRuntimeDefaults {
    public static var frameBatchSize: Int {
        #if os(iOS)
        1
        #else
        8
        #endif
    }
}

public struct EncodecFrameMetadata: Decodable, Sendable {
    public let schemaVersion: Int
    public let modelName: String
    public let bandwidthKbps: Double
    public let sampleRate: Int
    public let channels: Int
    public let segmentSamples: Int
    public let segmentStride: Int
    public let normalize: Bool
    public let numCodebooks: Int
    public let frameLength: Int
    public let bitsPerCodebook: Int
    public let codebookCardinality: Int

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelName = "model_name"
        case bandwidthKbps = "bandwidth_kbps"
        case sampleRate = "sample_rate"
        case channels
        case segmentSamples = "segment_samples"
        case segmentStride = "segment_stride"
        case normalize
        case numCodebooks = "num_codebooks"
        case frameLength = "frame_length"
        case bitsPerCodebook = "bits_per_codebook"
        case codebookCardinality = "codebook_cardinality"
    }

    static func load(from bundleURL: URL) throws -> EncodecFrameMetadata {
        let url = bundleURL.appendingPathComponent("bundle.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EncodecMLXRuntimeError.missingFile(url)
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

public struct EncodecMLXWeightManifest: Decodable, Sendable {
    public let schemaVersion: Int
    public let format: String
    public let sourceBundle: String
    public let sourceBundleName: String
    public let models: [String: Model]

    public struct Model: Decodable, Sendable {
        public let sourceModel: String
        public let safetensors: String
        public let initializerCount: Int
        public let parameterCount: Int
        public let opHistogram: [String: Int]
        public let inputs: [ValueInfo]
        public let outputs: [ValueInfo]
        public let tensors: [Tensor]
        public let sha256: String
    }

    public struct ValueInfo: Decodable, Sendable {
        public let name: String
        public let dtype: String
        public let shape: [String]
    }

    public struct Tensor: Decodable, Sendable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
    }

    static func load(from bundleURL: URL) throws -> EncodecMLXWeightManifest {
        let url = bundleURL.appendingPathComponent("mlx-manifest.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EncodecMLXRuntimeError.missingFile(url)
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

public struct EncodecMLXRuntimeSummary: Sendable {
    public let modelName: String
    public let bandwidthKbps: Double
    public let sampleRate: Int
    public let channels: Int
    public let numCodebooks: Int
    public let frameLength: Int
    public let encodeTensorCount: Int
    public let decodeTensorCount: Int
    public let encodeParameterCount: Int
    public let decodeParameterCount: Int
}

public struct EncodecNativeDecodedAudio: Sendable {
    /// Interleaved samples in frame-major order.
    public let samples: [Float]
    public let channels: Int
    public let frameCount: Int
}

public struct EncodecNativeStreamResult: Sendable {
    public let bytesWritten: Int
}

public enum EncodecMLXRuntimeError: Error, LocalizedError, Sendable {
    case missingFile(URL)
    case missingModel(String)
    case missingTensor(String)
    case emptyWeights(String)
    case unsupportedBundle(String)
    case nativeBridge(String)

    public var errorDescription: String? {
        switch self {
        case let .missingFile(url):
            return "Missing Encodec MLX bundle file: \(url.path)"
        case let .missingModel(name):
            return "Missing Encodec MLX model manifest entry: \(name)"
        case let .missingTensor(name):
            return "Missing Encodec MLX tensor: \(name)"
        case let .emptyWeights(name):
            return "Encodec MLX model has no loaded tensors: \(name)"
        case let .unsupportedBundle(detail):
            return detail
        case let .nativeBridge(detail):
            return detail
        }
    }
}

public protocol EncodecFrameBackend {
    var metadata: EncodecFrameMetadata { get }

    func encodeFrame(audio: MLXArray) throws -> (codes: MLXArray, scale: MLXArray)
    func decodeFrame(codes: MLXArray, scale: MLXArray) throws -> MLXArray
}

public final class MLXEncodecFrameBackend: EncodecFrameBackend {
    public let metadata: EncodecFrameMetadata
    public let manifest: EncodecMLXWeightManifest

    private let encodeTensorCount: Int
    private let decodeTensorCount: Int
    private let encoder: MLXEncodecFrameEncoder
    private let decoder: MLXEncodecFrameDecoder
    private let compiledEncoder: (@Sendable ([MLXArray]) -> [MLXArray])?
    private let compiledDecoder: (@Sendable ([MLXArray]) -> [MLXArray])?

    public init(bundleURL: URL) throws {
        self.metadata = try EncodecFrameMetadata.load(from: bundleURL)
        self.manifest = try EncodecMLXWeightManifest.load(from: bundleURL)

        guard let encodeModel = manifest.models["encode_frame"] else {
            throw EncodecMLXRuntimeError.missingModel("encode_frame")
        }
        guard let decodeModel = manifest.models["decode_frame"] else {
            throw EncodecMLXRuntimeError.missingModel("decode_frame")
        }

        let encodeWeights = try Self.loadWeightArrays(
            from: bundleURL.appendingPathComponent(encodeModel.safetensors)
        )
        let decodeWeights = try Self.loadWeightArrays(
            from: bundleURL.appendingPathComponent(decodeModel.safetensors)
        )

        guard !encodeWeights.isEmpty else {
            throw EncodecMLXRuntimeError.emptyWeights("encode_frame")
        }
        guard !decodeWeights.isEmpty else {
            throw EncodecMLXRuntimeError.emptyWeights("decode_frame")
        }
        self.encodeTensorCount = encodeWeights.count
        self.decodeTensorCount = decodeWeights.count

        let encoder = try MLXEncodecFrameEncoder(metadata: metadata, weights: encodeWeights)
        let decoder = try MLXEncodecFrameDecoder(metadata: metadata, weights: decodeWeights)
        self.encoder = encoder
        self.decoder = decoder

        if ProcessInfo.processInfo.environment["BITNEEDLE_MLX_COMPILE"] == "1" {
            self.compiledEncoder = compile { inputs in
                let encoded = try! encoder.encodeFrame(audio: inputs[0])
                return [encoded.codes, encoded.scale]
            }
            self.compiledDecoder = compile { inputs in
                [try! decoder.decodeFrame(codes: inputs[0], scale: inputs[1])]
            }
        } else {
            self.compiledEncoder = nil
            self.compiledDecoder = nil
        }
    }

    private static func loadWeightArrays(from url: URL) throws -> [String: MLXArray] {
        let resolvedURL = url.standardizedFileURL.resolvingSymlinksInPath()
        guard FileManager.default.fileExists(atPath: resolvedURL.path) else {
            throw EncodecMLXRuntimeError.missingFile(resolvedURL)
        }
        return try loadArrays(data: Data(contentsOf: resolvedURL))
    }

    public var summary: EncodecMLXRuntimeSummary {
        EncodecMLXRuntimeSummary(
            modelName: metadata.modelName,
            bandwidthKbps: metadata.bandwidthKbps,
            sampleRate: metadata.sampleRate,
            channels: metadata.channels,
            numCodebooks: metadata.numCodebooks,
            frameLength: metadata.frameLength,
            encodeTensorCount: encodeTensorCount,
            decodeTensorCount: decodeTensorCount,
            encodeParameterCount: manifest.models["encode_frame"]?.parameterCount ?? 0,
            decodeParameterCount: manifest.models["decode_frame"]?.parameterCount ?? 0
        )
    }

    public func encodeFrame(audio: MLXArray) throws -> (codes: MLXArray, scale: MLXArray) {
        if let compiledEncoder {
            let encoded = compiledEncoder([audio])
            return (encoded[0], encoded[1])
        }
        return try encoder.encodeFrame(audio: audio)
    }

    public func decodeFrame(codes: MLXArray, scale: MLXArray) throws -> MLXArray {
        if let compiledDecoder {
            return compiledDecoder([codes, scale])[0]
        }
        return try decoder.decodeFrame(codes: codes, scale: scale)
    }
}

public final class MLXEncodecNativePipeline {
    private let bundleURL: URL
    private let backend: MLXEncodecFrameBackend

    public init(bundleURL: URL) throws {
        self.bundleURL = bundleURL
        self.backend = try MLXEncodecFrameBackend(bundleURL: bundleURL)
    }

    public var summary: EncodecMLXRuntimeSummary {
        backend.summary
    }

    /// Evaluates one encoder and decoder batch to load weights and compile GPU kernels.
    public func prewarm(
        frameBatchSize: Int = EncodecMLXRuntimeDefaults.frameBatchSize
    ) throws {
        let batch = max(frameBatchSize, 1)
        let audio = zeros(
            [batch, backend.metadata.channels, backend.metadata.segmentSamples],
            type: Float.self
        )
        let encoded = try backend.encodeFrame(audio: audio)
        let codes = encoded.codes.asType(Int64.self)
        eval(codes, encoded.scale)

        let decoded = try backend.decodeFrame(codes: codes, scale: encoded.scale)
        eval(decoded)
    }

    public func decodeEcdc(
        _ payload: Data,
        frameBatchSize: Int = EncodecMLXRuntimeDefaults.frameBatchSize
    ) throws -> EncodecNativeDecodedAudio {
        let callbackBox = MLXNativeFrameCallbackBox(backend: backend)
        let callbacks = callbackBox.callbacks()
        let result = withExtendedLifetime(callbackBox) {
            bundleURL.path.withCString { bundlePath in
                payload.withUnsafeBytes { payloadBuffer in
                    let payloadBytes = payloadBuffer.bindMemory(to: UInt8.self)
                    return encodec_rs_mlx_decode_ecdc_interleaved(
                        bundlePath,
                        payloadBytes.baseAddress,
                        payloadBytes.count,
                        max(frameBatchSize, 1),
                        callbacks
                    )
                }
            }
        }

        guard result.ok else {
            throw EncodecMLXRuntimeError.nativeBridge(
                Self.bridgeError(result.error, callbackError: callbackBox.lastError)
            )
        }
        guard let ptr = result.ptr else {
            throw EncodecMLXRuntimeError.nativeBridge("native decode returned a null audio buffer")
        }
        defer { encodec_rs_mlx_free_audio(ptr, result.len) }

        let samples = Array(UnsafeBufferPointer(start: ptr, count: result.len))
        return EncodecNativeDecodedAudio(
            samples: samples,
            channels: result.channels,
            frameCount: result.samples
        )
    }

    /// Decodes ECDC into contiguous channel-major f32le samples.
    public func decodeEcdcToPlanarF32File(
        _ payload: Data,
        outputURL: URL,
        progressURL: URL? = nil,
        frameBatchSize: Int = EncodecMLXRuntimeDefaults.frameBatchSize
    ) throws -> EncodecNativeStreamResult {
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if let progressURL {
            try FileManager.default.createDirectory(
                at: progressURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        }

        let callbackBox = MLXNativeFrameCallbackBox(backend: backend)
        let callbacks = callbackBox.callbacks()
        let result = withExtendedLifetime(callbackBox) {
            bundleURL.path.withCString { bundlePath in
                outputURL.path.withCString { outputPath in
                    let invoke = { (progressPath: UnsafePointer<CChar>?) in
                        payload.withUnsafeBytes { payloadBuffer in
                            let payloadBytes = payloadBuffer.bindMemory(to: UInt8.self)
                            return encodec_rs_mlx_decode_ecdc_planar_f32le_to_path(
                                bundlePath,
                                payloadBytes.baseAddress,
                                payloadBytes.count,
                                max(frameBatchSize, 1),
                                outputPath,
                                progressPath,
                                callbacks
                            )
                        }
                    }
                    if let progressURL {
                        return progressURL.path.withCString { progressPath in
                            invoke(progressPath)
                        }
                    }
                    return invoke(nil)
                }
            }
        }

        guard result.ok else {
            throw EncodecMLXRuntimeError.nativeBridge(
                Self.bridgeError(result.error, callbackError: callbackBox.lastError)
            )
        }
        return EncodecNativeStreamResult(bytesWritten: result.len)
    }

    public func encodeEcdc(
        samples: [Float],
        channels: Int,
        useLM: Bool = true,
        frameBatchSize: Int = EncodecMLXRuntimeDefaults.frameBatchSize,
        chunkMilliseconds: Double? = nil,
        chunkCRC: Bool = true
    ) throws -> Data {
        guard channels > 0 else {
            throw EncodecMLXRuntimeError.nativeBridge("channel count must be positive")
        }
        guard samples.count % channels == 0 else {
            throw EncodecMLXRuntimeError.nativeBridge(
                "interleaved sample count \(samples.count) is not divisible by \(channels) channels"
            )
        }

        let callbackBox = MLXNativeFrameCallbackBox(backend: backend)
        let callbacks = callbackBox.callbacks()
        let frames = samples.count / channels
        let result = withExtendedLifetime(callbackBox) {
            bundleURL.path.withCString { bundlePath in
                samples.withUnsafeBufferPointer { sampleBuffer in
                    encodec_rs_mlx_encode_ecdc_interleaved(
                        bundlePath,
                        sampleBuffer.baseAddress,
                        channels,
                        frames,
                        useLM,
                        max(frameBatchSize, 1),
                        chunkCRC,
                        chunkMilliseconds ?? 0.0,
                        chunkMilliseconds != nil,
                        callbacks
                    )
                }
            }
        }

        guard result.ok else {
            throw EncodecMLXRuntimeError.nativeBridge(
                Self.bridgeError(result.error, callbackError: callbackBox.lastError)
            )
        }
        guard let ptr = result.ptr else {
            throw EncodecMLXRuntimeError.nativeBridge("native encode returned a null byte buffer")
        }
        defer { encodec_rs_mlx_free_bytes(ptr, result.len) }

        return Data(bytes: ptr, count: result.len)
    }

    public func encodeEcdcStreaming(
        samples: [Float],
        channels: Int,
        outputURL: URL,
        progressURL: URL? = nil,
        useLM: Bool = true,
        frameBatchSize: Int = EncodecMLXRuntimeDefaults.frameBatchSize,
        chunkMilliseconds: Double? = nil,
        chunkCRC: Bool = true
    ) throws -> EncodecNativeStreamResult {
        guard channels > 0 else {
            throw EncodecMLXRuntimeError.nativeBridge("channel count must be positive")
        }
        guard samples.count % channels == 0 else {
            throw EncodecMLXRuntimeError.nativeBridge(
                "interleaved sample count \(samples.count) is not divisible by \(channels) channels"
            )
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if let progressURL {
            try FileManager.default.createDirectory(
                at: progressURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        }

        let callbackBox = MLXNativeFrameCallbackBox(backend: backend)
        let callbacks = callbackBox.callbacks()
        let frames = samples.count / channels
        let result = withExtendedLifetime(callbackBox) {
            bundleURL.path.withCString { bundlePath in
                outputURL.path.withCString { outputPath in
                    let invoke = { (progressPath: UnsafePointer<CChar>?) in
                        samples.withUnsafeBufferPointer { sampleBuffer in
                            encodec_rs_mlx_encode_ecdc_interleaved_stream_to_path(
                                bundlePath,
                                sampleBuffer.baseAddress,
                                channels,
                                frames,
                                useLM,
                                max(frameBatchSize, 1),
                                chunkCRC,
                                chunkMilliseconds ?? 0.0,
                                chunkMilliseconds != nil,
                                outputPath,
                                progressPath,
                                callbacks
                            )
                        }
                    }
                    if let progressURL {
                        return progressURL.path.withCString { progressPath in
                            invoke(progressPath)
                        }
                    }
                    return invoke(nil)
                }
            }
        }

        guard result.ok else {
            throw EncodecMLXRuntimeError.nativeBridge(
                Self.bridgeError(result.error, callbackError: callbackBox.lastError)
            )
        }
        return EncodecNativeStreamResult(bytesWritten: result.len)
    }

    private static func consumeError(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
        guard let pointer else {
            return "native bridge failed without an error message"
        }
        defer { encodec_rs_mlx_free_string(pointer) }
        return String(cString: pointer)
    }

    private static func bridgeError(_ pointer: UnsafeMutablePointer<CChar>?, callbackError: String?) -> String {
        let bridgeError = consumeError(pointer)
        guard let callbackError else {
            return bridgeError
        }
        return "\(bridgeError): \(callbackError)"
    }
}

public func encodecInterleavedToPlanar(_ samples: [Float], channels: Int) throws -> [Float] {
    guard channels > 0 else {
        throw EncodecMLXRuntimeError.nativeBridge("channel count must be positive")
    }
    guard samples.count % channels == 0 else {
        throw EncodecMLXRuntimeError.nativeBridge(
            "interleaved sample count \(samples.count) is not divisible by \(channels) channels"
        )
    }

    let frames = samples.count / channels
    guard channels > 1 else {
        return samples
    }

    var planar = Array(repeating: Float(0), count: samples.count)
    for frame in 0 ..< frames {
        for channel in 0 ..< channels {
            planar[channel * frames + frame] = samples[frame * channels + channel]
        }
    }
    return planar
}

public func encodecPlanarToInterleaved(_ samples: [Float], channels: Int, frames: Int) throws -> [Float] {
    guard channels > 0, frames >= 0 else {
        throw EncodecMLXRuntimeError.nativeBridge("cannot convert invalid planar audio shape")
    }
    guard samples.count == channels * frames else {
        throw EncodecMLXRuntimeError.nativeBridge(
            "planar sample count \(samples.count) does not match \(channels) channels and \(frames) frames"
        )
    }
    guard channels > 1 else {
        return samples
    }

    var interleaved = Array(repeating: Float(0), count: samples.count)
    for frame in 0 ..< frames {
        for channel in 0 ..< channels {
            interleaved[frame * channels + channel] = samples[channel * frames + frame]
        }
    }
    return interleaved
}

private final class MLXNativeFrameCallbackBox {
    let backend: MLXEncodecFrameBackend
    var lastError: String?

    init(backend: MLXEncodecFrameBackend) {
        self.backend = backend
    }

    func callbacks() -> EncodecRsMlxFrameCallbacks {
        EncodecRsMlxFrameCallbacks(
            user_data: Unmanaged.passUnretained(self).toOpaque(),
            encode_frame: mlxNativeEncodeFrameCallback,
            decode_frame: mlxNativeDecodeFrameCallback
        )
    }

    func record(_ error: Error) {
        lastError = error.localizedDescription
    }

    func record(_ message: String) {
        lastError = message
    }
}

private func mlxCopyData(
    _ data: Data,
    to destination: UnsafeMutableRawPointer,
    byteCount: Int
) -> Bool {
    guard data.count == byteCount else {
        return false
    }
    guard byteCount > 0 else {
        return true
    }
    return data.withUnsafeBytes { source in
        guard let baseAddress = source.baseAddress else {
            return false
        }
        destination.copyMemory(from: baseAddress, byteCount: byteCount)
        return true
    }
}

// `asyncEval` measures host-side graph traversal/submission. The following
// synchronization bucket includes any outstanding device execution; neither is
// presented as isolated GPU kernel time.
private let mlxNativeEncodeFrameCallback: EncodecRsMlxEncodeFrameFn = { userData, audio, batch, channels, samples, codesOut, codesLen, scalesOut, scalesLen in
    guard let userData, let audio, let codesOut, let scalesOut else {
        return -1
    }

    do {
        let profileStarted = mlxNativeProfileStarted()
        let box = Unmanaged<MLXNativeFrameCallbackBox>.fromOpaque(userData).takeUnretainedValue()
        let audioCount = batch * channels * samples
        let audioArray = MLXArray(
            UnsafeBufferPointer(start: audio, count: audioCount),
            [batch, channels, samples]
        )
        let inputDone = mlxNativeProfileNow()
        let encoded = try box.backend.encodeFrame(audio: audioArray)
        let callbackCodes = encoded.codes.asType(Int64.self)
        let graphConstructionDone = mlxNativeProfileNow()
        let executionSubmitted: UInt64
        let synchronizationDone: UInt64
        if mlxNativeProfileEnabled {
            asyncEval([callbackCodes, encoded.scale])
            executionSubmitted = mlxNativeProfileNow()
            Stream.gpu.synchronize()
            synchronizationDone = mlxNativeProfileNow()
        } else {
            executionSubmitted = graphConstructionDone
            synchronizationDone = graphConstructionDone
        }
        let codeData = callbackCodes.asData(access: .noCopyIfContiguous).data
        let scaleData = encoded.scale.asData(access: .noCopyIfContiguous).data
        let readbackDone = mlxNativeProfileNow()
        let codeBytes = codesLen * MemoryLayout<Int64>.stride
        let scaleBytes = scalesLen * MemoryLayout<Float>.stride
        guard codeData.count == codeBytes, scaleData.count == scaleBytes else {
            box.record(
                "encode callback produced \(codeData.count) code bytes and \(scaleData.count) scale bytes, expected \(codeBytes) and \(scaleBytes)"
            )
            return -2
        }
        guard mlxCopyData(codeData, to: UnsafeMutableRawPointer(codesOut), byteCount: codeBytes),
              mlxCopyData(scaleData, to: UnsafeMutableRawPointer(scalesOut), byteCount: scaleBytes)
        else {
            box.record("encode callback could not copy contiguous MLX output")
            return -2
        }
        let ffiCopyDone = mlxNativeProfileNow()
        mlxNativeProfilePrint(
            "encode_callback batch=\(batch) " +
            "host_input_copy_ms=\(mlxNativeProfileMillis(profileStarted, inputDone)) " +
            "graph_construction_ms=\(mlxNativeProfileMillis(inputDone, graphConstructionDone)) " +
            "execution_submit_ms=\(mlxNativeProfileMillis(graphConstructionDone, executionSubmitted)) " +
            "synchronization_wait_ms=\(mlxNativeProfileMillis(executionSubmitted, synchronizationDone)) " +
            "host_readback_ms=\(mlxNativeProfileMillis(synchronizationDone, readbackDone)) " +
            "ffi_output_copy_ms=\(mlxNativeProfileMillis(readbackDone, ffiCopyDone)) " +
            "total_ms=\(mlxNativeProfileMillis(profileStarted, ffiCopyDone))"
        )
        return 0
    } catch {
        let box = Unmanaged<MLXNativeFrameCallbackBox>.fromOpaque(userData).takeUnretainedValue()
        box.record(error)
        return -3
    }
}

private let mlxNativeDecodeFrameCallback: EncodecRsMlxDecodeFrameFn = { userData, codes, batch, codebooks, frames, scales, scalesLen, audioOut, audioLen in
    guard let userData, let codes, let scales, let audioOut else {
        return -1
    }

    do {
        let profileStarted = mlxNativeProfileStarted()
        let box = Unmanaged<MLXNativeFrameCallbackBox>.fromOpaque(userData).takeUnretainedValue()
        let codeCount = batch * codebooks * frames
        let codeValues = UnsafeBufferPointer(start: codes, count: codeCount).map(Int32.init)
        let codeArray = MLXArray(codeValues, [batch, codebooks, frames])
        let scaleArray = MLXArray(
            UnsafeBufferPointer(start: scales, count: scalesLen),
            [batch, max(scalesLen / max(batch, 1), 1)]
        )
        let inputDone = mlxNativeProfileNow()
        let decoded = try box.backend.decodeFrame(codes: codeArray, scale: scaleArray)
        let graphConstructionDone = mlxNativeProfileNow()
        let executionSubmitted: UInt64
        let synchronizationDone: UInt64
        if mlxNativeProfileEnabled {
            asyncEval([decoded])
            executionSubmitted = mlxNativeProfileNow()
            Stream.gpu.synchronize()
            synchronizationDone = mlxNativeProfileNow()
        } else {
            executionSubmitted = graphConstructionDone
            synchronizationDone = graphConstructionDone
        }
        let audioData = decoded.asData(access: .noCopyIfContiguous).data
        let readbackDone = mlxNativeProfileNow()
        let audioBytes = audioLen * MemoryLayout<Float>.stride
        guard audioData.count == audioBytes else {
            box.record(
                "decode callback produced \(audioData.count) bytes, expected \(audioBytes)"
            )
            return -2
        }
        guard mlxCopyData(
            audioData,
            to: UnsafeMutableRawPointer(audioOut),
            byteCount: audioBytes
        ) else {
            box.record("decode callback could not copy contiguous MLX output")
            return -2
        }
        let ffiCopyDone = mlxNativeProfileNow()
        mlxNativeProfilePrint(
            "decode_callback batch=\(batch) " +
            "host_input_copy_ms=\(mlxNativeProfileMillis(profileStarted, inputDone)) " +
            "graph_construction_ms=\(mlxNativeProfileMillis(inputDone, graphConstructionDone)) " +
            "execution_submit_ms=\(mlxNativeProfileMillis(graphConstructionDone, executionSubmitted)) " +
            "synchronization_wait_ms=\(mlxNativeProfileMillis(executionSubmitted, synchronizationDone)) " +
            "host_readback_ms=\(mlxNativeProfileMillis(synchronizationDone, readbackDone)) " +
            "ffi_output_copy_ms=\(mlxNativeProfileMillis(readbackDone, ffiCopyDone)) " +
            "total_ms=\(mlxNativeProfileMillis(profileStarted, ffiCopyDone))"
        )
        return 0
    } catch {
        let box = Unmanaged<MLXNativeFrameCallbackBox>.fromOpaque(userData).takeUnretainedValue()
        box.record(error)
        return -3
    }
}

private let mlxNativeProfileEnabled = ProcessInfo.processInfo.environment["BITNEEDLE_MLX_PROFILE"] != nil

private func mlxNativeProfileStarted() -> UInt64 {
    mlxNativeProfileEnabled ? DispatchTime.now().uptimeNanoseconds : 0
}

private func mlxNativeProfileNow() -> UInt64 {
    mlxNativeProfileEnabled ? DispatchTime.now().uptimeNanoseconds : 0
}

private func mlxNativeProfileMillis(_ start: UInt64, _ end: UInt64) -> String {
    guard mlxNativeProfileEnabled, end >= start else {
        return "0.000"
    }
    return String(format: "%.3f", Double(end - start) / 1_000_000)
}

private func mlxNativeProfilePrint(_ message: String) {
    guard mlxNativeProfileEnabled else {
        return
    }
    print("mlx_profile: \(message)")
}

private struct MLXEncodecFrameEncoder {
    private struct Norm {
        let scale: MLXArray
        let bias: MLXArray
    }

    private struct LstmLayer {
        let inputWeight: MLXArray
        let recurrentWeight: MLXArray
        let bias: MLXArray
    }

    private struct Quantizer {
        let columns: MLXArray
        let columnNorm: MLXArray
        let codebook: MLXArray
    }

    private let metadata: EncodecFrameMetadata
    private let convolutionWeights: [String: MLXArray]
    private let convolutionBiases: [String: MLXArray]
    private let norms: [Norm]
    private let lstmLayers: [LstmLayer]
    private let quantizers: [Quantizer]

    init(metadata: EncodecFrameMetadata, weights: [String: MLXArray]) throws {
        guard metadata.modelName == "encodec_48khz" else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode currently supports encodec_48khz bundles only."
            )
        }
        guard metadata.sampleRate == 48_000, metadata.channels == 2 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode currently expects 48 kHz stereo bundles."
            )
        }

        self.metadata = metadata
        self.convolutionWeights = Dictionary(
            uniqueKeysWithValues: weights.compactMap { name, value in
                guard name.hasPrefix("model.encoder."), name.hasSuffix(".conv.conv.weight") else {
                    return nil
                }
                return (name, value.transposed(0, 2, 1))
            }
        )
        self.convolutionBiases = Dictionary(
            uniqueKeysWithValues: weights.compactMap { name, value in
                guard name.hasPrefix("model.encoder."),
                      name.hasSuffix(".conv.conv.bias"),
                      value.shape.count == 1
                else {
                    return nil
                }
                return (name, value.reshaped([1, value.shape[0], 1]))
            }
        )

        let lstmNames = generatedTensorNames(weights: weights, prefix: "onnx::LSTM_")
        guard lstmNames.count == 6 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode expected 6 generated LSTM tensors, got \(lstmNames.count)."
            )
        }
        self.lstmLayers = try stride(from: 0, to: lstmNames.count, by: 3).map { index in
            let inputWeight = try Self.required(weights, lstmNames[index])[0, 0..., 0...]
            let recurrentWeight = try Self.required(weights, lstmNames[index + 1])[0, 0..., 0...]
            let rawBias = try Self.required(weights, lstmNames[index + 2])[0, 0...]
            let hiddenSize = inputWeight.shape[0] / 4
            return LstmLayer(
                inputWeight: inputWeight.T,
                recurrentWeight: recurrentWeight.T,
                bias: rawBias[..<(4 * hiddenSize)]
                    + rawBias[(4 * hiddenSize)..<(8 * hiddenSize)]
            )
        }

        let scaleNames = generatedTensorNames(weights: weights, prefix: "onnx::Mul_")
        let biasNames = try Self.normBiasNames(weights: weights)
        guard scaleNames.count >= 18 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode expected at least 18 GroupNorm scale tensors, got \(scaleNames.count)."
            )
        }
        self.norms = try (0 ..< 18).map { index in
            let scale = try Self.required(weights, scaleNames[index])
            let bias = try Self.required(weights, biasNames[index])
            return Norm(
                scale: scale.reshaped([1, scale.shape[0], 1]),
                bias: bias.reshaped([1, bias.shape[0], 1])
            )
        }

        let matMulNames = generatedTensorNames(weights: weights, prefix: "onnx::MatMul_")
        guard matMulNames.count >= metadata.numCodebooks else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode expected at least \(metadata.numCodebooks) quantizer MatMul tensors, got \(matMulNames.count)."
            )
        }
        self.quantizers = try Array(matMulNames.prefix(metadata.numCodebooks)).enumerated().map {
            index, name in
            let columns = try Self.required(weights, name)
            let directName = "model.quantizer.vq.layers.\(index)._codebook.embed"
            return Quantizer(
                columns: columns,
                columnNorm: (columns * columns).sum(axis: 0, keepDims: true),
                codebook: weights[directName] ?? columns.T
            )
        }

    }

    func encodeFrame(audio: MLXArray) throws -> (codes: MLXArray, scale: MLXArray) {
        guard audio.shape.count == 3 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode expected audio with shape [batch, channels, samples]."
            )
        }
        guard audio.shape[1] == metadata.channels, audio.shape[2] > 0 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode expected [batch, \(metadata.channels), samples>0] audio, got \(audio.shape)."
            )
        }

        let (normalized, scale) = normalize(audio)
        let embedding = try encoderNetwork(normalized)
        let codes = try residualVectorQuantizerEncode(embedding)
        return (codes, scale)
    }

    private func normalize(_ audio: MLXArray) -> (MLXArray, MLXArray) {
        guard metadata.normalize else {
            let scale = audio.mean(axes: [1, 2], keepDims: true) * 0.0 + 1.0
            return (audio, scale.reshaped([audio.shape[0], 1]))
        }

        let mono = audio.mean(axis: 1, keepDims: true)
        let scale = sqrt((mono * mono).mean(axis: 2, keepDims: true)) + 0.00000001
        return (audio / scale, scale.reshaped([audio.shape[0], 1]))
    }

    private func encoderNetwork(_ audio: MLXArray) throws -> MLXArray {
        var x = try conv1dNCT(
            audio,
            weight: "model.encoder.model.0.conv.conv.weight",
            bias: "model.encoder.model.0.conv.conv.bias",
            kernelSize: 7,
            stride: 1,
            norm: 0
        )

        x = try residualBlock(x, prefix: "model.encoder.model.1", normStart: 1)

        x = elu(x)
        x = try conv1dNCT(
            x,
            weight: "model.encoder.model.3.conv.conv.weight",
            bias: "model.encoder.model.3.conv.conv.bias",
            kernelSize: 4,
            stride: 2,
            norm: 4
        )
        x = try residualBlock(x, prefix: "model.encoder.model.4", normStart: 5)

        x = elu(x)
        x = try conv1dNCT(
            x,
            weight: "model.encoder.model.6.conv.conv.weight",
            bias: "model.encoder.model.6.conv.conv.bias",
            kernelSize: 8,
            stride: 4,
            norm: 8
        )
        x = try residualBlock(x, prefix: "model.encoder.model.7", normStart: 9)

        x = elu(x)
        x = try conv1dNCT(
            x,
            weight: "model.encoder.model.9.conv.conv.weight",
            bias: "model.encoder.model.9.conv.conv.bias",
            kernelSize: 10,
            stride: 5,
            norm: 12
        )
        x = try residualBlock(x, prefix: "model.encoder.model.10", normStart: 13)

        x = elu(x)
        x = try conv1dNCT(
            x,
            weight: "model.encoder.model.12.conv.conv.weight",
            bias: "model.encoder.model.12.conv.conv.bias",
            kernelSize: 16,
            stride: 8,
            norm: 16
        )

        x = try slstm(x)

        x = elu(x)
        return try conv1dNCT(
            x,
            weight: "model.encoder.model.15.conv.conv.weight",
            bias: "model.encoder.model.15.conv.conv.bias",
            kernelSize: 7,
            stride: 1,
            norm: 17
        )
    }

    private func residualBlock(_ input: MLXArray, prefix: String, normStart: Int) throws -> MLXArray {
        let shortcut = try conv1dNCT(
            input,
            weight: "\(prefix).shortcut.conv.conv.weight",
            bias: "\(prefix).shortcut.conv.conv.bias",
            kernelSize: 1,
            stride: 1,
            norm: normStart
        )

        var block = elu(input)
        block = try conv1dNCT(
            block,
            weight: "\(prefix).block.1.conv.conv.weight",
            bias: "\(prefix).block.1.conv.conv.bias",
            kernelSize: 3,
            stride: 1,
            norm: normStart + 1
        )
        block = elu(block)
        block = try conv1dNCT(
            block,
            weight: "\(prefix).block.3.conv.conv.weight",
            bias: "\(prefix).block.3.conv.conv.bias",
            kernelSize: 1,
            stride: 1,
            norm: normStart + 2
        )

        return shortcut + block
    }

    private func conv1dNCT(
        _ input: MLXArray,
        weight weightName: String,
        bias biasName: String,
        kernelSize: Int,
        stride: Int,
        norm: Int
    ) throws -> MLXArray {
        let paddingTotal = kernelSize - stride
        let paddingRight = paddingTotal / 2
        let paddingLeft = paddingTotal - paddingRight
        let paddedInput = reflectPad1d(input, left: paddingLeft, right: paddingRight)
        let nlc = paddedInput.transposed(0, 2, 1)
        guard let weight = convolutionWeights[weightName] else {
            throw EncodecMLXRuntimeError.missingTensor(weightName)
        }
        guard let bias = convolutionBiases[biasName] else {
            throw EncodecMLXRuntimeError.missingTensor(biasName)
        }
        var y = conv1d(nlc, weight, stride: stride).transposed(0, 2, 1)
        guard y.shape[1] == bias.shape[1] else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode convolution \(weightName) produced shape \(y.shape), but bias \(biasName) has shape \(bias.shape)."
            )
        }
        y = y + bias
        return groupNorm(y, norm: norm)
    }

    private func groupNorm(_ input: MLXArray, norm index: Int) -> MLXArray {
        let mean = input.mean(axes: [1, 2], keepDims: true)
        let variance = input.variance(axes: [1, 2], keepDims: true)
        let normalized = (input - mean) / sqrt(variance + 0.00001)
        return normalized * norms[index].scale + norms[index].bias
    }

    private func slstm(_ input: MLXArray) throws -> MLXArray {
        let nlc = input.transposed(0, 2, 1)
        let first = try onnxLstm(nlc, layer: lstmLayers[0])
        let second = try onnxLstm(first, layer: lstmLayers[1])
        return (second + nlc).transposed(0, 2, 1)
    }

    private func onnxLstm(_ input: MLXArray, layer: LstmLayer) throws -> MLXArray {
        let projectedInput = matmul(input, layer.inputWeight) + layer.bias

        var hidden: MLXArray?
        var cell: MLXArray?
        var outputs = [MLXArray]()
        outputs.reserveCapacity(input.shape[1])

        for index in 0 ..< input.shape[1] {
            var gates = projectedInput[0..., index, 0...]
            if let hidden {
                gates = gates + matmul(hidden, layer.recurrentWeight)
            }

            let pieces = gates.split(parts: 4, axis: -1)
            let inputGate = sigmoid(pieces[0])
            let outputGate = sigmoid(pieces[1])
            let forgetGate = sigmoid(pieces[2])
            let cellGate = tanh(pieces[3])

            if let previousCell = cell {
                cell = forgetGate * previousCell + inputGate * cellGate
            } else {
                cell = inputGate * cellGate
            }

            let nextHidden = outputGate * tanh(cell!)
            hidden = nextHidden
            outputs.append(nextHidden)
        }

        return stacked(outputs, axis: 1)
    }

    private func residualVectorQuantizerEncode(_ embedding: MLXArray) throws -> MLXArray {
        var residual = embedding.transposed(0, 2, 1)
        var codes = [MLXArray]()
        codes.reserveCapacity(metadata.numCodebooks)

        for index in 0 ..< metadata.numCodebooks {
            let quantizer = quantizers[index]
            let residualFlat = residual.reshaped([-1, residual.shape[2]])
            let residualNorm = (residualFlat * residualFlat).sum(axis: 1, keepDims: true)
            let distances = residualNorm - matmul(residualFlat * 2.0, quantizer.columns)
                + quantizer.columnNorm
            let indices = argMax(-distances, axis: -1).reshaped([residual.shape[0], residual.shape[1]])
            codes.append(indices)

            if index + 1 < metadata.numCodebooks {
                let quantized = quantizer.codebook.take(indices, axis: 0)
                residual = residual - quantized
            }
        }

        return stacked(codes, axis: 1)
    }

    private func reflectPad1d(_ input: MLXArray, left: Int, right: Int) -> MLXArray {
        var pieces = [MLXArray]()
        if left > 0 {
            pieces.append(input[0..., 0..., .stride(from: left, to: 0, by: -1)])
        }
        pieces.append(input)
        if right > 0 {
            let length = input.shape[2]
            pieces.append(input[0..., 0..., .stride(from: length - 2, to: length - right - 2, by: -1)])
        }
        return concatenated(pieces, axis: 2)
    }

    private static func required(_ weights: [String: MLXArray], _ name: String) throws -> MLXArray {
        guard let value = weights[name] else {
            throw EncodecMLXRuntimeError.missingTensor(name)
        }
        return value
    }

    private static func normBiasNames(weights: [String: MLXArray]) throws -> [String] {
        let biasNames = generatedTensorNames(weights: weights, prefix: "onnx::Add_")
        if biasNames.count >= 18 {
            return Array(biasNames.prefix(18))
        }

        if biasNames.count == 14 {
            return [
                biasNames[0],
                biasNames[1],
                biasNames[2],
                biasNames[1],
                biasNames[3],
                biasNames[4],
                biasNames[5],
                biasNames[4],
                biasNames[6],
                biasNames[7],
                biasNames[8],
                biasNames[7],
                biasNames[9],
                biasNames[10],
                biasNames[11],
                biasNames[10],
                biasNames[12],
                biasNames[13],
            ]
        }

        throw EncodecMLXRuntimeError.unsupportedBundle(
            "MLX encode expected 14 or at least 18 GroupNorm bias tensors, got \(biasNames.count)."
        )
    }
}

private struct MLXEncodecFrameDecoder {
    private struct Norm {
        let scale: MLXArray
        let bias: MLXArray
    }

    private struct LstmLayer {
        let inputWeight: MLXArray
        let recurrentWeight: MLXArray
        let bias: MLXArray
    }

    private let metadata: EncodecFrameMetadata
    private let convolutionWeights: [String: MLXArray]
    private let convolutionBiases: [String: MLXArray]
    private let codebooks: [MLXArray]
    private let norms: [Norm]
    private let lstmLayers: [LstmLayer]

    init(metadata: EncodecFrameMetadata, weights: [String: MLXArray]) throws {
        guard metadata.modelName == "encodec_48khz" else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode currently supports encodec_48khz bundles only."
            )
        }
        guard metadata.sampleRate == 48_000, metadata.channels == 2 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode currently expects 48 kHz stereo bundles."
            )
        }

        self.metadata = metadata
        self.convolutionWeights = Dictionary(
            uniqueKeysWithValues: weights.compactMap { name, value in
                guard name.hasPrefix("model.decoder.") else {
                    return nil
                }
                if name.hasSuffix(".conv.conv.weight") {
                    return (name, value.transposed(0, 2, 1))
                }
                if name.hasSuffix(".convtr.convtr.weight") {
                    return (name, value.transposed(1, 2, 0))
                }
                return nil
            }
        )
        self.convolutionBiases = Dictionary(
            uniqueKeysWithValues: weights.compactMap { name, value in
                guard name.hasPrefix("model.decoder."),
                      name.hasSuffix(".bias"),
                      value.shape.count == 1
                else {
                    return nil
                }
                return (name, value.reshaped([1, value.shape[0], 1]))
            }
        )
        self.codebooks = try (0 ..< metadata.numCodebooks).map { index in
            try Self.required(weights, "model.quantizer.vq.layers.\(index)._codebook.embed")
        }

        let lstmNames = generatedTensorNames(weights: weights, prefix: "onnx::LSTM_")
        guard lstmNames.count == 6 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode expected 6 generated LSTM tensors, got \(lstmNames.count)."
            )
        }
        self.lstmLayers = try stride(from: 0, to: lstmNames.count, by: 3).map { index in
            let inputWeight = try Self.required(weights, lstmNames[index])[0, 0..., 0...]
            let recurrentWeight = try Self.required(weights, lstmNames[index + 1])[0, 0..., 0...]
            let rawBias = try Self.required(weights, lstmNames[index + 2])[0, 0...]
            let hiddenSize = inputWeight.shape[0] / 4
            return LstmLayer(
                inputWeight: inputWeight.T,
                recurrentWeight: recurrentWeight.T,
                bias: rawBias[..<(4 * hiddenSize)]
                    + rawBias[(4 * hiddenSize)..<(8 * hiddenSize)]
            )
        }

        let scaleNames = generatedTensorNames(weights: weights, prefix: "onnx::Mul_")
        let biasNames = try Self.normBiasNames(weights: weights)
        guard scaleNames.count >= 18 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode expected at least 18 GroupNorm scale tensors, got \(scaleNames.count)."
            )
        }
        self.norms = try (0 ..< 18).map { index in
            let scale = try Self.required(weights, scaleNames[index])
            let bias = try Self.required(weights, biasNames[index])
            return Norm(
                scale: scale.reshaped([1, scale.shape[0], 1]),
                bias: bias.reshaped([1, bias.shape[0], 1])
            )
        }

    }

    func decodeFrame(codes: MLXArray, scale: MLXArray) throws -> MLXArray {
        guard codes.shape.count == 3 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode expected codes with shape [batch, codebooks, frames]."
            )
        }
        guard codes.shape[1] == metadata.numCodebooks else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode expected \(metadata.numCodebooks) codebooks, got \(codes.shape[1])."
            )
        }

        var x = try residualVectorQuantizerDecode(codes: codes)
        x = try decoderNetwork(x)
        return x * scale.reshaped([scale.shape[0], 1, 1])
    }

    private func residualVectorQuantizerDecode(codes: MLXArray) throws -> MLXArray {
        let batch = codes.shape[0]
        let frames = codes.shape[2]
        var z = zeros([batch, frames, 128], type: Float.self)

        for index in 0 ..< metadata.numCodebooks {
            let codebookIndices = codes[0..., index, 0...]
            z = z + codebooks[index].take(codebookIndices, axis: 0)
        }

        return z.transposed(0, 2, 1)
    }

    private func decoderNetwork(_ embedding: MLXArray) throws -> MLXArray {
        var x = try conv1dNCT(
            embedding,
            weight: "model.decoder.model.0.conv.conv.weight",
            bias: "model.decoder.model.0.conv.conv.bias",
            kernelSize: 7,
            norm: 0
        )

        x = try slstm(x)

        x = elu(x)
        x = try convTranspose1dNCT(
            x,
            weight: "model.decoder.model.3.convtr.convtr.weight",
            bias: "model.decoder.model.3.convtr.convtr.bias",
            kernelSize: 16,
            stride: 8,
            norm: 1
        )
        x = try residualBlock(x, prefix: "model.decoder.model.4", normStart: 2)

        x = elu(x)
        x = try convTranspose1dNCT(
            x,
            weight: "model.decoder.model.6.convtr.convtr.weight",
            bias: "model.decoder.model.6.convtr.convtr.bias",
            kernelSize: 10,
            stride: 5,
            norm: 5
        )
        x = try residualBlock(x, prefix: "model.decoder.model.7", normStart: 6)

        x = elu(x)
        x = try convTranspose1dNCT(
            x,
            weight: "model.decoder.model.9.convtr.convtr.weight",
            bias: "model.decoder.model.9.convtr.convtr.bias",
            kernelSize: 8,
            stride: 4,
            norm: 9
        )
        x = try residualBlock(x, prefix: "model.decoder.model.10", normStart: 10)

        x = elu(x)
        x = try convTranspose1dNCT(
            x,
            weight: "model.decoder.model.12.convtr.convtr.weight",
            bias: "model.decoder.model.12.convtr.convtr.bias",
            kernelSize: 4,
            stride: 2,
            norm: 13
        )
        x = try residualBlock(x, prefix: "model.decoder.model.13", normStart: 14)

        x = elu(x)
        return try conv1dNCT(
            x,
            weight: "model.decoder.model.15.conv.conv.weight",
            bias: "model.decoder.model.15.conv.conv.bias",
            kernelSize: 7,
            norm: 17
        )
    }

    private func residualBlock(_ input: MLXArray, prefix: String, normStart: Int) throws -> MLXArray {
        let shortcut = try conv1dNCT(
            input,
            weight: "\(prefix).shortcut.conv.conv.weight",
            bias: "\(prefix).shortcut.conv.conv.bias",
            kernelSize: 1,
            norm: normStart
        )

        var block = elu(input)
        block = try conv1dNCT(
            block,
            weight: "\(prefix).block.1.conv.conv.weight",
            bias: "\(prefix).block.1.conv.conv.bias",
            kernelSize: 3,
            norm: normStart + 1
        )
        block = elu(block)
        block = try conv1dNCT(
            block,
            weight: "\(prefix).block.3.conv.conv.weight",
            bias: "\(prefix).block.3.conv.conv.bias",
            kernelSize: 1,
            norm: normStart + 2
        )

        return shortcut + block
    }

    private func conv1dNCT(
        _ input: MLXArray,
        weight weightName: String,
        bias biasName: String,
        kernelSize: Int,
        norm: Int
    ) throws -> MLXArray {
        let paddedInput = reflectPad1d(input, left: (kernelSize - 1) - ((kernelSize - 1) / 2), right: (kernelSize - 1) / 2)
        let nlc = paddedInput.transposed(0, 2, 1)
        guard let weight = convolutionWeights[weightName] else {
            throw EncodecMLXRuntimeError.missingTensor(weightName)
        }
        guard let bias = convolutionBiases[biasName] else {
            throw EncodecMLXRuntimeError.missingTensor(biasName)
        }
        var y = conv1d(nlc, weight).transposed(0, 2, 1)
        y = y + bias
        return groupNorm(y, norm: norm)
    }

    private func convTranspose1dNCT(
        _ input: MLXArray,
        weight weightName: String,
        bias biasName: String,
        kernelSize: Int,
        stride: Int,
        norm: Int
    ) throws -> MLXArray {
        let nlc = input.transposed(0, 2, 1)
        guard let weight = convolutionWeights[weightName] else {
            throw EncodecMLXRuntimeError.missingTensor(weightName)
        }
        guard let bias = convolutionBiases[biasName] else {
            throw EncodecMLXRuntimeError.missingTensor(biasName)
        }
        var y = convTransposed1d(nlc, weight, stride: stride).transposed(0, 2, 1)
        y = y + bias
        y = groupNorm(y, norm: norm)

        let paddingTotal = kernelSize - stride
        let paddingRight = paddingTotal / 2
        let paddingLeft = paddingTotal - paddingRight
        let end = y.shape[2] - paddingRight
        return y[0..., 0..., paddingLeft ..< end]
    }

    private func groupNorm(_ input: MLXArray, norm index: Int) -> MLXArray {
        let mean = input.mean(axes: [1, 2], keepDims: true)
        let variance = input.variance(axes: [1, 2], keepDims: true)
        let normalized = (input - mean) / sqrt(variance + 0.00001)
        return normalized * norms[index].scale + norms[index].bias
    }

    private func slstm(_ input: MLXArray) throws -> MLXArray {
        let nlc = input.transposed(0, 2, 1)
        let first = try onnxLstm(nlc, layer: lstmLayers[0])
        let second = try onnxLstm(first, layer: lstmLayers[1])
        return (second + nlc).transposed(0, 2, 1)
    }

    private func onnxLstm(_ input: MLXArray, layer: LstmLayer) throws -> MLXArray {
        let projectedInput = matmul(input, layer.inputWeight) + layer.bias

        var hidden: MLXArray?
        var cell: MLXArray?
        var outputs = [MLXArray]()
        outputs.reserveCapacity(input.shape[1])

        for index in 0 ..< input.shape[1] {
            var gates = projectedInput[0..., index, 0...]
            if let hidden {
                gates = gates + matmul(hidden, layer.recurrentWeight)
            }

            let pieces = gates.split(parts: 4, axis: -1)
            let inputGate = sigmoid(pieces[0])
            let outputGate = sigmoid(pieces[1])
            let forgetGate = sigmoid(pieces[2])
            let cellGate = tanh(pieces[3])

            if let previousCell = cell {
                cell = forgetGate * previousCell + inputGate * cellGate
            } else {
                cell = inputGate * cellGate
            }

            let nextHidden = outputGate * tanh(cell!)
            hidden = nextHidden
            outputs.append(nextHidden)
        }

        return stacked(outputs, axis: 1)
    }

    private func reflectPad1d(_ input: MLXArray, left: Int, right: Int) -> MLXArray {
        var pieces = [MLXArray]()
        if left > 0 {
            pieces.append(input[0..., 0..., .stride(from: left, to: 0, by: -1)])
        }
        pieces.append(input)
        if right > 0 {
            let length = input.shape[2]
            pieces.append(input[0..., 0..., .stride(from: length - 2, to: length - right - 2, by: -1)])
        }
        return concatenated(pieces, axis: 2)
    }

    private static func required(_ weights: [String: MLXArray], _ name: String) throws -> MLXArray {
        guard let value = weights[name] else {
            throw EncodecMLXRuntimeError.missingTensor(name)
        }
        return value
    }

    private static func normBiasNames(weights: [String: MLXArray]) throws -> [String] {
        let biasNames = generatedTensorNames(weights: weights, prefix: "onnx::Add_")
        if biasNames.count >= 18 {
            return Array(biasNames.prefix(18))
        }

        if biasNames.count == 14 {
            return [
                biasNames[0],
                biasNames[1],
                biasNames[2],
                biasNames[3],
                biasNames[2],
                biasNames[4],
                biasNames[5],
                biasNames[6],
                biasNames[5],
                biasNames[7],
                biasNames[8],
                biasNames[9],
                biasNames[8],
                biasNames[10],
                biasNames[11],
                biasNames[12],
                biasNames[11],
                biasNames[13],
            ]
        }

        throw EncodecMLXRuntimeError.unsupportedBundle(
            "MLX decode expected 14 or at least 18 GroupNorm bias tensors, got \(biasNames.count)."
        )
    }
}

private func generatedTensorNames(weights: [String: MLXArray], prefix: String) -> [String] {
    weights.keys
        .compactMap { name -> (Int, String)? in
            guard name.hasPrefix(prefix), let number = Int(name.dropFirst(prefix.count)) else {
                return nil
            }
            return (number, name)
        }
        .sorted { left, right in left.0 < right.0 }
        .map(\.1)
}
