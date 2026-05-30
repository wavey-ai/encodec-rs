import Foundation
import MLX
import MLXNN
import CEncodecMLXBridge

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

    private let encodeWeights: [String: MLXArray]
    private let decodeWeights: [String: MLXArray]
    private let encoder: MLXEncodecFrameEncoder
    private let decoder: MLXEncodecFrameDecoder

    public init(bundleURL: URL) throws {
        self.metadata = try EncodecFrameMetadata.load(from: bundleURL)
        self.manifest = try EncodecMLXWeightManifest.load(from: bundleURL)

        guard let encodeModel = manifest.models["encode_frame"] else {
            throw EncodecMLXRuntimeError.missingModel("encode_frame")
        }
        guard let decodeModel = manifest.models["decode_frame"] else {
            throw EncodecMLXRuntimeError.missingModel("decode_frame")
        }

        self.encodeWeights = try Self.loadWeightArrays(
            from: bundleURL.appendingPathComponent(encodeModel.safetensors)
        )
        self.decodeWeights = try Self.loadWeightArrays(
            from: bundleURL.appendingPathComponent(decodeModel.safetensors)
        )

        guard !encodeWeights.isEmpty else {
            throw EncodecMLXRuntimeError.emptyWeights("encode_frame")
        }
        guard !decodeWeights.isEmpty else {
            throw EncodecMLXRuntimeError.emptyWeights("decode_frame")
        }

        self.encoder = try MLXEncodecFrameEncoder(metadata: metadata, weights: encodeWeights)
        self.decoder = try MLXEncodecFrameDecoder(metadata: metadata, weights: decodeWeights)
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
            encodeTensorCount: encodeWeights.count,
            decodeTensorCount: decodeWeights.count,
            encodeParameterCount: manifest.models["encode_frame"]?.parameterCount ?? 0,
            decodeParameterCount: manifest.models["decode_frame"]?.parameterCount ?? 0
        )
    }

    public func encodeFrame(audio: MLXArray) throws -> (codes: MLXArray, scale: MLXArray) {
        try encoder.encodeFrame(audio: audio)
    }

    public func decodeFrame(codes: MLXArray, scale: MLXArray) throws -> MLXArray {
        try decoder.decodeFrame(codes: codes, scale: scale)
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

    public func decodeEcdc(_ payload: Data) throws -> EncodecNativeDecodedAudio {
        let callbackBox = MLXNativeFrameCallbackBox(backend: backend)
        let callbacks = callbackBox.callbacks()
        let result = withExtendedLifetime(callbackBox) {
            bundleURL.path.withCString { bundlePath in
                payload.withUnsafeBytes { payloadBuffer in
                    let payloadBytes = payloadBuffer.bindMemory(to: UInt8.self)
                    return encodec_rs_mlx_decode_ecdc(
                        bundlePath,
                        payloadBytes.baseAddress,
                        payloadBytes.count,
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

        let planarSamples = Array(UnsafeBufferPointer(start: ptr, count: result.len))
        let samples = try encodecPlanarToInterleaved(
            planarSamples,
            channels: result.channels,
            frames: result.samples
        )
        return EncodecNativeDecodedAudio(
            samples: samples,
            channels: result.channels,
            frameCount: result.samples
        )
    }

    public func encodeEcdc(
        samples: [Float],
        channels: Int,
        useLM: Bool = true,
        frameBatchSize: Int = 1,
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
        let planarSamples = try encodecInterleavedToPlanar(samples, channels: channels)
        let result = withExtendedLifetime(callbackBox) {
            bundleURL.path.withCString { bundlePath in
                planarSamples.withUnsafeBufferPointer { sampleBuffer in
                    encodec_rs_mlx_encode_ecdc(
                        bundlePath,
                        sampleBuffer.baseAddress,
                        channels,
                        frames,
                        useLM,
                        frameBatchSize,
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

private let mlxNativeEncodeFrameCallback: EncodecRsMlxEncodeFrameFn = { userData, audio, batch, channels, samples, codesOut, codesLen, scalesOut, scalesLen in
    guard let userData, let audio, let codesOut, let scalesOut else {
        return -1
    }

    do {
        let profileStarted = mlxNativeProfileStarted()
        let box = Unmanaged<MLXNativeFrameCallbackBox>.fromOpaque(userData).takeUnretainedValue()
        let audioCount = batch * channels * samples
        let audioValues = Array(UnsafeBufferPointer(start: audio, count: audioCount))
        let audioArray = MLXArray(audioValues, [batch, channels, samples])
        let inputDone = mlxNativeProfileNow()
        let encoded = try box.backend.encodeFrame(audio: audioArray)
        let graphDone = mlxNativeProfileNow()
        let codeValues = encoded.codes.asArray(Int64.self)
        let scaleValues = encoded.scale.asArray(Float.self)
        let readbackDone = mlxNativeProfileNow()
        guard codeValues.count == codesLen, scaleValues.count == scalesLen else {
            box.record(
                "encode callback produced \(codeValues.count) codes and \(scaleValues.count) scales, expected \(codesLen) and \(scalesLen)"
            )
            return -2
        }
        for index in 0 ..< codesLen {
            codesOut[index] = codeValues[index]
        }
        for index in 0 ..< scalesLen {
            scalesOut[index] = scaleValues[index]
        }
        mlxNativeProfilePrint(
            "encode_callback batch=\(batch) input_ms=\(mlxNativeProfileMillis(profileStarted, inputDone)) " +
            "graph_ms=\(mlxNativeProfileMillis(inputDone, graphDone)) " +
            "readback_ms=\(mlxNativeProfileMillis(graphDone, readbackDone)) " +
            "copy_ms=\(mlxNativeProfileMillis(readbackDone, mlxNativeProfileNow())) " +
            "total_ms=\(mlxNativeProfileMillis(profileStarted, mlxNativeProfileNow()))"
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
        let codeValues = Array(UnsafeBufferPointer(start: codes, count: codeCount)).map(Int32.init)
        let scaleValues = Array(UnsafeBufferPointer(start: scales, count: scalesLen))
        let codeArray = MLXArray(codeValues, [batch, codebooks, frames])
        let scaleArray = MLXArray(scaleValues, [batch, max(scalesLen / max(batch, 1), 1)])
        let inputDone = mlxNativeProfileNow()
        let decoded = try box.backend.decodeFrame(codes: codeArray, scale: scaleArray)
        let graphDone = mlxNativeProfileNow()
        let audioValues = decoded.asArray(Float.self)
        let readbackDone = mlxNativeProfileNow()
        guard audioValues.count == audioLen else {
            box.record(
                "decode callback produced \(audioValues.count) samples, expected \(audioLen)"
            )
            return -2
        }
        for index in 0 ..< audioLen {
            audioOut[index] = audioValues[index]
        }
        mlxNativeProfilePrint(
            "decode_callback batch=\(batch) input_ms=\(mlxNativeProfileMillis(profileStarted, inputDone)) " +
            "graph_ms=\(mlxNativeProfileMillis(inputDone, graphDone)) " +
            "readback_ms=\(mlxNativeProfileMillis(graphDone, readbackDone)) " +
            "copy_ms=\(mlxNativeProfileMillis(readbackDone, mlxNativeProfileNow())) " +
            "total_ms=\(mlxNativeProfileMillis(profileStarted, mlxNativeProfileNow()))"
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
        let inputWeight: String
        let recurrentWeight: String
        let bias: String
    }

    private let metadata: EncodecFrameMetadata
    private let weights: [String: MLXArray]
    private let norms: [Norm]
    private let lstmLayers: [LstmLayer]
    private let codebookMatMulNames: [String]

    init(metadata: EncodecFrameMetadata, weights: [String: MLXArray]) throws {
        guard metadata.modelName == "encodec_48khz" else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode currently supports encodec_48khz bundles only."
            )
        }
        guard metadata.sampleRate == 48_000, metadata.channels == 2, metadata.frameLength == 150 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode currently expects 48 kHz stereo frames with length 150."
            )
        }

        self.metadata = metadata
        self.weights = weights

        let lstmNames = generatedTensorNames(weights: weights, prefix: "onnx::LSTM_")
        guard lstmNames.count == 6 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode expected 6 generated LSTM tensors, got \(lstmNames.count)."
            )
        }
        self.lstmLayers = stride(from: 0, to: lstmNames.count, by: 3).map { index in
            LstmLayer(
                inputWeight: lstmNames[index],
                recurrentWeight: lstmNames[index + 1],
                bias: lstmNames[index + 2]
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
            Norm(
                scale: try Self.required(weights, scaleNames[index]),
                bias: try Self.required(weights, biasNames[index])
            )
        }

        let matMulNames = generatedTensorNames(weights: weights, prefix: "onnx::MatMul_")
        guard matMulNames.count >= metadata.numCodebooks else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode expected at least \(metadata.numCodebooks) quantizer MatMul tensors, got \(matMulNames.count)."
            )
        }
        self.codebookMatMulNames = Array(matMulNames.prefix(metadata.numCodebooks))
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
        let weight = try tensor(weightName).transposed(0, 2, 1)
        let bias = try tensor(biasName)
        var y = conv1d(nlc, weight, stride: stride).transposed(0, 2, 1)
        guard y.shape[1] == bias.shape[0] else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX encode convolution \(weightName) produced shape \(y.shape), but bias \(biasName) has shape \(bias.shape)."
            )
        }
        y = y + bias.reshaped([1, bias.shape[0], 1])
        return groupNorm(y, norm: norm)
    }

    private func groupNorm(_ input: MLXArray, norm index: Int) -> MLXArray {
        let mean = input.mean(axes: [1, 2], keepDims: true)
        let variance = input.variance(axes: [1, 2], keepDims: true)
        let normalized = (input - mean) / sqrt(variance + 0.00001)
        return normalized * norms[index].scale.reshaped([1, norms[index].scale.shape[0], 1])
            + norms[index].bias.reshaped([1, norms[index].bias.shape[0], 1])
    }

    private func slstm(_ input: MLXArray) throws -> MLXArray {
        let nlc = input.transposed(0, 2, 1)
        let first = try onnxLstm(nlc, layer: lstmLayers[0])
        let second = try onnxLstm(first, layer: lstmLayers[1])
        return (second + nlc).transposed(0, 2, 1)
    }

    private func onnxLstm(_ input: MLXArray, layer: LstmLayer) throws -> MLXArray {
        let w = try tensor(layer.inputWeight)[0, 0..., 0...]
        let r = try tensor(layer.recurrentWeight)[0, 0..., 0...]
        let b = try tensor(layer.bias)[0, 0...]
        let hiddenSize = w.shape[0] / 4
        let bias = b[..<(4 * hiddenSize)] + b[(4 * hiddenSize)..<(8 * hiddenSize)]
        let projectedInput = matmul(input, w.T) + bias

        var hidden: MLXArray?
        var cell: MLXArray?
        var outputs = [MLXArray]()
        outputs.reserveCapacity(input.shape[1])

        for index in 0 ..< input.shape[1] {
            var gates = projectedInput[0..., index, 0...]
            if let hidden {
                gates = gates + matmul(hidden, r.T)
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
            let codebookColumns = try tensor(codebookMatMulNames[index])
            let residualFlat = residual.reshaped([-1, residual.shape[2]])
            let residualNorm = (residualFlat * residualFlat).sum(axis: 1, keepDims: true)
            let codebookNorm = (codebookColumns * codebookColumns).sum(axis: 0, keepDims: true)
            let distances = residualNorm - matmul(residualFlat * 2.0, codebookColumns) + codebookNorm
            let indices = argMax(-distances, axis: -1).reshaped([residual.shape[0], residual.shape[1]])
            codes.append(indices)

            if index + 1 < metadata.numCodebooks {
                let codebook = try quantizerCodebook(index)
                let quantized = codebook.take(indices, axis: 0)
                residual = residual - quantized
            }
        }

        return stacked(codes, axis: 1)
    }

    private func quantizerCodebook(_ index: Int) throws -> MLXArray {
        let directName = "model.quantizer.vq.layers.\(index)._codebook.embed"
        if let direct = weights[directName] {
            return direct
        }

        let columns = try tensor(codebookMatMulNames[index])
        return columns.T
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

    private func tensor(_ name: String) throws -> MLXArray {
        try Self.required(weights, name)
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
        let inputWeight: String
        let recurrentWeight: String
        let bias: String
    }

    private let metadata: EncodecFrameMetadata
    private let weights: [String: MLXArray]
    private let norms: [Norm]
    private let lstmLayers: [LstmLayer]

    init(metadata: EncodecFrameMetadata, weights: [String: MLXArray]) throws {
        guard metadata.modelName == "encodec_48khz" else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode currently supports encodec_48khz bundles only."
            )
        }
        guard metadata.sampleRate == 48_000, metadata.channels == 2, metadata.frameLength == 150 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode currently expects 48 kHz stereo frames with length 150."
            )
        }

        self.metadata = metadata
        self.weights = weights

        let lstmNames = generatedTensorNames(weights: weights, prefix: "onnx::LSTM_")
        guard lstmNames.count == 6 else {
            throw EncodecMLXRuntimeError.unsupportedBundle(
                "MLX decode expected 6 generated LSTM tensors, got \(lstmNames.count)."
            )
        }
        self.lstmLayers = stride(from: 0, to: lstmNames.count, by: 3).map { index in
            LstmLayer(
                inputWeight: lstmNames[index],
                recurrentWeight: lstmNames[index + 1],
                bias: lstmNames[index + 2]
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
            Norm(
                scale: try Self.required(weights, scaleNames[index]),
                bias: try Self.required(weights, biasNames[index])
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
            let codebook = try tensor("model.quantizer.vq.layers.\(index)._codebook.embed")
            let codebookIndices = codes[0..., index, 0...]
            z = z + codebook.take(codebookIndices, axis: 0)
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
        let weight = try tensor(weightName).transposed(0, 2, 1)
        let bias = try tensor(biasName)
        var y = conv1d(nlc, weight).transposed(0, 2, 1)
        y = y + bias.reshaped([1, bias.shape[0], 1])
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
        let weight = try tensor(weightName).transposed(1, 2, 0)
        let bias = try tensor(biasName)
        var y = convTransposed1d(nlc, weight, stride: stride).transposed(0, 2, 1)
        y = y + bias.reshaped([1, bias.shape[0], 1])
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
        return normalized * norms[index].scale.reshaped([1, norms[index].scale.shape[0], 1])
            + norms[index].bias.reshaped([1, norms[index].bias.shape[0], 1])
    }

    private func slstm(_ input: MLXArray) throws -> MLXArray {
        let nlc = input.transposed(0, 2, 1)
        let first = try onnxLstm(nlc, layer: lstmLayers[0])
        let second = try onnxLstm(first, layer: lstmLayers[1])
        return (second + nlc).transposed(0, 2, 1)
    }

    private func onnxLstm(_ input: MLXArray, layer: LstmLayer) throws -> MLXArray {
        let w = try tensor(layer.inputWeight)[0, 0..., 0...]
        let r = try tensor(layer.recurrentWeight)[0, 0..., 0...]
        let b = try tensor(layer.bias)[0, 0...]
        let hiddenSize = w.shape[0] / 4
        let bias = b[..<(4 * hiddenSize)] + b[(4 * hiddenSize)..<(8 * hiddenSize)]
        let projectedInput = matmul(input, w.T) + bias

        var hidden: MLXArray?
        var cell: MLXArray?
        var outputs = [MLXArray]()
        outputs.reserveCapacity(input.shape[1])

        for index in 0 ..< input.shape[1] {
            var gates = projectedInput[0..., index, 0...]
            if let hidden {
                gates = gates + matmul(hidden, r.T)
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

    private func tensor(_ name: String) throws -> MLXArray {
        try Self.required(weights, name)
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
