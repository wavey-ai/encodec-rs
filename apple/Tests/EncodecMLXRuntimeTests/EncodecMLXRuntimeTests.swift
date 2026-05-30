import Foundation
import MLX
import XCTest
 import EncodecMLXRuntime

final class EncodecMLXRuntimeTests: XCTestCase {
    func testAudioLayoutHelpersConvertStereo() throws {
        let interleaved: [Float] = [1, 10, 2, 20, 3, 30]

        let planar = try encodecInterleavedToPlanar(interleaved, channels: 2)
        let roundtrip = try encodecPlanarToInterleaved(planar, channels: 2, frames: 3)

        XCTAssertEqual(planar, [1, 2, 3, 10, 20, 30])
        XCTAssertEqual(roundtrip, interleaved)
    }

    func testAudioLayoutHelpersKeepMono() throws {
        let mono: [Float] = [1, 2, 3]

        let planar = try encodecInterleavedToPlanar(mono, channels: 1)
        let interleaved = try encodecPlanarToInterleaved(planar, channels: 1, frames: 3)

        XCTAssertEqual(planar, mono)
        XCTAssertEqual(interleaved, mono)
    }

    func testLoadsGeneratedEncodecBundles() throws {
        let sixKbpsBackend = try MLXEncodecFrameBackend(bundleURL: mlxBundleURL("encodec_48khz_6kbps"))
        let sixKbpsSummary = sixKbpsBackend.summary

        XCTAssertEqual(sixKbpsSummary.modelName, "encodec_48khz")
        XCTAssertEqual(sixKbpsSummary.bandwidthKbps, 6.0)
        XCTAssertEqual(sixKbpsSummary.numCodebooks, 4)
        XCTAssertEqual(sixKbpsSummary.frameLength, 150)
        XCTAssertEqual(sixKbpsSummary.encodeTensorCount, 81)
        XCTAssertEqual(sixKbpsSummary.decodeTensorCount, 78)

        let twelveKbpsBackend = try MLXEncodecFrameBackend(bundleURL: mlxBundleURL("encodec_48khz_12kbps"))
        let twelveKbpsSummary = twelveKbpsBackend.summary

        XCTAssertEqual(twelveKbpsSummary.modelName, "encodec_48khz")
        XCTAssertEqual(twelveKbpsSummary.bandwidthKbps, 12.0)
        XCTAssertEqual(twelveKbpsSummary.numCodebooks, 8)
        XCTAssertEqual(twelveKbpsSummary.frameLength, 150)
        XCTAssertEqual(twelveKbpsSummary.encodeTensorCount, 89)
        XCTAssertEqual(twelveKbpsSummary.decodeTensorCount, 82)
    }

    func testDecodeZeroCodesEvaluatesOneFrame() throws {
        let backend = try MLXEncodecFrameBackend(bundleURL: mlxBundleURL("encodec_48khz_6kbps"))
        let codes = zeros(
            [1, backend.metadata.numCodebooks, backend.metadata.frameLength],
            type: Int32.self
        )
        let scale = MLXArray([Float(1.0)], [1, 1])

        let decoded = try backend.decodeFrame(codes: codes, scale: scale)

        XCTAssertEqual(decoded.shape, [1, 2, 48_000])
        try checkedEval(decoded)
    }

    func testEncodeZeroAudioEvaluatesOneFrame() throws {
        for bundleName in ["encodec_48khz_6kbps", "encodec_48khz_12kbps"] {
            let backend = try MLXEncodecFrameBackend(bundleURL: mlxBundleURL(bundleName))
            let audio = zeros(
                [1, backend.metadata.channels, backend.metadata.segmentSamples],
                type: Float.self
            )

            let encoded = try backend.encodeFrame(audio: audio)

            XCTAssertEqual(encoded.codes.shape, [1, backend.metadata.numCodebooks, backend.metadata.frameLength])
            XCTAssertEqual(encoded.scale.shape, [1, 1])
            try checkedEval(encoded.codes, encoded.scale)
        }
    }

    func testNativePipelineEncodesAndDecodesOneRawFrame() throws {
        let pipeline = try MLXEncodecNativePipeline(bundleURL: mlxBundleURL("encodec_48khz_6kbps"))
        let samples = Array(repeating: Float(0), count: 2 * 48_000)

        let payload = try pipeline.encodeEcdc(
            samples: samples,
            channels: 2,
            useLM: false,
            frameBatchSize: 1
        )
        let decoded = try pipeline.decodeEcdc(payload)

        XCTAssertGreaterThan(payload.count, 0)
        XCTAssertEqual(decoded.channels, 2)
        XCTAssertEqual(decoded.frameCount, 48_000)
        XCTAssertEqual(decoded.samples.count, 2 * 48_000)
    }

    func testNativePipelineEncodesAndDecodesOneLmFrame() throws {
        let pipeline = try MLXEncodecNativePipeline(bundleURL: mlxBundleURL("encodec_48khz_6kbps"))
        let samples = Array(repeating: Float(0), count: 2 * 48_000)

        let payload = try pipeline.encodeEcdc(
            samples: samples,
            channels: 2,
            useLM: true,
            frameBatchSize: 1
        )
        let decoded = try pipeline.decodeEcdc(payload)

        XCTAssertGreaterThan(payload.count, 0)
        XCTAssertEqual(decoded.channels, 2)
        XCTAssertEqual(decoded.frameCount, 48_000)
        XCTAssertEqual(decoded.samples.count, 2 * 48_000)
    }

    func testNativePipelineDecodesExistingRawEcdcFixture() throws {
        let fixtureURL = encodecRootURL()
            .appendingPathComponent("target")
            .appendingPathComponent("wasm-smoke")
            .appendingPathComponent("westside_4s_48khz_stereo.raw.ecdc")
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("missing generated ECDC fixture at \(fixtureURL.path)")
        }

        let pipeline = try MLXEncodecNativePipeline(bundleURL: mlxBundleURL("encodec_48khz_12kbps"))
        let decoded = try pipeline.decodeEcdc(Data(contentsOf: fixtureURL))

        XCTAssertEqual(decoded.channels, 2)
        XCTAssertGreaterThan(decoded.frameCount, 0)
        XCTAssertEqual(decoded.samples.count, decoded.channels * decoded.frameCount)
        XCTAssertTrue(decoded.samples.contains { $0 != 0 })
    }

    func testBenchmarkNativeMLXEcdcRoundtrip() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["BITNEEDLE_MLX_BENCH"] == "1" else {
            throw XCTSkip("set BITNEEDLE_MLX_BENCH=1 to run the native MLX benchmark")
        }

        let sourceURL = environment["BITNEEDLE_MLX_BENCH_WAV"]
            .map(URL.init(fileURLWithPath:))
            ?? encodecRootURL()
                .appendingPathComponent("testdata")
                .appendingPathComponent("westside_4s_48khz_stereo.wav")
        let outputDir = environment["BITNEEDLE_MLX_BENCH_OUT"]
            .map(URL.init(fileURLWithPath:))
            ?? encodecRootURL()
                .appendingPathComponent("target")
                .appendingPathComponent("mlx-bench-current")
        let bundleNames = environment["BITNEEDLE_MLX_BENCH_BUNDLES"]?
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            ?? ["encodec_48khz_6kbps", "encodec_48khz_12kbps"]
        let useLM = envFlag(environment["BITNEEDLE_MLX_BENCH_LM"], defaultValue: true)
        let warmup = envFlag(environment["BITNEEDLE_MLX_BENCH_WARMUP"], defaultValue: true)
        let encodeOnly = envFlag(environment["BITNEEDLE_MLX_BENCH_ENCODE_ONLY"], defaultValue: false)
        let parsedFrameBatchSize = environment["BITNEEDLE_MLX_BENCH_BATCH_SIZE"].flatMap(Int.init)
        let frameBatchSize = parsedFrameBatchSize.map { max($0, 1) } ?? 1
        let chunkMilliseconds = environment["BITNEEDLE_MLX_BENCH_CHUNK_MS"].flatMap(Double.init)

        let audio = try readBenchmarkWav(sourceURL)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        print(
            "benchmark: source_wav=\(sourceURL.path) frames=\(audio.frameCount) " +
            "duration=\(String(format: "%.3f", audio.durationSeconds))s " +
            "sample_rate=\(audio.sampleRate) channels=\(audio.channels)"
        )
        print(
            "benchmark: use_lm=\(useLM) frame_batch_size=\(frameBatchSize) " +
            "chunk_ms=\(chunkMilliseconds.map { String(format: "%.3f", $0) } ?? "default") " +
            "warmup=\(warmup) encode_only=\(encodeOnly) output_dir=\(outputDir.path)"
        )

        for bundleName in bundleNames {
            let loadStart = BenchmarkClock.now()
            let pipeline = try MLXEncodecNativePipeline(bundleURL: mlxBundleURL(bundleName))
            let loadElapsed = loadStart.elapsed()
            let summary = pipeline.summary
            XCTAssertEqual(audio.sampleRate, summary.sampleRate)
            XCTAssertEqual(audio.channels, summary.channels)
            print(
                "benchmark: bundle=\(bundleName) load_elapsed_s=" +
                String(format: "%.3f", loadElapsed)
            )

            if warmup {
                let warmupFrames = min(audio.frameCount, summary.sampleRate)
                let warmupSamples = Array(audio.samples.prefix(warmupFrames * audio.channels))
                let warmupStart = BenchmarkClock.now()
                let warmupPayload = try pipeline.encodeEcdc(
                    samples: warmupSamples,
                    channels: audio.channels,
                    useLM: useLM,
                    frameBatchSize: frameBatchSize,
                    chunkMilliseconds: chunkMilliseconds
                )
                if !encodeOnly {
                    _ = try pipeline.decodeEcdc(warmupPayload)
                }
                print(
                    "benchmark: bundle=\(bundleName) warmup_elapsed_s=" +
                    String(format: "%.3f", warmupStart.elapsed())
                )
            }

            let modeName = useLM ? "lm" : "raw"
            let outputStem = "\(sourceURL.deletingPathExtension().lastPathComponent).\(bundleName).\(modeName).mlx"
            let ecdcURL = outputDir.appendingPathComponent(outputStem).appendingPathExtension("ecdc")
            let wavURL = outputDir.appendingPathComponent(outputStem).appendingPathExtension("decoded.wav")

            let encodeStart = BenchmarkClock.now()
            let payload = try pipeline.encodeEcdc(
                samples: audio.samples,
                channels: audio.channels,
                useLM: useLM,
                frameBatchSize: frameBatchSize,
                chunkMilliseconds: chunkMilliseconds
            )
            let encodeElapsed = encodeStart.elapsed()
            try payload.write(to: ecdcURL)

            if encodeOnly {
                print(
                    "benchmark: bundle=\(bundleName) bitrate_kbps=" +
                    String(format: "%.1f", summary.bandwidthKbps) +
                    " encode_s=\(String(format: "%.3f", encodeElapsed))" +
                    " ecdc_bytes=\(payload.count)" +
                    " ecdc=\(ecdcURL.path)"
                )
                continue
            }

            let decodeStart = BenchmarkClock.now()
            let decoded = try pipeline.decodeEcdc(payload)
            let decodeElapsed = decodeStart.elapsed()
            try writeBenchmarkWav(
                wavURL,
                samples: decoded.samples,
                channels: decoded.channels,
                sampleRate: summary.sampleRate
            )

            XCTAssertEqual(decoded.channels, audio.channels)
            XCTAssertEqual(decoded.frameCount, audio.frameCount)
            print(
                "benchmark: bundle=\(bundleName) bitrate_kbps=" +
                String(format: "%.1f", summary.bandwidthKbps) +
                " encode_s=\(String(format: "%.3f", encodeElapsed))" +
                " decode_s=\(String(format: "%.3f", decodeElapsed))" +
                " ecdc_bytes=\(payload.count)" +
                " decoded_frames=\(decoded.frameCount)" +
                " ecdc=\(ecdcURL.path)" +
                " wav=\(wavURL.path)"
            )
        }
    }

    private func mlxBundleURL(_ name: String) -> URL {
        encodecRootURL()
            .appendingPathComponent("target")
            .appendingPathComponent("mlx-bundles")
            .appendingPathComponent(name)
    }

    private func encodecRootURL() -> URL {
        let testFile = URL(fileURLWithPath: #filePath)
        let appleRoot = testFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return appleRoot.deletingLastPathComponent()
    }
}

private struct BenchmarkClock {
    private let start: UInt64

    static func now() -> BenchmarkClock {
        BenchmarkClock(start: DispatchTime.now().uptimeNanoseconds)
    }

    func elapsed() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    }
}

private struct BenchmarkWavAudio {
    let samples: [Float]
    let channels: Int
    let sampleRate: Int
    let frameCount: Int

    var durationSeconds: Double {
        Double(frameCount) / Double(sampleRate)
    }
}

private func envFlag(_ value: String?, defaultValue: Bool) -> Bool {
    guard let value else {
        return defaultValue
    }
    switch value.lowercased() {
    case "1", "true", "yes", "on":
        return true
    case "0", "false", "no", "off":
        return false
    default:
        return defaultValue
    }
}

private func readBenchmarkWav(_ url: URL) throws -> BenchmarkWavAudio {
    let data = try Data(contentsOf: url)
    guard data.count >= 44,
          ascii(data, 0, 4) == "RIFF",
          ascii(data, 8, 4) == "WAVE"
    else {
        throw XCTSkip("unsupported benchmark WAV container at \(url.path)")
    }

    var offset = 12
    var audioFormat: UInt16?
    var channels: Int?
    var sampleRate: Int?
    var bitsPerSample: UInt16?
    var dataOffset: Int?
    var dataSize: Int?

    while offset + 8 <= data.count {
        let chunkID = ascii(data, offset, 4)
        let chunkSize = Int(readUInt32LE(data, offset + 4))
        let chunkDataOffset = offset + 8
        guard chunkDataOffset + chunkSize <= data.count else {
            throw XCTSkip("truncated WAV chunk \(chunkID) in \(url.path)")
        }

        if chunkID == "fmt " {
            guard chunkSize >= 16 else {
                throw XCTSkip("unsupported short WAV fmt chunk in \(url.path)")
            }
            audioFormat = readUInt16LE(data, chunkDataOffset)
            channels = Int(readUInt16LE(data, chunkDataOffset + 2))
            sampleRate = Int(readUInt32LE(data, chunkDataOffset + 4))
            bitsPerSample = readUInt16LE(data, chunkDataOffset + 14)
        } else if chunkID == "data" {
            dataOffset = chunkDataOffset
            dataSize = chunkSize
        }

        offset = chunkDataOffset + chunkSize + (chunkSize & 1)
    }

    guard let audioFormat,
          let channels,
          let sampleRate,
          let bitsPerSample,
          let dataOffset,
          let dataSize
    else {
        throw XCTSkip("missing WAV fmt/data chunk in \(url.path)")
    }
    guard channels > 0 else {
        throw XCTSkip("benchmark WAV has no channels: \(url.path)")
    }

    let samples: [Float]
    switch (audioFormat, bitsPerSample) {
    case (1, 16):
        guard dataSize % 2 == 0 else {
            throw XCTSkip("unaligned 16-bit WAV data in \(url.path)")
        }
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 2).map { index in
            Float(Int16(bitPattern: readUInt16LE(data, index))) / Float(Int16.max)
        }
    case (3, 32):
        guard dataSize % 4 == 0 else {
            throw XCTSkip("unaligned 32-bit float WAV data in \(url.path)")
        }
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 4).map { index in
            Float(bitPattern: readUInt32LE(data, index))
        }
    default:
        throw XCTSkip("unsupported benchmark WAV format \(audioFormat)/\(bitsPerSample) in \(url.path)")
    }

    guard samples.count % channels == 0 else {
        throw XCTSkip("benchmark WAV sample count is not divisible by channels: \(url.path)")
    }
    return BenchmarkWavAudio(
        samples: samples,
        channels: channels,
        sampleRate: sampleRate,
        frameCount: samples.count / channels
    )
}

private func writeBenchmarkWav(_ url: URL, samples: [Float], channels: Int, sampleRate: Int) throws {
    guard channels > 0, samples.count % channels == 0 else {
        throw EncodecMLXRuntimeError.nativeBridge("cannot write WAV with invalid channel/sample count")
    }
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )

    let dataByteCount = samples.count * MemoryLayout<Int16>.size
    var data = Data(capacity: 44 + dataByteCount)
    data.append(contentsOf: "RIFF".utf8)
    appendUInt32LE(UInt32(36 + dataByteCount), to: &data)
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    appendUInt32LE(16, to: &data)
    appendUInt16LE(1, to: &data)
    appendUInt16LE(UInt16(channels), to: &data)
    appendUInt32LE(UInt32(sampleRate), to: &data)
    appendUInt32LE(UInt32(sampleRate * channels * MemoryLayout<Int16>.size), to: &data)
    appendUInt16LE(UInt16(channels * MemoryLayout<Int16>.size), to: &data)
    appendUInt16LE(16, to: &data)
    data.append(contentsOf: "data".utf8)
    appendUInt32LE(UInt32(dataByteCount), to: &data)

    for sample in samples {
        let clamped = min(max(sample, -1), 1)
        appendUInt16LE(UInt16(bitPattern: Int16((clamped * Float(Int16.max)).rounded())), to: &data)
    }

    try data.write(to: url)
}

private func ascii(_ data: Data, _ offset: Int, _ count: Int) -> String {
    guard offset >= 0, count >= 0, offset + count <= data.count else {
        return ""
    }
    return String(decoding: data[offset ..< offset + count], as: UTF8.self)
}

private func readUInt16LE(_ data: Data, _ offset: Int) -> UInt16 {
    UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
}

private func readUInt32LE(_ data: Data, _ offset: Int) -> UInt32 {
    UInt32(data[offset])
        | (UInt32(data[offset + 1]) << 8)
        | (UInt32(data[offset + 2]) << 16)
        | (UInt32(data[offset + 3]) << 24)
}

private func appendUInt16LE(_ value: UInt16, to data: inout Data) {
    data.append(UInt8(value & 0xff))
    data.append(UInt8(value >> 8))
}

private func appendUInt32LE(_ value: UInt32, to data: inout Data) {
    data.append(UInt8(value & 0xff))
    data.append(UInt8((value >> 8) & 0xff))
    data.append(UInt8((value >> 16) & 0xff))
    data.append(UInt8((value >> 24) & 0xff))
}
