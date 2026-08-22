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

    func testBenchmarkWavReaderAcceptsExtensiblePacked24BitPcm() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let expectedIntegers: [Int32] = [-8_388_608, -1, 0, 8_388_607]
        var data = Data()
        data.append(contentsOf: "RIFF".utf8)
        appendUInt32LE(UInt32(60 + expectedIntegers.count * 3), to: &data)
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        appendUInt32LE(40, to: &data)
        appendUInt16LE(0xfffe, to: &data)
        appendUInt16LE(2, to: &data)
        appendUInt32LE(48_000, to: &data)
        appendUInt32LE(48_000 * 2 * 3, to: &data)
        appendUInt16LE(6, to: &data)
        appendUInt16LE(24, to: &data)
        appendUInt16LE(22, to: &data)
        appendUInt16LE(24, to: &data)
        appendUInt32LE(3, to: &data)
        data.append(contentsOf: [
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
            0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
        ])
        data.append(contentsOf: "data".utf8)
        appendUInt32LE(UInt32(expectedIntegers.count * 3), to: &data)
        for sample in expectedIntegers {
            let packed = UInt32(bitPattern: sample)
            data.append(UInt8(packed & 0xff))
            data.append(UInt8((packed >> 8) & 0xff))
            data.append(UInt8((packed >> 16) & 0xff))
        }
        try data.write(to: url)

        let audio = try readBenchmarkWav(url)

        XCTAssertEqual(audio.channels, 2)
        XCTAssertEqual(audio.sampleRate, 48_000)
        XCTAssertEqual(audio.frameCount, 2)
        XCTAssertEqual(
            audio.samples,
            expectedIntegers.map { Float($0) / 8_388_608 }
        )
    }

    func testLoadsGeneratedEncodecBundles() throws {
        let sixKbpsBackend = try MLXEncodecFrameBackend(
            bundleURL: mlxBundleURL("encodec_48khz_6kbps_1333ms")
        )
        let sixKbpsSummary = sixKbpsBackend.summary

        XCTAssertEqual(sixKbpsSummary.modelName, "encodec_48khz")
        XCTAssertEqual(sixKbpsSummary.bandwidthKbps, 6.0)
        XCTAssertEqual(sixKbpsSummary.numCodebooks, 4)
        XCTAssertEqual(sixKbpsSummary.frameLength, 203)
        XCTAssertEqual(sixKbpsSummary.encodeTensorCount, 81)
        XCTAssertEqual(sixKbpsSummary.decodeTensorCount, 78)

        let twelveKbpsBackend = try MLXEncodecFrameBackend(
            bundleURL: mlxBundleURL("encodec_48khz_12kbps_1333ms")
        )
        let twelveKbpsSummary = twelveKbpsBackend.summary

        XCTAssertEqual(twelveKbpsSummary.modelName, "encodec_48khz")
        XCTAssertEqual(twelveKbpsSummary.bandwidthKbps, 12.0)
        XCTAssertEqual(twelveKbpsSummary.numCodebooks, 8)
        XCTAssertEqual(twelveKbpsSummary.frameLength, 203)
        XCTAssertEqual(twelveKbpsSummary.encodeTensorCount, 89)
        XCTAssertEqual(twelveKbpsSummary.decodeTensorCount, 82)
    }

    func testDecodeZeroCodesEvaluatesOneFrame() throws {
        let backend = try MLXEncodecFrameBackend(
            bundleURL: mlxBundleURL("encodec_48khz_6kbps_1333ms")
        )
        let codes = zeros(
            [1, backend.metadata.numCodebooks, backend.metadata.frameLength],
            type: Int32.self
        )
        let scale = MLXArray([Float(1.0)], [1, 1])

        let decoded = try backend.decodeFrame(codes: codes, scale: scale)

        XCTAssertEqual(decoded.shape, [1, 2, backend.metadata.segmentSamples])
        try checkedEval(decoded)
    }

    func testEncodeZeroAudioEvaluatesOneFrame() throws {
        for bundleName in ["encodec_48khz_6kbps_1333ms", "encodec_48khz_12kbps_1333ms"] {
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

    func testNativePipelineRejectsRawEcdcEncoding() throws {
        let pipeline = try MLXEncodecNativePipeline(
            bundleURL: mlxBundleURL("encodec_48khz_6kbps_1333ms")
        )
        let samples = Array(repeating: Float(0), count: 2 * 48_000)

        XCTAssertThrowsError(
            try pipeline.encodeEcdc(
                samples: samples,
                channels: 2,
                useLM: false,
                frameBatchSize: 1
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("use_lm=false is unsupported"))
        }
    }

    func testNativePipelineEncodesAndDecodesOneLmFrame() throws {
        let pipeline = try MLXEncodecNativePipeline(
            bundleURL: mlxBundleURL("encodec_48khz_6kbps_1333ms")
        )
        try pipeline.prewarm()
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

    func testNativePipelineDerivesExactSixKbpsStream() throws {
        let frameCount = 48_000
        let samples = (0 ..< frameCount).flatMap { frame -> [Float] in
            let phase = Float(frame) / 48_000
            return [
                0.2 * sin(phase * 440 * 2 * .pi),
                0.2 * sin(phase * 659 * 2 * .pi),
            ]
        }

        for profileMilliseconds in [1333, 1800] {
            let primaryBundle = mlxBundleURL(
                "encodec_48khz_12kbps_\(profileMilliseconds)ms"
            )
            let derivedBundle = mlxBundleURL(
                "encodec_48khz_6kbps_\(profileMilliseconds)ms"
            )
            let primaryPipeline = try MLXEncodecNativePipeline(bundleURL: primaryBundle)
            let derivedPipeline = try MLXEncodecNativePipeline(bundleURL: derivedBundle)

            let primaryOnly = try primaryPipeline.encodeEcdc(
                samples: samples,
                channels: 2,
                frameBatchSize: 1
            )
            let dual = try primaryPipeline.encodeEcdcOutputs(
                samples: samples,
                channels: 2,
                derived6KBundleURL: derivedBundle,
                frameBatchSize: 1
            )
            let derivedSeparate = try derivedPipeline.encodeEcdc(
                samples: samples,
                channels: 2,
                frameBatchSize: 1
            )

            XCTAssertEqual(dual.primary, primaryOnly)
            XCTAssertEqual(dual.derived6K, derivedSeparate)

        }
    }

    func testNativePipelineDecodesExistingRawEcdcFixture() throws {
        let fixtureURL = encodecRootURL()
            .appendingPathComponent("target")
            .appendingPathComponent("wasm-smoke")
            .appendingPathComponent("westside_4s_48khz_stereo.raw.ecdc")
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("missing generated ECDC fixture at \(fixtureURL.path)")
        }

        let pipeline = try MLXEncodecNativePipeline(
            bundleURL: mlxBundleURL("encodec_48khz_12kbps_1333ms")
        )
        let decoded = try pipeline.decodeEcdc(Data(contentsOf: fixtureURL))

        XCTAssertEqual(decoded.channels, 2)
        XCTAssertGreaterThan(decoded.frameCount, 0)
        XCTAssertEqual(decoded.samples.count, decoded.channels * decoded.frameCount)
        XCTAssertTrue(decoded.samples.contains { $0 != 0 })
    }

    /// Release-only direct-frame gate. Run `record` on the unoptimized source
    /// before changing it, then `verify` on each candidate in priority order.
    func testReleaseBatchOneFrameOptimizationParity() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let mode = environment["BITNEEDLE_MLX_ACCEPTANCE_MODE"],
              mode == "record" || mode == "verify"
        else {
            throw XCTSkip(
                "set BITNEEDLE_MLX_ACCEPTANCE_MODE=record or verify to run the MLX parity gate"
            )
        }
#if DEBUG
        throw XCTSkip("the MLX parity gate must be compiled and run in Release")
#else
        guard let referencePath = environment["BITNEEDLE_MLX_ACCEPTANCE_REFERENCE_DIR"] else {
            XCTFail("BITNEEDLE_MLX_ACCEPTANCE_REFERENCE_DIR is required")
            return
        }
        let referenceURL = URL(fileURLWithPath: referencePath, isDirectory: true)
        let bundleURL = mlxBundleURL("encodec_48khz_6kbps_1333ms")
        guard FileManager.default.fileExists(
            atPath: bundleURL.appendingPathComponent("mlx-manifest.json").path
        ) else {
            throw XCTSkip("missing MLX acceptance bundle at \(bundleURL.path)")
        }

        let current = try evaluateBatchOneFrame(bundleURL: bundleURL)
        if mode == "record" {
            try recordFrameReference(current, at: referenceURL)
            printBaselineRecord(current, referenceURL: referenceURL)
            return
        }

        let baseline = try loadFrameReference(from: referenceURL)
        let codeParity = baseline.codes == current.codes
        guard codeParity else {
            XCTFail("candidate changed encoded codes")
            return
        }

        let scaleMaxAbs = maximumAbsoluteDifference(baseline.scale, current.scale)
        let pcm = pcmComparison(baseline.pcm, current.pcm)
        let pcmMaxAbsTolerance = 1e-5
        let pcmRMSETolerance = 1e-6
        guard scaleMaxAbs <= 1e-7 else {
            XCTFail("candidate scale max-abs \(scaleMaxAbs) exceeds 1e-7")
            return
        }
        guard pcm.maxAbs <= pcmMaxAbsTolerance, pcm.rmse <= pcmRMSETolerance else {
            XCTFail(
                "candidate PCM max-abs/rmse \(pcm.maxAbs)/\(pcm.rmse) " +
                "exceeds \(pcmMaxAbsTolerance)/\(pcmRMSETolerance)"
            )
            return
        }

        printQualificationRecord(
            baseline: baseline,
            candidate: current,
            candidateLabel: environment["BITNEEDLE_MLX_ACCEPTANCE_CANDIDATE"] ?? "current",
            codeParity: codeParity,
            scaleMaxAbs: scaleMaxAbs,
            pcm: pcm,
            pcmMaxAbsTolerance: pcmMaxAbsTolerance,
            pcmRMSETolerance: pcmRMSETolerance
        )
#endif
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
            ?? ["encodec_48khz_6kbps_1333ms", "encodec_48khz_12kbps_1333ms"]
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
                    _ = try pipeline.decodeEcdc(
                        warmupPayload,
                        frameBatchSize: frameBatchSize
                    )
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
            let decoded = try pipeline.decodeEcdc(
                payload,
                frameBatchSize: frameBatchSize
            )
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

    func testBenchmarkOptionalDerivedSixKbpsEncode() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["BITNEEDLE_MLX_DUAL_BENCH"] == "1" else {
            throw XCTSkip(
                "set BITNEEDLE_MLX_DUAL_BENCH=1 to run the optional 6 kbps benchmark"
            )
        }

        let sourceURL = environment["BITNEEDLE_MLX_BENCH_WAV"]
            .map(URL.init(fileURLWithPath:))
            ?? encodecRootURL()
                .appendingPathComponent("testdata")
                .appendingPathComponent("westside_4s_48khz_stereo.wav")
        let profileMilliseconds = environment["BITNEEDLE_MLX_DUAL_PROFILE_MS"]
            .flatMap(Int.init) ?? 1333
        let frameBatchSize = environment["BITNEEDLE_MLX_BENCH_BATCH_SIZE"]
            .flatMap(Int.init).map { max($0, 1) }
            ?? EncodecMLXRuntimeDefaults.frameBatchSize
        let chunkMilliseconds = environment["BITNEEDLE_MLX_BENCH_CHUNK_MS"].flatMap(Double.init)
        let outputDir = environment["BITNEEDLE_MLX_BENCH_OUT"]
            .map(URL.init(fileURLWithPath:))
            ?? encodecRootURL()
                .appendingPathComponent("target")
                .appendingPathComponent("mlx-dual-bench")

        let audio = try readBenchmarkWav(sourceURL)
        let primaryBundle = mlxBundleURL(
            "encodec_48khz_12kbps_\(profileMilliseconds)ms"
        )
        let derivedBundle = mlxBundleURL(
            "encodec_48khz_6kbps_\(profileMilliseconds)ms"
        )
        let primaryPipeline = try MLXEncodecNativePipeline(bundleURL: primaryBundle)
        let derivedPipeline = try MLXEncodecNativePipeline(bundleURL: derivedBundle)
        XCTAssertEqual(audio.sampleRate, primaryPipeline.summary.sampleRate)
        XCTAssertEqual(audio.channels, primaryPipeline.summary.channels)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let warmupFrames = min(audio.frameCount, audio.sampleRate)
        let warmupSamples = Array(audio.samples.prefix(warmupFrames * audio.channels))
        _ = try primaryPipeline.encodeEcdcOutputs(
            samples: warmupSamples,
            channels: audio.channels,
            derived6KBundleURL: derivedBundle,
            frameBatchSize: frameBatchSize,
            chunkMilliseconds: chunkMilliseconds
        )
        _ = try derivedPipeline.encodeEcdc(
            samples: warmupSamples,
            channels: audio.channels,
            frameBatchSize: frameBatchSize,
            chunkMilliseconds: chunkMilliseconds
        )

        let primaryStart = BenchmarkClock.now()
        let primaryOnly = try primaryPipeline.encodeEcdc(
            samples: audio.samples,
            channels: audio.channels,
            frameBatchSize: frameBatchSize,
            chunkMilliseconds: chunkMilliseconds
        )
        let primaryElapsed = primaryStart.elapsed()

        let dualStart = BenchmarkClock.now()
        let dual = try primaryPipeline.encodeEcdcOutputs(
            samples: audio.samples,
            channels: audio.channels,
            derived6KBundleURL: derivedBundle,
            frameBatchSize: frameBatchSize,
            chunkMilliseconds: chunkMilliseconds
        )
        let dualElapsed = dualStart.elapsed()

        let separateStart = BenchmarkClock.now()
        let separatePrimary = try primaryPipeline.encodeEcdc(
            samples: audio.samples,
            channels: audio.channels,
            frameBatchSize: frameBatchSize,
            chunkMilliseconds: chunkMilliseconds
        )
        let separateDerived = try derivedPipeline.encodeEcdc(
            samples: audio.samples,
            channels: audio.channels,
            frameBatchSize: frameBatchSize,
            chunkMilliseconds: chunkMilliseconds
        )
        let separateElapsed = separateStart.elapsed()

        XCTAssertEqual(dual.primary, primaryOnly)
        XCTAssertEqual(separatePrimary, primaryOnly)
        XCTAssertEqual(dual.derived6K, separateDerived)
        guard let dualDerived = dual.derived6K else {
            XCTFail("dual encode did not return its requested 6 kbps stream")
            return
        }

        let stem = sourceURL.deletingPathExtension().lastPathComponent
        try dual.primary.write(
            to: outputDir.appendingPathComponent("\(stem).dual.12kbps.ecdc")
        )
        try dualDerived.write(
            to: outputDir.appendingPathComponent("\(stem).dual.6kbps.ecdc")
        )

        let duration = audio.durationSeconds
        print(
            "dual_benchmark: profile_ms=\(profileMilliseconds) " +
            "duration_s=\(String(format: "%.3f", duration)) " +
            "frame_batch_size=\(frameBatchSize)"
        )
        print(
            "dual_benchmark: mode=12k_only elapsed_s=\(String(format: "%.3f", primaryElapsed)) " +
            "rtfx=\(String(format: "%.3f", duration / primaryElapsed)) " +
            "bytes_12k=\(primaryOnly.count)"
        )
        print(
            "dual_benchmark: mode=12k_plus_derived_6k elapsed_s=\(String(format: "%.3f", dualElapsed)) " +
            "rtfx=\(String(format: "%.3f", duration / dualElapsed)) " +
            "overhead_vs_12k_pct=\(String(format: "%.1f", (dualElapsed / primaryElapsed - 1) * 100)) " +
            "bytes_12k=\(dual.primary.count) bytes_6k=\(dualDerived.count)"
        )
        print(
            "dual_benchmark: mode=separate_12k_and_6k elapsed_s=\(String(format: "%.3f", separateElapsed)) " +
            "rtfx=\(String(format: "%.3f", duration / separateElapsed)) " +
            "dual_speedup=\(String(format: "%.3f", separateElapsed / dualElapsed))"
        )
        print("dual_benchmark: exact_primary=true exact_derived=true output_dir=\(outputDir.path)")
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

private struct FramePhaseProfile {
    let backendInitializationMs: Double
    let hostInputCopyMs: Double
    let encodeGraphConstructionMs: Double
    let encodeExecutionSubmitMs: Double
    let encodeSynchronizationWaitMs: Double
    let codeHostReadbackMs: Double
    let decodeGraphConstructionMs: Double
    let decodeExecutionSubmitMs: Double
    let decodeSynchronizationWaitMs: Double
    let pcmHostReadbackMs: Double

    var json: [String: Double] {
        [
            "backend_initialization_ms": backendInitializationMs,
            "host_input_copy_ms": hostInputCopyMs,
            "encode_graph_construction_ms": encodeGraphConstructionMs,
            "encode_execution_submit_ms": encodeExecutionSubmitMs,
            "encode_synchronization_wait_ms": encodeSynchronizationWaitMs,
            "code_host_readback_ms": codeHostReadbackMs,
            "decode_graph_construction_ms": decodeGraphConstructionMs,
            "decode_execution_submit_ms": decodeExecutionSubmitMs,
            "decode_synchronization_wait_ms": decodeSynchronizationWaitMs,
            "pcm_host_readback_ms": pcmHostReadbackMs,
        ]
    }
}

private struct FrameResult {
    let codes: [Int64]
    let scale: [Float]
    let pcm: [Float]
    let profile: FramePhaseProfile
}

private struct FrameReference {
    let codes: [Int64]
    let scale: [Float]
    let pcm: [Float]
    let profile: [String: Any]
}

private struct PCMComparison {
    let maxAbs: Double
    let rmse: Double
}

private func evaluateBatchOneFrame(bundleURL: URL) throws -> FrameResult {
    let initializationStart = BenchmarkClock.now()
    let backend = try MLXEncodecFrameBackend(bundleURL: bundleURL)
    let backendInitializationMs = initializationStart.elapsed() * 1_000
    XCTAssertEqual(backend.metadata.channels, 2)
    let audio = makeDeterministicFrame(
        channels: backend.metadata.channels,
        samples: backend.metadata.segmentSamples
    )
    XCTAssertEqual(audio.count, backend.metadata.channels * backend.metadata.segmentSamples)

    let inputStart = BenchmarkClock.now()
    let audioArray = MLXArray(
        audio,
        [1, backend.metadata.channels, backend.metadata.segmentSamples]
    )
    let hostInputCopyMs = inputStart.elapsed() * 1_000

    let encodeGraphStart = BenchmarkClock.now()
    let encoded = try backend.encodeFrame(audio: audioArray)
    let callbackCodes = encoded.codes.asType(Int64.self)
    let encodeGraphConstructionMs = encodeGraphStart.elapsed() * 1_000
    XCTAssertEqual(encoded.codes.shape[0], 1, "acceptance benchmark must stay at batch one")

    let encodeSubmitStart = BenchmarkClock.now()
    asyncEval([callbackCodes, encoded.scale])
    let encodeExecutionSubmitMs = encodeSubmitStart.elapsed() * 1_000
    let encodeSynchronizationStart = BenchmarkClock.now()
    Stream.gpu.synchronize()
    let encodeSynchronizationWaitMs = encodeSynchronizationStart.elapsed() * 1_000

    let codeReadbackStart = BenchmarkClock.now()
    let codes = callbackCodes.asArray(Int64.self)
    let scale = encoded.scale.asArray(Float.self)
    let codeHostReadbackMs = codeReadbackStart.elapsed() * 1_000

    let decodeGraphStart = BenchmarkClock.now()
    let decoded = try backend.decodeFrame(codes: encoded.codes, scale: encoded.scale)
    let decodeGraphConstructionMs = decodeGraphStart.elapsed() * 1_000
    XCTAssertEqual(decoded.shape[0], 1, "acceptance benchmark must stay at batch one")

    let decodeSubmitStart = BenchmarkClock.now()
    asyncEval([decoded])
    let decodeExecutionSubmitMs = decodeSubmitStart.elapsed() * 1_000
    let decodeSynchronizationStart = BenchmarkClock.now()
    Stream.gpu.synchronize()
    let decodeSynchronizationWaitMs = decodeSynchronizationStart.elapsed() * 1_000

    let pcmReadbackStart = BenchmarkClock.now()
    let pcm = decoded.asArray(Float.self)
    let pcmHostReadbackMs = pcmReadbackStart.elapsed() * 1_000

    return FrameResult(
        codes: codes,
        scale: scale,
        pcm: pcm,
        profile: FramePhaseProfile(
            backendInitializationMs: backendInitializationMs,
            hostInputCopyMs: hostInputCopyMs,
            encodeGraphConstructionMs: encodeGraphConstructionMs,
            encodeExecutionSubmitMs: encodeExecutionSubmitMs,
            encodeSynchronizationWaitMs: encodeSynchronizationWaitMs,
            codeHostReadbackMs: codeHostReadbackMs,
            decodeGraphConstructionMs: decodeGraphConstructionMs,
            decodeExecutionSubmitMs: decodeExecutionSubmitMs,
            decodeSynchronizationWaitMs: decodeSynchronizationWaitMs,
            pcmHostReadbackMs: pcmHostReadbackMs
        )
    )
}

private func makeDeterministicFrame(channels: Int, samples: Int) -> [Float] {
    precondition(channels == 2)
    var audio = Array(repeating: Float(0), count: channels * samples)
    for sample in 0 ..< samples {
        let phase = Double(sample)
        audio[sample] = Float(0.16 * sin(phase * 0.011) + 0.03 * sin(phase * 0.037))
        audio[samples + sample] = Float(0.14 * cos(phase * 0.013) - 0.025 * sin(phase * 0.041))
    }
    return audio
}

private func maximumAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Double {
    guard lhs.count == rhs.count else {
        return .infinity
    }
    return zip(lhs, rhs).reduce(0) { maximum, values in
        max(maximum, abs(Double(values.0) - Double(values.1)))
    }
}

private func pcmComparison(_ baseline: [Float], _ candidate: [Float]) -> PCMComparison {
    guard baseline.count == candidate.count, !baseline.isEmpty else {
        return PCMComparison(maxAbs: .infinity, rmse: .infinity)
    }
    var maxAbs = 0.0
    var squaredError = 0.0
    for (reference, value) in zip(baseline, candidate) {
        let error = Double(value) - Double(reference)
        maxAbs = max(maxAbs, abs(error))
        squaredError += error * error
    }
    return PCMComparison(
        maxAbs: maxAbs,
        rmse: sqrt(squaredError / Double(baseline.count))
    )
}

private func printQualificationRecord(
    baseline: FrameReference,
    candidate: FrameResult,
    candidateLabel: String,
    codeParity: Bool,
    scaleMaxAbs: Double,
    pcm: PCMComparison,
    pcmMaxAbsTolerance: Double,
    pcmRMSETolerance: Double
) {
    let record: [String: Any] = [
        "schema": "encodec.mlx.frame-optimization-parity.v1",
        "configuration": "release",
        "bundle": "encodec_48khz_6kbps_1333ms",
        "frame_batch_size": 1,
        "lstm_loop_eval_interval": NSNull(),
        "candidate": candidateLabel,
        "codes_exact": codeParity,
        "code_count": baseline.codes.count,
        "code_hash_fnv1a64": codeHash(baseline.codes),
        "scale_max_abs": scaleMaxAbs,
        "pcm_max_abs": pcm.maxAbs,
        "pcm_rmse": pcm.rmse,
        "pcm_max_abs_tolerance": pcmMaxAbsTolerance,
        "pcm_rmse_tolerance": pcmRMSETolerance,
        "baseline_profile": baseline.profile,
        "candidate_profile": candidate.profile.json,
        "ffi_output_copy_ms": NSNull(),
        "ffi_note": "not applicable to direct-backend parity; callback profiler reports this separately",
    ]
    if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
       let json = String(data: data, encoding: .utf8)
    {
        print("mlx_acceptance: \(json)")
    }
}

private func recordFrameReference(_ result: FrameResult, at directory: URL) throws {
    let files = frameReferenceFiles(directory)
    let existing = [files.codes, files.scale, files.pcm, files.metadata].filter {
        FileManager.default.fileExists(atPath: $0.path)
    }
    guard existing.isEmpty else {
        throw EncodecMLXRuntimeError.unsupportedBundle(
            "refusing to overwrite MLX parity reference files in \(directory.path)"
        )
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try int64LittleEndianData(result.codes).write(to: files.codes, options: .atomic)
    try floatLittleEndianData(result.scale).write(to: files.scale, options: .atomic)
    try floatLittleEndianData(result.pcm).write(to: files.pcm, options: .atomic)
    let metadata: [String: Any] = [
        "schema": "encodec.mlx.frame-reference.v1",
        "configuration": "release",
        "bundle": "encodec_48khz_6kbps_1333ms",
        "frame_batch_size": 1,
        "lstm_loop_eval_interval": NSNull(),
        "code_count": result.codes.count,
        "code_hash_fnv1a64": codeHash(result.codes),
        "scale_count": result.scale.count,
        "pcm_count": result.pcm.count,
        "profile": result.profile.json,
    ]
    let data = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: files.metadata, options: .atomic)
}

private func loadFrameReference(from directory: URL) throws -> FrameReference {
    let files = frameReferenceFiles(directory)
    let metadataData = try Data(contentsOf: files.metadata)
    let metadata = try JSONSerialization.jsonObject(with: metadataData) as? [String: Any] ?? [:]
    guard metadata["configuration"] as? String == "release",
          metadata["bundle"] as? String == "encodec_48khz_6kbps_1333ms",
          metadata["frame_batch_size"] as? Int == 1
    else {
        throw EncodecMLXRuntimeError.unsupportedBundle(
            "MLX parity reference metadata is not Release/batch-one for the required bundle"
        )
    }
    let codes = try decodeInt64LittleEndian(Data(contentsOf: files.codes))
    let scale = try decodeFloatLittleEndian(Data(contentsOf: files.scale))
    let pcm = try decodeFloatLittleEndian(Data(contentsOf: files.pcm))
    guard metadata["code_count"] as? Int == codes.count,
          metadata["code_hash_fnv1a64"] as? String == codeHash(codes),
          metadata["scale_count"] as? Int == scale.count,
          metadata["pcm_count"] as? Int == pcm.count
    else {
        throw EncodecMLXRuntimeError.unsupportedBundle(
            "MLX parity reference files do not match their recorded counts/hash"
        )
    }
    return FrameReference(
        codes: codes,
        scale: scale,
        pcm: pcm,
        profile: metadata["profile"] as? [String: Any] ?? [:]
    )
}

private func printBaselineRecord(_ result: FrameResult, referenceURL: URL) {
    let record: [String: Any] = [
        "schema": "encodec.mlx.frame-optimization-parity.v1",
        "status": "baseline_recorded",
        "configuration": "release",
        "bundle": "encodec_48khz_6kbps_1333ms",
        "frame_batch_size": 1,
        "lstm_loop_eval_interval": NSNull(),
        "code_count": result.codes.count,
        "code_hash_fnv1a64": codeHash(result.codes),
        "profile": result.profile.json,
        "reference_dir": referenceURL.path,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
       let json = String(data: data, encoding: .utf8)
    {
        print("mlx_acceptance: \(json)")
    }
}

private func frameReferenceFiles(_ directory: URL) -> (
    codes: URL, scale: URL, pcm: URL, metadata: URL
) {
    (
        directory.appendingPathComponent("codes.i64le"),
        directory.appendingPathComponent("scale.f32le"),
        directory.appendingPathComponent("decoded-pcm.f32le"),
        directory.appendingPathComponent("baseline.json")
    )
}

private func int64LittleEndianData(_ values: [Int64]) -> Data {
    var data = Data(capacity: values.count * MemoryLayout<Int64>.size)
    for value in values {
        var bits = UInt64(bitPattern: value).littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}

private func floatLittleEndianData(_ values: [Float]) -> Data {
    var data = Data(capacity: values.count * MemoryLayout<Float>.size)
    for value in values {
        var bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    return data
}

private func decodeInt64LittleEndian(_ data: Data) throws -> [Int64] {
    guard data.count % MemoryLayout<Int64>.size == 0 else {
        throw EncodecMLXRuntimeError.unsupportedBundle("unaligned MLX reference code data")
    }
    return stride(from: 0, to: data.count, by: 8).map { offset in
        var bits = UInt64(0)
        for byte in 0 ..< 8 {
            bits |= UInt64(data[offset + byte]) << UInt64(byte * 8)
        }
        return Int64(bitPattern: bits)
    }
}

private func decodeFloatLittleEndian(_ data: Data) throws -> [Float] {
    guard data.count % MemoryLayout<Float>.size == 0 else {
        throw EncodecMLXRuntimeError.unsupportedBundle("unaligned MLX reference PCM data")
    }
    return stride(from: 0, to: data.count, by: 4).map { offset in
        Float(bitPattern: readUInt32LE(data, offset))
    }
}

private func codeHash(_ codes: [Int64]) -> String {
    var hash = UInt64(0xcbf29ce484222325)
    for code in codes {
        var value = UInt64(bitPattern: code).littleEndian
        withUnsafeBytes(of: &value) { bytes in
            for byte in bytes {
                hash ^= UInt64(byte)
                hash &*= 0x100000001b3
            }
        }
    }
    return String(format: "%016llx", hash)
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
    var validBitsPerSample: UInt16?
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
            var resolvedAudioFormat = readUInt16LE(data, chunkDataOffset)
            channels = Int(readUInt16LE(data, chunkDataOffset + 2))
            sampleRate = Int(readUInt32LE(data, chunkDataOffset + 4))
            let containerBits = readUInt16LE(data, chunkDataOffset + 14)
            bitsPerSample = containerBits
            validBitsPerSample = containerBits

            if resolvedAudioFormat == 0xfffe {
                guard chunkSize >= 40, readUInt16LE(data, chunkDataOffset + 16) >= 22 else {
                    throw XCTSkip("short WAVE_FORMAT_EXTENSIBLE fmt chunk in \(url.path)")
                }
                let subformatTail: [UInt8] = [
                    0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
                    0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
                ]
                let subformat = readUInt32LE(data, chunkDataOffset + 24)
                guard subformat <= UInt16.max,
                      Array(data[chunkDataOffset + 28 ..< chunkDataOffset + 40]) == subformatTail
                else {
                    throw XCTSkip("unsupported WAVE_FORMAT_EXTENSIBLE subtype in \(url.path)")
                }
                resolvedAudioFormat = UInt16(subformat)
                validBitsPerSample = readUInt16LE(data, chunkDataOffset + 18)
            }
            audioFormat = resolvedAudioFormat
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
          let validBitsPerSample,
          let dataOffset,
          let dataSize
    else {
        throw XCTSkip("missing WAV fmt/data chunk in \(url.path)")
    }
    guard channels > 0 else {
        throw XCTSkip("benchmark WAV has no channels: \(url.path)")
    }
    guard validBitsPerSample > 0, validBitsPerSample <= bitsPerSample else {
        throw XCTSkip(
            "invalid benchmark WAV valid/container bit depths " +
                "\(validBitsPerSample)/\(bitsPerSample) in \(url.path)"
        )
    }

    let samples: [Float]
    switch (audioFormat, bitsPerSample) {
    case (1, 16):
        guard dataSize % 2 == 0 else {
            throw XCTSkip("unaligned 16-bit WAV data in \(url.path)")
        }
        let shift = Int(bitsPerSample - validBitsPerSample)
        let denominator = Float(1 << (Int(validBitsPerSample) - 1))
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 2).map { index in
            let sample = Int32(Int16(bitPattern: readUInt16LE(data, index))) >> shift
            return Float(sample) / denominator
        }
    case (1, 24):
        guard dataSize % 3 == 0 else {
            throw XCTSkip("unaligned 24-bit WAV data in \(url.path)")
        }
        let shift = Int(bitsPerSample - validBitsPerSample)
        let denominator = Float(1 << (Int(validBitsPerSample) - 1))
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 3).map { index in
            let packed = UInt32(data[index])
                | (UInt32(data[index + 1]) << 8)
                | (UInt32(data[index + 2]) << 16)
            let sample = (Int32(bitPattern: packed << 8) >> 8) >> shift
            return Float(sample) / denominator
        }
    case (1, 32):
        guard dataSize % 4 == 0 else {
            throw XCTSkip("unaligned 32-bit PCM WAV data in \(url.path)")
        }
        let shift = Int(bitsPerSample - validBitsPerSample)
        let denominator = Float(UInt64(1) << UInt64(validBitsPerSample - 1))
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 4).map { index in
            let sample = Int32(bitPattern: readUInt32LE(data, index)) >> shift
            return Float(sample) / denominator
        }
    case (3, 32):
        guard validBitsPerSample == 32, dataSize % 4 == 0 else {
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
