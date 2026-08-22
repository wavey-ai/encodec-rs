import AVFoundation
import Foundation
import SwiftUI
import EncodecMLXRuntime

@main
struct EncodecMLXDeviceBenchApp: App {
    @StateObject private var model = DeviceBenchModel()

    var body: some Scene {
        WindowGroup {
            DeviceBenchView(model: model)
                .task {
                    model.start()
                }
        }
    }
}

private struct DeviceBenchView: View {
    @ObservedObject var model: DeviceBenchModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Encodec MLX device test")
                    .font(.title2.weight(.bold))

                Text("Lori Asha — Confirmation")
                    .font(.headline)

                Text("This app tests only encodec-rs, MLX, and the bundled model.")
                    .foregroundStyle(.secondary)

                metric("Status", model.status)
                metric("Encode", model.encodeProgress)
                metric("Decode", model.decodeProgress)

                if !model.results.isEmpty {
                    Divider()
                    Text(model.results)
                        .textSelection(.enabled)
                }

                if model.finished {
                    Button("Run again") {
                        model.start(force: true)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .fontDesign(.monospaced)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(24)
        }
        .preferredColorScheme(.dark)
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.body.monospacedDigit())
        }
    }
}

@MainActor
private final class DeviceBenchModel: ObservableObject {
    @Published var status = "Ready"
    @Published var encodeProgress = "Waiting"
    @Published var decodeProgress = "Waiting"
    @Published var results = ""
    @Published var finished = false

    private var running = false

    func start(force: Bool = false) {
        guard !running, force || !finished else {
            return
        }
        running = true
        finished = false
        status = "Starting"
        encodeProgress = "Waiting"
        decodeProgress = "Waiting"
        results = ""

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let result = try DeviceBenchmark.run { update in
                    DispatchQueue.main.async {
                        self?.apply(update)
                    }
                }
                print("ENCODEC_MLX_DEVICE_BENCH_RESULT\n\(result.description)")
                DispatchQueue.main.async {
                    self?.status = "Complete"
                    self?.results = result.description
                    self?.running = false
                    self?.finished = true
                }
            } catch {
                print("ENCODEC_MLX_DEVICE_BENCH_FAILURE\n\(error.localizedDescription)")
                DispatchQueue.main.async {
                    self?.status = "Failed"
                    self?.results = error.localizedDescription
                    self?.running = false
                    self?.finished = true
                }
            }
        }
    }

    private func apply(_ update: BenchUpdate) {
        switch update {
        case let .status(value):
            status = value
        case let .encode(value):
            encodeProgress = value
        case let .decode(value):
            decodeProgress = value
        }
    }
}

private enum BenchUpdate {
    case status(String)
    case encode(String)
    case decode(String)
}

private struct DeviceBenchmarkResult: Encodable {
    let duration: Double
    let frameCount: Int
    let ecdcBytes: Int
    let ecdcHash: UInt64
    let decodedBytes: Int
    let encodeSeconds: Double
    let decodeSeconds: Double

    var description: String {
        [
            String(format: "Track: %.3f seconds (%d frames)", duration, frameCount),
            "Batch size: 1",
            String(format: "Encode: %.3f seconds = %.3f× realtime", encodeSeconds, duration / encodeSeconds),
            String(format: "Decode: %.3f seconds = %.3f× realtime", decodeSeconds, duration / decodeSeconds),
            "ECDC: \(ecdcBytes) bytes",
            String(format: "ECDC FNV-1a: %016llx", ecdcHash),
            "Decoded planar f32: \(decodedBytes) bytes",
        ].joined(separator: "\n")
    }
}

private enum DeviceBenchmark {
    private static let modelName = "encodec_48khz_12kbps_1333ms"
    private static let frameBatchSize = 1

    static func run(update: @escaping (BenchUpdate) -> Void) throws -> DeviceBenchmarkResult {
        guard let trackURL = Bundle.main.url(forResource: "confirmation", withExtension: "wav") else {
            throw BenchError.missingTrack
        }
        guard let resourceURL = Bundle.main.resourceURL else {
            throw BenchError.missingResources
        }
        let modelURL = resourceURL
            .appendingPathComponent("mlx-bundles", isDirectory: true)
            .appendingPathComponent(modelName, isDirectory: true)

        update(.status("Loading the Confirmation master"))
        var audio = try loadAudio(trackURL)
        guard audio.sampleRate == 48_000, audio.channels == 2 else {
            throw BenchError.unsupportedAudio(audio.sampleRate, audio.channels)
        }
        let duration = Double(audio.frameCount) / Double(audio.sampleRate)

        update(.status("Loading and prewarming MLX"))
        let pipeline = try MLXEncodecNativePipeline(bundleURL: modelURL)
        try pipeline.prewarm()

        let outputDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("encodec-mlx-device-bench", isDirectory: true)
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        let ecdcURL = outputDirectory.appendingPathComponent("confirmation.ecdc")
        let encodeProgressURL = outputDirectory.appendingPathComponent("encode-progress.json")
        let decodedURL = outputDirectory.appendingPathComponent("confirmation-planar.f32le")
        let decodeProgressURL = outputDirectory.appendingPathComponent("decode-progress.json")
        let resultURL = outputDirectory.appendingPathComponent("result.json")
        for url in [ecdcURL, encodeProgressURL, decodedURL, decodeProgressURL, resultURL] {
            try? FileManager.default.removeItem(at: url)
        }

        update(.status("Encoding the full track"))
        update(.encode("Starting"))
        let encodePoller = ProgressPoller(url: encodeProgressURL) { progress in
            update(.encode(progress.text(prefix: "Chunks")))
        }
        encodePoller.start()
        let encodeStarted = DispatchTime.now().uptimeNanoseconds
        let encodeResult: EncodecNativeStreamResult
        do {
            encodeResult = try pipeline.encodeEcdcStreaming(
                samples: audio.samples,
                channels: audio.channels,
                outputURL: ecdcURL,
                progressURL: encodeProgressURL,
                frameBatchSize: frameBatchSize
            )
        } catch {
            encodePoller.stop()
            throw error
        }
        let encodeSeconds = elapsedSeconds(since: encodeStarted)
        encodePoller.stop()
        update(.encode(String(format: "Complete — %.3f× realtime", duration / encodeSeconds)))

        audio.samples.removeAll(keepingCapacity: false)
        let payload = try Data(contentsOf: ecdcURL, options: .mappedIfSafe)

        update(.status("Decoding the full track"))
        update(.decode("Starting"))
        let decodePoller = ProgressPoller(url: decodeProgressURL) { progress in
            let label = progress.stage == "entropy" ? "Entropy chunks" : "Model frames"
            update(.decode(progress.text(prefix: label)))
        }
        decodePoller.start()
        let decodeStarted = DispatchTime.now().uptimeNanoseconds
        let decodeResult: EncodecNativeStreamResult
        do {
            decodeResult = try pipeline.decodeEcdcToPlanarF32File(
                payload,
                outputURL: decodedURL,
                progressURL: decodeProgressURL,
                frameBatchSize: frameBatchSize
            )
        } catch {
            decodePoller.stop()
            throw error
        }
        let decodeSeconds = elapsedSeconds(since: decodeStarted)
        decodePoller.stop()
        update(.decode(String(format: "Complete — %.3f× realtime", duration / decodeSeconds)))

        let expectedDecodedBytes = audio.frameCount * audio.channels * MemoryLayout<Float>.size
        guard decodeResult.bytesWritten == expectedDecodedBytes else {
            throw BenchError.decodedSize(decodeResult.bytesWritten, expectedDecodedBytes)
        }

        let result = DeviceBenchmarkResult(
            duration: duration,
            frameCount: audio.frameCount,
            ecdcBytes: encodeResult.bytesWritten,
            ecdcHash: fnv1a64(payload),
            decodedBytes: decodeResult.bytesWritten,
            encodeSeconds: encodeSeconds,
            decodeSeconds: decodeSeconds
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(result).write(to: resultURL, options: .atomic)
        return result
    }

    private static func elapsedSeconds(since started: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000.0
    }

    private static func fnv1a64(_ data: Data) -> UInt64 {
        data.reduce(14_695_981_039_346_656_037) { hash, byte in
            (hash ^ UInt64(byte)) &* 1_099_511_628_211
        }
    }

    private static func loadAudio(_ url: URL) throws -> LoadedAudio {
        let file = try AVAudioFile(
            forReading: url,
            commonFormat: .pcmFormatFloat32,
            interleaved: true
        )
        let format = file.processingFormat
        let capacity = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
            throw BenchError.audioBuffer
        }
        try file.read(into: buffer)
        let channels = Int(format.channelCount)
        let frameCount = Int(buffer.frameLength)
        let sampleCount = frameCount * channels
        let audioBuffer = buffer.audioBufferList.pointee.mBuffers
        guard let data = audioBuffer.mData else {
            throw BenchError.audioBuffer
        }
        let pointer = data.assumingMemoryBound(to: Float.self)
        let samples = Array(UnsafeBufferPointer(start: pointer, count: sampleCount))
        return LoadedAudio(
            samples: samples,
            channels: channels,
            sampleRate: Int(format.sampleRate),
            frameCount: frameCount
        )
    }
}

private struct LoadedAudio {
    var samples: [Float]
    let channels: Int
    let sampleRate: Int
    let frameCount: Int
}

private struct FileProgress: Decodable {
    let stage: String
    let completed: Int
    let total: Int

    func text(prefix: String) -> String {
        let percentage = total > 0 ? (Double(completed) / Double(total)) * 100.0 : 0.0
        return String(format: "%@: %d/%d (%.1f%%)", prefix, completed, total, percentage)
    }
}

private final class ProgressPoller {
    private let url: URL
    private let onProgress: (FileProgress) -> Void
    private let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
    private var started = false

    init(url: URL, onProgress: @escaping (FileProgress) -> Void) {
        self.url = url
        self.onProgress = onProgress
    }

    func start() {
        guard !started else {
            return
        }
        started = true
        timer.schedule(deadline: .now(), repeating: .milliseconds(150))
        timer.setEventHandler { [url, onProgress] in
            guard let data = try? Data(contentsOf: url),
                  let progress = try? JSONDecoder().decode(FileProgress.self, from: data)
            else {
                return
            }
            onProgress(progress)
        }
        timer.resume()
    }

    func stop() {
        guard started else {
            return
        }
        timer.cancel()
        started = false
    }
}

private enum BenchError: LocalizedError {
    case missingTrack
    case missingResources
    case unsupportedAudio(Int, Int)
    case audioBuffer
    case decodedSize(Int, Int)

    var errorDescription: String? {
        switch self {
        case .missingTrack:
            return "The bundled Confirmation master is missing."
        case .missingResources:
            return "The app resource directory is missing."
        case let .unsupportedAudio(sampleRate, channels):
            return "The track uses \(sampleRate) Hz and \(channels) channels. This test requires 48 kHz stereo."
        case .audioBuffer:
            return "The test could not allocate the audio buffer."
        case let .decodedSize(actual, expected):
            return "The decoder wrote \(actual) bytes. The test expected \(expected) bytes."
        }
    }
}
