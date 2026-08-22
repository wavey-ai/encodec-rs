import EncodecMLXRuntime
import Foundation

enum Mode {
    case encode
    case decode
}

struct Options {
    var mode: Mode = .encode
    var bundle: URL?
    var input: URL?
    var output: URL?
    var batchSize = 8
    var chunkMilliseconds: Double? = 1333.333333
    var useLM = true
    var streaming = true
    var progress: URL?
}

enum CliError: Error, LocalizedError {
    case usage(String)

    var errorDescription: String? {
        switch self {
        case let .usage(message):
            return message
        }
    }
}

func parseOptions(_ args: [String]) throws -> Options {
    var options = Options()
    var index = 0
    while index < args.count {
        let arg = args[index]
        func value() throws -> String {
            guard index + 1 < args.count else {
                throw CliError.usage("missing value for \(arg)")
            }
            index += 1
            return args[index]
        }

        switch arg {
        case "--encode":
            options.mode = .encode
        case "--decode":
            options.mode = .decode
        case "--bundle":
            options.bundle = URL(fileURLWithPath: try value())
        case "--input":
            options.input = URL(fileURLWithPath: try value())
        case "--output":
            options.output = URL(fileURLWithPath: try value())
        case "--progress":
            options.progress = URL(fileURLWithPath: try value())
        case "--batch-size":
            guard let parsed = Int(try value()), parsed > 0 else {
                throw CliError.usage("--batch-size must be a positive integer")
            }
            options.batchSize = parsed
        case "--chunk-ms":
            let raw = try value()
            if raw == "default" || raw == "none" {
                options.chunkMilliseconds = nil
            } else if let parsed = Double(raw), parsed > 0 {
                options.chunkMilliseconds = parsed
            } else {
                throw CliError.usage("--chunk-ms must be a positive number, default, or none")
            }
        case "--no-lm":
            options.useLM = false
        case "--stream":
            options.streaming = true
        case "--no-stream":
            options.streaming = false
        case "-h", "--help":
            throw CliError.usage(usage())
        default:
            throw CliError.usage("unknown argument: \(arg)\n\n\(usage())")
        }
        index += 1
    }
    guard options.bundle != nil, options.input != nil, options.output != nil else {
        throw CliError.usage(usage())
    }
    return options
}

func usage() -> String {
    """
    Usage:
      EncodecMLXEncode [--encode] --bundle DIR --input INPUT.wav --output OUTPUT.ecdc [--progress PROGRESS.json] [--batch-size N] [--chunk-ms MS] [--no-lm] [--no-stream]
      EncodecMLXEncode --decode --bundle DIR --input INPUT.ecdc --output OUTPUT.f32le [--batch-size N]
    """
}

struct WavAudio {
    let samples: [Float]
    let channels: Int
    let sampleRate: Int
    let frameCount: Int
}

func readWav(_ url: URL) throws -> WavAudio {
    let data = try Data(contentsOf: url)
    guard data.count >= 44, ascii(data, 0, 4) == "RIFF", ascii(data, 8, 4) == "WAVE" else {
        throw CliError.usage("unsupported WAV container: \(url.path)")
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
        let body = offset + 8
        guard body + chunkSize <= data.count else {
            throw CliError.usage("truncated WAV chunk \(chunkID): \(url.path)")
        }
        if chunkID == "fmt " {
            guard chunkSize >= 16 else {
                throw CliError.usage("short WAV fmt chunk: \(url.path)")
            }
            var resolvedAudioFormat = readUInt16LE(data, body)
            channels = Int(readUInt16LE(data, body + 2))
            sampleRate = Int(readUInt32LE(data, body + 4))
            let containerBits = readUInt16LE(data, body + 14)
            bitsPerSample = containerBits
            validBitsPerSample = containerBits

            if resolvedAudioFormat == 0xfffe {
                guard chunkSize >= 40, readUInt16LE(data, body + 16) >= 22 else {
                    throw CliError.usage("short WAVE_FORMAT_EXTENSIBLE fmt chunk: \(url.path)")
                }
                let subformatTail: [UInt8] = [
                    0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
                    0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
                ]
                let subformat = readUInt32LE(data, body + 24)
                guard subformat <= UInt16.max,
                      Array(data[body + 28 ..< body + 40]) == subformatTail
                else {
                    throw CliError.usage("unsupported WAVE_FORMAT_EXTENSIBLE subtype: \(url.path)")
                }
                resolvedAudioFormat = UInt16(subformat)
                validBitsPerSample = readUInt16LE(data, body + 18)
            }
            audioFormat = resolvedAudioFormat
        } else if chunkID == "data" {
            dataOffset = body
            dataSize = chunkSize
        }
        offset = body + chunkSize + (chunkSize & 1)
    }

    guard let audioFormat,
          let channels,
          let sampleRate,
          let bitsPerSample,
          let validBitsPerSample,
          let dataOffset,
          let dataSize
    else {
        throw CliError.usage("missing WAV fmt/data chunk: \(url.path)")
    }
    guard channels > 0 else {
        throw CliError.usage("WAV channel count must be positive: \(url.path)")
    }
    guard validBitsPerSample > 0, validBitsPerSample <= bitsPerSample else {
        throw CliError.usage(
            "invalid WAV valid/container bit depths \(validBitsPerSample)/\(bitsPerSample): \(url.path)"
        )
    }

    let samples: [Float]
    switch (audioFormat, bitsPerSample) {
    case (1, 16):
        guard dataSize % 2 == 0 else {
            throw CliError.usage("unaligned 16-bit WAV data: \(url.path)")
        }
        let shift = Int(bitsPerSample - validBitsPerSample)
        let denominator = Float(1 << (Int(validBitsPerSample) - 1))
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 2).map { index in
            let sample = Int32(Int16(bitPattern: readUInt16LE(data, index))) >> shift
            return Float(sample) / denominator
        }
    case (1, 24):
        guard dataSize % 3 == 0 else {
            throw CliError.usage("unaligned 24-bit WAV data: \(url.path)")
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
            throw CliError.usage("unaligned 32-bit PCM WAV data: \(url.path)")
        }
        let shift = Int(bitsPerSample - validBitsPerSample)
        let denominator = Float(UInt64(1) << UInt64(validBitsPerSample - 1))
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 4).map { index in
            let sample = Int32(bitPattern: readUInt32LE(data, index)) >> shift
            return Float(sample) / denominator
        }
    case (3, 32):
        guard validBitsPerSample == 32, dataSize % 4 == 0 else {
            throw CliError.usage("invalid 32-bit float WAV data: \(url.path)")
        }
        samples = stride(from: dataOffset, to: dataOffset + dataSize, by: 4).map { index in
            Float(bitPattern: readUInt32LE(data, index))
        }
    default:
        throw CliError.usage("unsupported WAV format \(audioFormat)/\(bitsPerSample): \(url.path)")
    }

    guard samples.count % channels == 0 else {
        throw CliError.usage("WAV sample count is not divisible by channels: \(url.path)")
    }

    return WavAudio(
        samples: samples,
        channels: channels,
        sampleRate: sampleRate,
        frameCount: samples.count / channels
    )
}

func ascii(_ data: Data, _ offset: Int, _ count: Int) -> String {
    guard offset >= 0, count >= 0, offset + count <= data.count else {
        return ""
    }
    return String(decoding: data[offset ..< offset + count], as: UTF8.self)
}

func readUInt16LE(_ data: Data, _ offset: Int) -> UInt16 {
    UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
}

func readUInt32LE(_ data: Data, _ offset: Int) -> UInt32 {
    UInt32(data[offset])
        | (UInt32(data[offset + 1]) << 8)
        | (UInt32(data[offset + 2]) << 16)
        | (UInt32(data[offset + 3]) << 24)
}

func jsonString(_ value: String) -> String {
    let data = try! JSONEncoder().encode(value)
    return String(decoding: data, as: UTF8.self)
}

func writePcmF32LE(_ url: URL, samples: [Float]) throws -> Int {
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var data = Data(capacity: samples.count * MemoryLayout<Float>.size)
    for sample in samples {
        var bits = sample.bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
    }
    try data.write(to: url, options: .atomic)
    return data.count
}

do {
    let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
    let bundle = options.bundle!
    let input = options.input!
    let output = options.output!
    let pipeline = try MLXEncodecNativePipeline(bundleURL: bundle)
    if options.mode == .decode {
        let payload = try Data(contentsOf: input)
        let started = DispatchTime.now().uptimeNanoseconds
        let decoded = try pipeline.decodeEcdc(payload, frameBatchSize: options.batchSize)
        let bytes = try writePcmF32LE(output, samples: decoded.samples)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
        let sampleRate = pipeline.summary.sampleRate
        let duration = Double(decoded.frameCount) / Double(sampleRate)
        print(
            "{" +
                "\"input\":\(jsonString(input.path))," +
                "\"output\":\(jsonString(output.path))," +
                "\"bundle\":\(jsonString(bundle.path))," +
                "\"sample_rate\":\(sampleRate)," +
                "\"channels\":\(decoded.channels)," +
                "\"frames\":\(decoded.frameCount)," +
                "\"duration_s\":\(duration)," +
                "\"decode_s\":\(elapsed)," +
                "\"decode_rtfx\":\(duration / elapsed)," +
                "\"pcm_format\":\(jsonString("f32le"))," +
                "\"bytes\":\(bytes)" +
            "}"
        )
        exit(0)
    }
    let audio = try readWav(input)
    let started = DispatchTime.now().uptimeNanoseconds
    let bytes: Int
    if options.streaming {
        let result = try pipeline.encodeEcdcStreaming(
            samples: audio.samples,
            channels: audio.channels,
            outputURL: output,
            progressURL: options.progress,
            useLM: options.useLM,
            frameBatchSize: options.batchSize,
            chunkMilliseconds: options.chunkMilliseconds
        )
        bytes = result.bytesWritten
    } else {
        let payload = try pipeline.encodeEcdc(
            samples: audio.samples,
            channels: audio.channels,
            useLM: options.useLM,
            frameBatchSize: options.batchSize,
            chunkMilliseconds: options.chunkMilliseconds
        )
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try payload.write(to: output, options: .atomic)
        bytes = payload.count
    }
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
    let duration = Double(audio.frameCount) / Double(audio.sampleRate)
    print(
        "{" +
            "\"input\":\(jsonString(input.path))," +
            "\"output\":\(jsonString(output.path))," +
            "\"bundle\":\(jsonString(bundle.path))," +
            "\"sample_rate\":\(audio.sampleRate)," +
            "\"channels\":\(audio.channels)," +
            "\"frames\":\(audio.frameCount)," +
            "\"duration_s\":\(duration)," +
            "\"encode_s\":\(elapsed)," +
            "\"encode_rtfx\":\(duration / elapsed)," +
            "\"streaming\":\(options.streaming)," +
            "\"bytes\":\(bytes)" +
        "}"
    )
} catch {
    fputs("\(error.localizedDescription)\n", stderr)
    exit(1)
}
