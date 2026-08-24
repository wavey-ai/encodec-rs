// swift-tools-version: 5.10
import Foundation
import PackageDescription

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let rustLibrarySearchFlags = [
    "-L\(packageRoot.appendingPathComponent("target/release/deps").path)",
    "-L\(packageRoot.appendingPathComponent("target/release").path)",
    "-L\(packageRoot.appendingPathComponent("target/debug/deps").path)",
    "-L\(packageRoot.appendingPathComponent("target/debug").path)",
]

let package = Package(
    name: "EncodecMLXRuntime",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "EncodecMLXRuntime", targets: ["EncodecMLXRuntime"]),
        .executable(name: "EncodecMLXEncode", targets: ["EncodecMLXEncode"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.3"),
    ],
    targets: [
        .target(
            name: "CEncodecMLXBridge",
            path: "apple/Sources/CEncodecMLXBridge",
            publicHeadersPath: "include"
        ),
        .target(
            name: "EncodecMLXRuntime",
            dependencies: [
                "CEncodecMLXBridge",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "apple/Sources/EncodecMLXRuntime"
        ),
        .testTarget(
            name: "EncodecMLXRuntimeTests",
            dependencies: ["EncodecMLXRuntime"],
            path: "apple/Tests/EncodecMLXRuntimeTests",
            linkerSettings: [
                .unsafeFlags(rustLibrarySearchFlags),
                .linkedLibrary("encodec_rs", .when(platforms: [.macOS])),
            ]
        ),
        .executableTarget(
            name: "EncodecMLXEncode",
            dependencies: ["EncodecMLXRuntime"],
            path: "apple/Sources/EncodecMLXEncode",
            linkerSettings: [
                .unsafeFlags(rustLibrarySearchFlags),
                .linkedLibrary("encodec_rs", .when(platforms: [.macOS])),
            ]
        ),
    ]
)
