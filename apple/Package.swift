// swift-tools-version: 5.10
import PackageDescription

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
            path: "Sources/CEncodecMLXBridge",
            publicHeadersPath: "include"
        ),
        .target(
            name: "EncodecMLXRuntime",
            dependencies: [
                "CEncodecMLXBridge",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/EncodecMLXRuntime"
        ),
        .testTarget(
            name: "EncodecMLXRuntimeTests",
            dependencies: ["EncodecMLXRuntime"],
            path: "Tests/EncodecMLXRuntimeTests",
            linkerSettings: [
                .unsafeFlags([
                    "-L../target/release/deps",
                    "-L../target/release",
                    "-L../target/debug/deps",
                    "-L../target/debug",
                ]),
                .linkedLibrary("encodec_rs", .when(platforms: [.macOS])),
            ]
        ),
        .executableTarget(
            name: "EncodecMLXEncode",
            dependencies: ["EncodecMLXRuntime"],
            path: "Sources/EncodecMLXEncode",
            linkerSettings: [
                .unsafeFlags([
                    "-L../target/release/deps",
                    "-L../target/release",
                    "-L../target/debug/deps",
                    "-L../target/debug",
                ]),
                .linkedLibrary("encodec_rs", .when(platforms: [.macOS])),
            ]
        ),
    ]
)
