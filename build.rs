use std::path::Path;

fn main() {
    let kernel_sources = [
        "scripts/kernels/encodec_convtranspose.c",
        "scripts/kernels/encodec_encoder.c",
        "scripts/kernels/native/wasm_simd128.h",
        "scripts/kernels/native/emscripten/emscripten.h",
    ];
    for source in kernel_sources {
        println!("cargo:rerun-if-changed={source}");
    }

    if std::env::var_os("CARGO_FEATURE_NATIVE_KERNEL").is_none() {
        return;
    }

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "wasm32" {
        return;
    }

    let shim_dir = Path::new("scripts/kernels/native");
    cc::Build::new()
        .files([
            "scripts/kernels/encodec_convtranspose.c",
            "scripts/kernels/encodec_encoder.c",
        ])
        .include(shim_dir)
        // Match the non-relaxed WASM kernel: no contraction so the arithmetic
        // order is identical across the two backends.
        .flag("-ffp-contract=off")
        .opt_level(3)
        .warnings(false)
        .compile("encodec_custom_kernel");
}
