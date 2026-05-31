use std::path::PathBuf;
use std::process::{Command, Stdio};

#[test]
#[ignore = "runs the restartable mobygratis MLX streaming encoder"]
fn encode_mobygratis_with_mlx_streaming() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("scripts").join("encode_mobygratis_mlx.sh");
    let mut command = Command::new("bash");
    command
        .arg(script)
        .current_dir(&root)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if std::env::var_os("LIMIT").is_none() {
        command.env("LIMIT", "1");
    }
    if std::env::var_os("SOURCE_DIR").is_none() {
        command.env("SOURCE_DIR", root.join("../bitneedle/mobygratis"));
    }
    if std::env::var_os("OUT_DIR").is_none() {
        command.env("OUT_DIR", root.join("target/mobygratis-ecdc"));
    }
    if std::env::var_os("BUNDLE_DIR").is_none() {
        command.env(
            "BUNDLE_DIR",
            root.join("target/mlx-bundles/encodec_48khz_12kbps_1333ms"),
        );
    }
    if std::env::var_os("BATCH_SIZE").is_none() {
        command.env("BATCH_SIZE", "8");
    }
    if std::env::var_os("CHUNK_MS").is_none() {
        command.env("CHUNK_MS", "1333.333333");
    }

    let status = command
        .status()
        .expect("failed to launch scripts/encode_mobygratis_mlx.sh");
    assert!(
        status.success(),
        "mobygratis MLX streaming encoder failed: {status}"
    );
}

#[test]
#[ignore = "runs the restartable mobygratis MLX decoder and caches SoundKit Opus packets"]
fn decode_mobygratis_with_mlx_soundkit_opus_cache() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root
        .join("scripts")
        .join("decode_mobygratis_mlx_opus_cache.sh");
    let mut command = Command::new("bash");
    command
        .arg(script)
        .current_dir(&root)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if std::env::var_os("LIMIT").is_none() {
        command.env("LIMIT", "1");
    }
    if std::env::var_os("OUT_DIR").is_none() {
        command.env("OUT_DIR", root.join("target/mobygratis-ecdc"));
    }
    if std::env::var_os("ECDC_DIR").is_none() {
        command.env("ECDC_DIR", root.join("target/mobygratis-ecdc/ecdc"));
    }
    if std::env::var_os("BUNDLE_DIR").is_none() {
        command.env(
            "BUNDLE_DIR",
            root.join("target/mlx-bundles/encodec_48khz_12kbps_1333ms_mobygratisv0"),
        );
    }
    if std::env::var_os("OPUS_BITRATE").is_none() {
        command.env("OPUS_BITRATE", "64000");
    }
    if std::env::var_os("OPUS_FRAME_MS").is_none() {
        command.env("OPUS_FRAME_MS", "20");
    }

    let status = command
        .status()
        .expect("failed to launch scripts/decode_mobygratis_mlx_opus_cache.sh");
    assert!(
        status.success(),
        "mobygratis MLX SoundKit Opus cache decoder failed: {status}"
    );
}
