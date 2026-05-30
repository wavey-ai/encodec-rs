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
        command.env("BUNDLE_DIR", root.join("target/mlx-bundles/encodec_48khz_12kbps"));
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
    assert!(status.success(), "mobygratis MLX streaming encoder failed: {status}");
}
