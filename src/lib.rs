use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use thiserror::Error;

#[derive(Debug, Clone, Default)]
pub struct EncodecOptions {
    pub bandwidth: Option<f32>,
    pub high_quality: bool,
    pub language_model: bool,
    pub force: bool,
    pub rescale: bool,
}

#[derive(Debug, Clone)]
enum Launcher {
    Binary { program: PathBuf },
    PythonModule { python: PathBuf },
}

#[derive(Debug, Clone)]
pub struct Encodec {
    launcher: Launcher,
}

#[derive(Debug, Error)]
pub enum EncodecError {
    #[error("failed to launch encodec command: {0}")]
    Io(#[from] std::io::Error),
    #[error("encodec command failed with status {status}: {command}")]
    CommandFailed { command: String, status: ExitStatus },
}

impl Encodec {
    pub fn new() -> Self {
        Self {
            launcher: Launcher::Binary {
                program: PathBuf::from("encodec"),
            },
        }
    }

    pub fn with_binary(program: impl Into<PathBuf>) -> Self {
        Self {
            launcher: Launcher::Binary {
                program: program.into(),
            },
        }
    }

    pub fn with_python_module(python: impl Into<PathBuf>) -> Self {
        Self {
            launcher: Launcher::PythonModule {
                python: python.into(),
            },
        }
    }

    pub fn from_env() -> Self {
        if let Ok(program) = std::env::var("ENCODEC_BIN") {
            return Self::with_binary(program);
        }
        if let Ok(python) = std::env::var("ENCODEC_PYTHON") {
            return Self::with_python_module(python);
        }
        Self::new()
    }

    pub fn encode_file(
        &self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
        options: &EncodecOptions,
    ) -> Result<(), EncodecError> {
        let args = Self::build_encode_args(input.as_ref(), output.as_ref(), options);
        self.run(&args)
    }

    pub fn decode_file(
        &self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
        options: &EncodecOptions,
    ) -> Result<(), EncodecError> {
        let args = Self::build_decode_args(input.as_ref(), output.as_ref(), options);
        self.run(&args)
    }

    pub fn roundtrip_to_wav(
        &self,
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
        options: &EncodecOptions,
    ) -> Result<(), EncodecError> {
        let args = Self::build_encode_args(input.as_ref(), output.as_ref(), options);
        self.run(&args)
    }

    fn run(&self, args: &[OsString]) -> Result<(), EncodecError> {
        let mut command = match &self.launcher {
            Launcher::Binary { program } => Command::new(program),
            Launcher::PythonModule { python } => {
                let mut command = Command::new(python);
                command.arg("-m").arg("encodec");
                command
            }
        };
        command.args(args);

        let rendered = render_command(&command);
        let status = command.status()?;
        if status.success() {
            return Ok(());
        }

        Err(EncodecError::CommandFailed {
            command: rendered,
            status,
        })
    }

    fn build_encode_args(input: &Path, output: &Path, options: &EncodecOptions) -> Vec<OsString> {
        let mut args = Vec::new();
        if let Some(bandwidth) = options.bandwidth {
            args.push(OsString::from("-b"));
            args.push(OsString::from(bandwidth.to_string()));
        }
        if options.high_quality {
            args.push(OsString::from("--hq"));
        }
        if options.language_model {
            args.push(OsString::from("--lm"));
        }
        if options.force {
            args.push(OsString::from("--force"));
        }
        if options.rescale {
            args.push(OsString::from("--rescale"));
        }
        args.push(input.as_os_str().to_os_string());
        args.push(output.as_os_str().to_os_string());
        args
    }

    fn build_decode_args(input: &Path, output: &Path, options: &EncodecOptions) -> Vec<OsString> {
        let mut args = Vec::new();
        if options.force {
            args.push(OsString::from("--force"));
        }
        if options.rescale {
            args.push(OsString::from("--rescale"));
        }
        args.push(input.as_os_str().to_os_string());
        args.push(output.as_os_str().to_os_string());
        args
    }
}

impl Default for Encodec {
    fn default() -> Self {
        Self::new()
    }
}

fn render_command(command: &Command) -> String {
    let mut parts = Vec::new();
    parts.push(os_to_string(command.get_program()));
    parts.extend(command.get_args().map(os_to_string));
    parts.join(" ")
}

fn os_to_string(value: &OsStr) -> String {
    value.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_encode_args_includes_codec_flags() {
        let args = Encodec::build_encode_args(
            Path::new("input.wav"),
            Path::new("output.ecdc"),
            &EncodecOptions {
                bandwidth: Some(6.0),
                high_quality: true,
                language_model: true,
                force: true,
                rescale: true,
            },
        );
        let rendered: Vec<String> = args
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            rendered,
            vec![
                "-b",
                "6",
                "--hq",
                "--lm",
                "--force",
                "--rescale",
                "input.wav",
                "output.ecdc",
            ]
        );
    }

    #[test]
    fn env_prefers_explicit_binary() {
        let _guard = EnvGuard::set("ENCODEC_BIN", Some("custom-encodec"));
        let _clear_python = EnvGuard::set("ENCODEC_PYTHON", None);
        let encodec = Encodec::from_env();
        match encodec.launcher {
            Launcher::Binary { program } => assert_eq!(program, PathBuf::from("custom-encodec")),
            Launcher::PythonModule { .. } => panic!("expected binary launcher"),
        }
    }

    #[test]
    fn env_falls_back_to_python_module() {
        let _clear_bin = EnvGuard::set("ENCODEC_BIN", None);
        let _guard = EnvGuard::set("ENCODEC_PYTHON", Some("python3.12"));
        let encodec = Encodec::from_env();
        match encodec.launcher {
            Launcher::PythonModule { python } => assert_eq!(python, PathBuf::from("python3.12")),
            Launcher::Binary { .. } => panic!("expected python launcher"),
        }
    }

    struct EnvGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var_os(key);
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
            Self { key, original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}
