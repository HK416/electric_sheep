//! Slang → SPIR-V, offline with a content-hash cache (spec §2.3, §11.4).
//!
//! The cache is content-addressed on *everything* the output depends on: source, entry,
//! profile, defines, requested execution modes, include directories and the `slangc` version
//! — spec §3.4 item 7 pins the compiler, not only the source. A hit does not start a
//! process, which is what makes `es task compile` able to package the cache into a bundle so
//! that a deployment target needs no Slang at all (§11.4), and what makes the cache
//! shareable between ranks (§22).

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::caps::ExecModes;
use crate::error::GpuError;
use crate::spirv;

/// A compiled SPIR-V module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpirvModule {
    pub words: Vec<u32>,
    /// Cache key = content hash of the compile inputs. The `compile_hash` ingredient of
    /// spec §11.4.
    pub hash: String,
    pub entry: String,
}

/// Invokes `slangc`, or serves the cache.
#[derive(Debug)]
pub struct SlangCompiler {
    exe: PathBuf,
    version: String,
    cache_dir: PathBuf,
    include_dirs: Vec<PathBuf>,
    invocations: Cell<usize>,
}

impl SlangCompiler {
    /// Find `slangc` (`ES_SLANGC`, else `PATH`) and record its version.
    pub fn new() -> Result<Self, GpuError> {
        let exe = PathBuf::from(std::env::var("ES_SLANGC").unwrap_or_else(|_| "slangc".to_owned()));
        let out = Command::new(&exe)
            .arg("-v")
            .output()
            .map_err(|e| GpuError::SlangcMissing(format!("{}: {e}", exe.display())))?;
        // slangc prints its version on stderr.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let version = if stderr.trim().is_empty() {
            stdout.trim().to_owned()
        } else {
            stderr.trim().to_owned()
        };
        Ok(Self {
            exe,
            version,
            cache_dir: default_cache_dir(),
            include_dirs: Vec::new(),
            invocations: Cell::new(0),
        })
    }

    /// Add an `-I` directory (e.g. `crates/es-math/slang` for `approx.slang`).
    #[must_use]
    pub fn with_include(mut self, dir: impl Into<PathBuf>) -> Self {
        self.include_dirs.push(dir.into());
        self
    }

    /// `slangc -v` output; part of every cache key.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// How many times `slangc` has actually been started. A cache hit does not move this.
    pub fn invocations(&self) -> usize {
        self.invocations.get()
    }

    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Compile `source`, applying the SPIR-V execution modes of `modes` (spec §3.4 step 3).
    pub fn compile(
        &self,
        source: &str,
        entry: &str,
        profile: &str,
        defines: &BTreeMap<String, String>,
        modes: ExecModes,
    ) -> Result<SpirvModule, GpuError> {
        let hash = self.cache_key(source, entry, profile, defines, modes);
        let cached = self.cache_dir.join(format!("{hash}.spv"));
        if let Ok(bytes) = std::fs::read(&cached) {
            return Ok(SpirvModule {
                words: words_from_bytes(&bytes)?,
                hash,
                entry: entry.to_owned(),
            });
        }

        std::fs::create_dir_all(&self.cache_dir)?;
        // Scratch names carry the process id and an invocation counter: two processes (or
        // two test threads) compiling the same key must not delete each other's input.
        let tag = format!("{}-{}", std::process::id(), self.invocations.get());
        let src_path = self.cache_dir.join(format!("{hash}.{tag}.slang"));
        let out_path = self.cache_dir.join(format!("{hash}.{tag}.raw.spv"));
        std::fs::write(&src_path, source)?;

        let mut cmd = Command::new(&self.exe);
        cmd.arg(&src_path)
            .args(["-target", "spirv"])
            .args(["-profile", profile])
            .args(["-entry", entry])
            .arg("-emit-spirv-directly")
            // Determinism (spec §3.4): no reassociation, no fast-math, no optimizer.
            .arg("-O0")
            .args(["-fp-mode", "precise"])
            .args(["-o"])
            .arg(&out_path);
        if modes.denorm_flush_to_zero_f32 {
            cmd.args(["-denorm-mode-fp32", "ftz"]);
        }
        for dir in &self.include_dirs {
            cmd.arg("-I").arg(dir);
        }
        for (k, v) in defines {
            cmd.arg(format!("-D{k}={v}"));
        }

        self.invocations.set(self.invocations.get() + 1);
        let out = cmd
            .output()
            .map_err(|e| GpuError::SlangcMissing(format!("{}: {e}", self.exe.display())))?;
        if !out.status.success() {
            let _ = std::fs::remove_file(&src_path);
            return Err(GpuError::SlangcFailed(
                String::from_utf8_lossy(&out.stderr).trim().to_owned(),
            ));
        }

        let raw = words_from_bytes(&std::fs::read(&out_path)?)?;
        // slangc emits DenormFlushToZero from `-denorm-mode-fp32 ftz`; RoundingModeRTE,
        // SignedZeroInfNanPreserve and NoContraction it does not emit, so they are patched
        // in. See docs/api-notes/slang.md.
        let words = spirv::apply_exec_modes(&raw, modes)
            .ok_or_else(|| GpuError::Spirv("slangc output has no entry point".to_owned()))?;
        // Publish the cache entry by rename, so a concurrent reader never sees half a file.
        let partial = self.cache_dir.join(format!("{hash}.{tag}.partial"));
        std::fs::write(&partial, bytes_from_words(&words))?;
        std::fs::rename(&partial, &cached)?;
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&out_path);

        Ok(SpirvModule {
            words,
            hash,
            entry: entry.to_owned(),
        })
    }

    /// Compile a file from disk.
    pub fn compile_file(
        &self,
        path: impl AsRef<Path>,
        entry: &str,
        profile: &str,
        defines: &BTreeMap<String, String>,
        modes: ExecModes,
    ) -> Result<SpirvModule, GpuError> {
        let source = std::fs::read_to_string(path.as_ref())?;
        self.compile(&source, entry, profile, defines, modes)
    }

    fn cache_key(
        &self,
        source: &str,
        entry: &str,
        profile: &str,
        defines: &BTreeMap<String, String>,
        modes: ExecModes,
    ) -> String {
        let mut h = blake3::Hasher::new();
        h.update(source.as_bytes());
        h.update(b"\0entry\0");
        h.update(entry.as_bytes());
        h.update(b"\0profile\0");
        h.update(profile.as_bytes());
        h.update(b"\0defines\0");
        for (k, v) in defines {
            h.update(k.as_bytes());
            h.update(b"=");
            h.update(v.as_bytes());
            h.update(b";");
        }
        h.update(b"\0modes\0");
        h.update(&modes.key_bytes());
        h.update(b"\0slangc\0");
        h.update(self.version.as_bytes());
        h.update(b"\0include\0");
        for dir in &self.include_dirs {
            h.update(dir.to_string_lossy().as_bytes());
            h.update(b";");
            // Include files are inputs too: hash their contents, not just the path.
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut files: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
                files.sort();
                for f in files {
                    if let Ok(bytes) = std::fs::read(&f) {
                        h.update(
                            f.file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .as_bytes(),
                        );
                        h.update(&bytes);
                    }
                }
            }
        }
        h.finalize().to_hex().to_string()
    }
}

/// `target/es-slang-cache`, overridable with `ES_SLANG_CACHE` (spec §2.3).
fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ES_SLANG_CACHE") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        return PathBuf::from(dir).join("es-slang-cache");
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/es-slang-cache")
}

fn words_from_bytes(bytes: &[u8]) -> Result<Vec<u32>, GpuError> {
    if bytes.len() % 4 != 0 || bytes.is_empty() {
        return Err(GpuError::Spirv(format!(
            "not a SPIR-V module: {} bytes",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn bytes_from_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compiler(version: &str) -> SlangCompiler {
        SlangCompiler {
            exe: PathBuf::from("slangc"),
            version: version.to_owned(),
            cache_dir: PathBuf::from("/nonexistent"),
            include_dirs: Vec::new(),
            invocations: Cell::new(0),
        }
    }

    #[test]
    fn cache_key_separates_every_input() {
        let c = compiler("v1");
        let d = BTreeMap::new();
        let base = c.cache_key("src", "main", "glsl_450", &d, ExecModes::none());

        assert_ne!(
            base,
            c.cache_key("src2", "main", "glsl_450", &d, ExecModes::none())
        );
        assert_ne!(
            base,
            c.cache_key("src", "other", "glsl_450", &d, ExecModes::none())
        );
        assert_ne!(
            base,
            c.cache_key("src", "main", "sm_6_0", &d, ExecModes::none())
        );
        assert_ne!(
            base,
            c.cache_key("src", "main", "glsl_450", &d, ExecModes::deterministic())
        );
        let defines = BTreeMap::from([("N".to_owned(), "4".to_owned())]);
        assert_ne!(
            base,
            c.cache_key("src", "main", "glsl_450", &defines, ExecModes::none())
        );
        // Spec 3.4 item 7: the compiler version is part of the identity.
        assert_ne!(
            base,
            compiler("v2").cache_key("src", "main", "glsl_450", &d, ExecModes::none())
        );
    }

    #[test]
    fn cache_key_is_stable_and_define_order_independent() {
        let c = compiler("v1");
        let a = BTreeMap::from([
            ("A".to_owned(), "1".to_owned()),
            ("B".to_owned(), "2".to_owned()),
        ]);
        let mut b = BTreeMap::new();
        b.insert("B".to_owned(), "2".to_owned());
        b.insert("A".to_owned(), "1".to_owned());
        assert_eq!(
            c.cache_key("s", "main", "p", &a, ExecModes::none()),
            c.cache_key("s", "main", "p", &b, ExecModes::none())
        );
    }

    #[test]
    fn words_round_trip() {
        let w = vec![0x0723_0203, 1, 2, 3];
        assert_eq!(words_from_bytes(&bytes_from_words(&w)).unwrap(), w);
        assert!(words_from_bytes(&[1, 2, 3]).is_err());
        assert!(words_from_bytes(&[]).is_err());
    }
}
