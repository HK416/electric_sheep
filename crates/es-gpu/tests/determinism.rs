//! GPU tests. Every one of them starts by opening a device and prints `SKIP <reason>` when
//! there is none — CI has no GPU (spec §1.4: the oracle must be runnable everywhere, even
//! when it can only report that it did not run).
//!
//! Oracles here:
//!
//! * the §3.4 step-3 execution modes are present **in the SPIR-V**, checked by parsing it;
//! * the Slang cache serves a second compile without starting `slangc` (§2.3);
//! * a fixed-order tree reduction over 1M f32 is bit-identical across two runs and equal to
//!   the CPU mirror bit for bit (§3.4, §3.5 tier 1);
//! * §28.7 gate 3: `es_math::approx` on the CPU vs `approx.slang` on the GPU, bit for bit;
//! * `spirv-val` accepts the patched modules, so the hand-written SPIR-V is checked rather
//!   than trusted (review `docs/reviews/M4.md` S-6);
//! * `download` hands back exactly the requested byte count, not the allocator's padding
//!   (S-12).

use std::collections::BTreeMap;
use std::path::PathBuf;

use es_gpu::{
    spirv_float_arithmetic_count, spirv_has_execution_mode, spirv_no_contraction_count,
    BindingDesc, BindingKind, Buffer, CommandRecorder, ComputePipeline, ExecModes, Gpu, GpuOptions,
    SlangCompiler, SpirvModule, Usage, EXEC_MODE_DENORM_FLUSH_TO_ZERO, EXEC_MODE_ROUNDING_MODE_RTE,
    EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
};
use es_math::{BinnedAcc, DeterministicAcc};

fn kernel(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("kernels")
        .join(name)
}

fn math_slang_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../es-math/slang")
}

/// `Some(gpu)` or a printed SKIP.
fn open_gpu(test: &str) -> Option<Gpu> {
    match Gpu::open(GpuOptions::default()) {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            println!("SKIP {test}: no Vulkan device ({e})");
            None
        }
    }
}

/// `Some(compiler)` or a printed SKIP.
fn open_slang(test: &str) -> Option<SlangCompiler> {
    match SlangCompiler::new() {
        Ok(c) => Some(c),
        Err(e) => {
            println!("SKIP {test}: no slangc ({e})");
            None
        }
    }
}

fn compile(
    compiler: &SlangCompiler,
    file: &str,
    defines: &BTreeMap<String, String>,
    modes: ExecModes,
) -> SpirvModule {
    compiler
        .compile_file(kernel(file), "main", "glsl_450", defines, modes)
        .unwrap_or_else(|e| panic!("compiling {file}: {e}"))
}

/// Deterministic inputs: no global RNG (spec §3.4), just a fixed-seed xorshift.
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Uniform in [-1, 1) from the top 24 bits.
            let unit = ((state >> 40) as f32) / f32::from(1u16 << 12) / 4096.0;
            unit.mul_add(2.0, -1.0)
        })
        .collect()
}

/// Inputs per transcendental in the gate-3 probe.
const N: usize = 4096;

/// Distance in ULPs between two f32, for an honest failure report.
// Exact comparison and the sign-magnitude to two's-complement flip are the whole point here.
#[allow(clippy::float_cmp, clippy::cast_possible_wrap)]
fn ulp_distance(a: f32, b: f32) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let ordered = |x: f32| -> i64 {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

/// The CPU mirror of `kernels/sum.slang`: the same pairwise tree, in f32, stopping at the
/// same level.
fn cpu_tree(data: &[f32]) -> Vec<f32> {
    let mut level = data.to_vec();
    while level.len() >= 128 {
        level = level.chunks_exact(2).map(|c| c[0] + c[1]).collect();
    }
    level
}

/// Finish a 64-element level the same way, down to one value.
fn cpu_finish(mut level: Vec<f32>) -> f32 {
    while level.len() > 1 {
        level = level.chunks_exact(2).map(|c| c[0] + c[1]).collect();
    }
    level[0]
}

/// Run the tree on the GPU, returning the 64-element level it stops at.
fn gpu_tree(gpu: &Gpu, pipeline: &ComputePipeline<'_>, data: &[f32]) -> Vec<f32> {
    let bytes = std::mem::size_of_val(data) as u64;
    let mut a = Buffer::from_f32(gpu, data, Usage::Storage).expect("input buffer");
    let mut b = Buffer::new(gpu, bytes, Usage::Storage).expect("scratch buffer");

    let mut recorder = CommandRecorder::new(gpu).expect("recorder");
    let mut n = data.len();
    let mut src_is_a = true;
    while n >= 128 {
        let groups = u32::try_from(n / 2 / 64).expect("group count");
        let pair: [&Buffer<'_>; 2] = if src_is_a { [&b, &a] } else { [&a, &b] };
        recorder
            .dispatch(pipeline, &pair, [groups, 1, 1])
            .expect("dispatch");
        src_is_a = !src_is_a;
        n /= 2;
    }
    recorder.submit_and_wait().expect("submit");
    pipeline.reset_descriptors().expect("reset descriptors");

    let out = if src_is_a {
        a.download_f32().expect("download")
    } else {
        b.download_f32().expect("download")
    };
    out[..64].to_vec()
}

#[test]
fn device_capabilities_are_queried_and_reported() {
    let Some(gpu) = open_gpu("device_capabilities_are_queried_and_reported") else {
        return;
    };
    let caps = gpu.capabilities();
    println!("RAN device: {}", caps.device_name);
    println!(
        "RAN driver: {} {} / api {}.{}.{}",
        caps.driver_name,
        caps.driver_info,
        caps.api_version.0,
        caps.api_version.1,
        caps.api_version.2
    );
    println!(
        "RAN float-controls f32: ftz={} rte={} szinp={} denorm_indep={:?} rounding_indep={:?}",
        caps.float_controls.shader_denorm_flush_to_zero_float32,
        caps.float_controls.shader_rounding_mode_rte_float32,
        caps.float_controls
            .shader_signed_zero_inf_nan_preserve_float32,
        caps.float_controls.denorm_behavior_independence,
        caps.float_controls.rounding_mode_independence,
    );
    println!(
        "RAN subgroup: size={} ops={:?} f64={} i64={} coop_matrix={}",
        caps.subgroup.size,
        caps.subgroup.supported_ops,
        caps.shader_float64,
        caps.shader_int64,
        caps.cooperative_matrix
    );
    println!("RAN determinism tier: {:?}", gpu.determinism_tier());
    assert!(!caps.device_name.is_empty());
    assert!(caps.memory_heaps.iter().any(|h| h.size_bytes > 0));
}

#[test]
fn execution_modes_are_present_in_the_spirv() {
    let test = "execution_modes_are_present_in_the_spirv";
    let Some(compiler) = open_slang(test) else {
        return;
    };
    let module = compile(
        &compiler,
        "sum.slang",
        &BTreeMap::new(),
        ExecModes::deterministic(),
    );
    for (name, mode) in [
        ("DenormFlushToZero", EXEC_MODE_DENORM_FLUSH_TO_ZERO),
        ("RoundingModeRTE", EXEC_MODE_ROUNDING_MODE_RTE),
        (
            "SignedZeroInfNanPreserve",
            EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
        ),
    ] {
        assert!(
            spirv_has_execution_mode(&module.words, mode),
            "missing execution mode {name}"
        );
    }
    let arithmetic = spirv_float_arithmetic_count(&module.words);
    let decorated = spirv_no_contraction_count(&module.words);
    assert!(arithmetic > 0, "kernel has no float arithmetic to decorate");
    assert_eq!(
        arithmetic, decorated,
        "every float arithmetic result must carry NoContraction"
    );
    println!(
        "RAN exec modes in SPIR-V: 3/3, NoContraction {decorated}/{arithmetic}, hash {}",
        &module.hash[..16]
    );
}

/// Review `docs/reviews/M4.md` S-6, packet `P-M4-R7`: `spirv.rs` hand-writes SPIR-V, so the
/// patched module is checked by `spirv-val` rather than by a header parse. `ES_REQUIRE_SPIRV_VAL=1`
/// turns "the tool is not installed" into a failure; without it that is a printed SKIP.
#[test]
fn patched_kernels_pass_spirv_val() {
    let test = "patched_kernels_pass_spirv_val";
    let Some(compiler) = open_slang(test) else {
        return;
    };
    let compiler = compiler.with_include(math_slang_dir());
    for file in ["sum.slang", "approx_probe.slang"] {
        let defines = BTreeMap::from([("ES_N".to_owned(), N.to_string())]);
        let module = compile(&compiler, file, &defines, ExecModes::deterministic());
        let ran = es_gpu::validate_spirv(&module.words)
            .unwrap_or_else(|e| panic!("spirv-val rejected the patched {file}: {e}"));
        if ran {
            println!("RAN spirv-val: patched {file} accepted");
        } else {
            println!(
                "SKIP {test}: no spirv-val on PATH (set ES_REQUIRE_SPIRV_VAL=1 to require it)"
            );
            return;
        }
    }
}

/// Review `docs/reviews/M4.md` S-12: `download` used to hand back the whole allocation,
/// which `gpu-allocator` pads, so `download_f32().len()` could exceed the requested count.
#[test]
fn download_returns_exactly_the_requested_length() {
    let test = "download_returns_exactly_the_requested_length";
    let Some(gpu) = open_gpu(test) else {
        return;
    };
    // Three floats: below any plausible allocation alignment, so padding would show.
    let data = [1.0f32, 2.0, 3.0];
    for usage in [Usage::Storage, Usage::Staging] {
        let mut buffer = Buffer::from_f32(&gpu, &data, usage).expect("buffer");
        let got = buffer.download_f32().expect("download");
        assert_eq!(got.len(), data.len(), "{usage:?} download length");
        assert_eq!(got, data, "{usage:?} download contents");
        assert_eq!(buffer.download().expect("download").len(), data.len() * 4);
    }
    println!(
        "RAN buffer download: {} f32 in, {} f32 out",
        data.len(),
        data.len()
    );
}

#[test]
fn cache_hit_does_not_start_slangc() {
    let test = "cache_hit_does_not_start_slangc";
    let Some(compiler) = open_slang(test) else {
        return;
    };
    // A define no other run has used, so the first compile is always a cold miss and the
    // second is always the warm path being tested.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let defines = BTreeMap::from([("ES_CACHE_PROBE".to_owned(), nonce.to_string())]);
    let first = compile(&compiler, "sum.slang", &defines, ExecModes::deterministic());
    let after_first = compiler.invocations();
    assert_eq!(after_first, 1, "first compile must be a cache miss");
    let second = compile(&compiler, "sum.slang", &defines, ExecModes::deterministic());
    assert_eq!(
        compiler.invocations(),
        after_first,
        "second compile must be served from the cache"
    );
    assert_eq!(first.hash, second.hash);
    assert_eq!(first.words, second.words);
    println!(
        "RAN cache: {} invocation(s) for two compiles, key {}",
        compiler.invocations(),
        &first.hash[..16]
    );
}

#[test]
fn tree_reduction_is_bit_identical_across_runs_and_matches_the_cpu() {
    let test = "tree_reduction_is_bit_identical_across_runs_and_matches_the_cpu";
    let (Some(gpu), Some(compiler)) = (open_gpu(test), open_slang(test)) else {
        return;
    };
    // Ask for exactly the modes this device advertises (spec 3.4 step 2).
    let modes = gpu.deterministic_execution_modes();
    println!(
        "RAN requested exec modes: {modes:?}, device tier {:?}",
        gpu.determinism_tier()
    );
    let module = compile(&compiler, "sum.slang", &BTreeMap::new(), modes);
    let bindings = [
        BindingDesc {
            binding: 0,
            kind: BindingKind::Storage,
        },
        BindingDesc {
            binding: 1,
            kind: BindingKind::Storage,
        },
    ];
    let pipeline = ComputePipeline::new(&gpu, &module, &bindings).expect("pipeline");

    let data = pseudo_random(1 << 20, 0x5eed_1234);
    let first = gpu_tree(&gpu, &pipeline, &data);
    let second = gpu_tree(&gpu, &pipeline, &data);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&first), bits(&second), "two runs disagree");

    let cpu = cpu_tree(&data);
    assert_eq!(cpu.len(), first.len());
    let worst = first
        .iter()
        .zip(&cpu)
        .map(|(g, c)| ulp_distance(*g, *c))
        .max()
        .unwrap_or(0);
    println!("RAN tree reduction: 1M f32, two runs bit-identical, CPU max ULP {worst}");
    assert_eq!(bits(&first), bits(&cpu), "GPU tree differs from CPU tree");

    // BinnedAcc (spec §18.4) is the accuracy reference, not a bit reference: an exact binned
    // sum and an f32 tree are different numbers by construction.
    let gpu_total = cpu_finish(first);
    let mut acc = BinnedAcc::<3>::default();
    for x in &data {
        acc.add(f64::from(*x));
    }
    let exact = acc.finish();
    let relative = (f64::from(gpu_total) - exact).abs() / exact.abs().max(1e-30);
    println!("RAN tree total {gpu_total:e} vs BinnedAcc {exact:e}, relative {relative:e}");
    assert!(relative < 1e-4, "f32 tree drifted from the exact sum");
}

#[test]
fn gate3_cpu_and_slang_transcendentals_agree_bit_for_bit() {
    let test = "gate3_cpu_and_slang_transcendentals_agree_bit_for_bit";
    let (Some(gpu), Some(compiler)) = (open_gpu(test), open_slang(test)) else {
        return;
    };
    let compiler = compiler.with_include(math_slang_dir());
    let defines = BTreeMap::from([("ES_N".to_owned(), N.to_string())]);
    let module = compile(
        &compiler,
        "approx_probe.slang",
        &defines,
        gpu.deterministic_execution_modes(),
    );
    let bindings = [
        BindingDesc {
            binding: 0,
            kind: BindingKind::Storage,
        },
        BindingDesc {
            binding: 1,
            kind: BindingKind::Storage,
        },
    ];
    let pipeline = ComputePipeline::new(&gpu, &module, &bindings).expect("pipeline");

    // Inputs spread over the range the kernels are used in: sin needs range reduction, exp
    // needs the scaling path.
    let inputs: Vec<f32> = pseudo_random(N, 0xa5a5_0f0f)
        .iter()
        .enumerate()
        .map(|(i, u)| u * (1.0 + (i % 16) as f32))
        .collect();

    let input = Buffer::from_f32(&gpu, &inputs, Usage::Storage).expect("input");
    let mut output = Buffer::new(&gpu, (N * 2 * 4) as u64, Usage::Storage).expect("output");
    let mut recorder = CommandRecorder::new(&gpu).expect("recorder");
    recorder
        .dispatch(&pipeline, &[&output, &input], [(N / 64) as u32, 1, 1])
        .expect("dispatch");
    recorder.submit_and_wait().expect("submit");
    let got = output.download_f32().expect("download");

    let mut worst_sin = 0u64;
    let mut worst_exp = 0u64;
    let mut mismatched = 0usize;
    for (i, x) in inputs.iter().enumerate() {
        let sin = ulp_distance(got[i], es_math::approx::sin(*x));
        let exp = ulp_distance(got[i + N], es_math::approx::exp(*x));
        worst_sin = worst_sin.max(sin);
        worst_exp = worst_exp.max(exp);
        mismatched += usize::from(sin != 0 || exp != 0);
    }
    println!(
        "RAN gate 3: {N} inputs, {mismatched} mismatched, max ULP sin {worst_sin} exp {worst_exp}"
    );
    assert_eq!(
        (worst_sin, worst_exp),
        (0, 0),
        "spec 28.7 gate 3 UNMET: CPU/Slang not bit-equal (max ULP sin {worst_sin}, exp \
         {worst_exp}); the modes are in the SPIR-V, so this is a driver or codegen difference"
    );
}
