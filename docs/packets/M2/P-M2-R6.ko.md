<!-- Korean translation of docs/packets/M2/P-M2-R6.md. The English file is the working copy; regenerate this when it changes. -->

# P-M2-R6 — `diffusers`에 대조한 DDPM 스케줄 (`es-policy`)

Spec: spec 8.3 (`HeadKind::Diffusion` 파라미터), spec 8.7 (lowering), spec 8.9 (tier 4),
spec 1.4 (오라클이 구현보다 먼저 오며, *독립적*이어야 한다), spec 3.4 (결정성, 전역 RNG
없음), spec 5.3 (해시 체인), spec 14.4 (LeRobot config).
설계 노트: `docs/design/learning-lowering.md` 8절 (리뷰 등급 C).
API 노트: `docs/api-notes/torch.md`, `docs/api-notes/lerobot-config.md`.

M2 리뷰 후속 조치. 발견 사항 둘, 근본 원인 하나: `HeadKind::Diffusion`은 `n_steps`와
스케줄러 종류만 싣고 있었으므로, W3는 스케줄의 나머지를 lowering 상수로 지어내야 했다 —
*추론* 스텝들에 걸쳐 펼쳐진 선형 `1e-4 .. 0.02` beta 스케줄과 `sigma_t = sqrt(beta_t)`
(diffusers의 `fixed_large`). `diffusers`와 LeRobot은 beta들을 `num_train_timesteps`에
걸쳐 만들고 서브샘플링하며, 기본값은 `fixed_small`이다. 실제 Diffusion Policy
체크포인트는 이 아래에서 재현되지 않았을 것이다. 두 번째 발견 사항은 아무도 알아채지
못한 이유다: `reference.rs:26`은 자신이 검사해야 할 lowering으로부터
`diffusion_schedule`을 import하므로, tier-4 테스트는 Rust<->PyTorch 합치만 증명했을
뿐 계수에 대해서는 아무것도 증명하지 못했다.

## context (범위)

```
crates/es-ir/src/learning.rs                  (extended: HeadKind::Diffusion fields, additive, serde defaults)
crates/es-policy/src/lower/torch.rs           (rewritten: diffusion_schedule, _DdpmHead)
crates/es-policy/src/reference.rs             (extended: the diffusers oracle and its two tests)
crates/es-policy/src/torch_runtime.rs         (python_candidates is pub(crate))
crates/es-policy/python/ddpm_ref_check.py     (new: the independent oracle)
crates/es-data/src/lerobot_config.rs          (extended: the new fields pass through)
crates/es-data/tests/lerobot_config.rs        (extended: one pass-through test)
docs/design/learning-lowering.md              (rewritten: sections 8.3, 8.5)
docs/api-notes/torch.md                       (extended: diffusers surface, known gaps)
docs/api-notes/lerobot-config.md              (extended: the six scheduler fields)
docs/packets/M2/P-M2-R6.md                    (new)
```

## spec (사양)

- `HeadKind::Diffusion`은 `num_train_timesteps`, `beta_schedule`, `variance_type`,
  `prediction_type`, `clip_sample`, `clip_sample_range`를 얻으며, 각각은 LeRobot
  Diffusion Policy 기본값과 같은 `serde` 기본값을 가진다(`100`, `squaredcos_cap_v2`,
  `fixed_small`, `epsilon`, `true`, `1.0` — `미검증 (unverified)`, 가져온 것이 아니라
  기억을 되살린 것). 추가적(additive)이다: 이 패킷 이전에 작성된 IR 문서는 여전히
  역직렬화된다. `HeadKind`는 `Eq`를 잃는다(이제 `f32`를 싣기 때문이다); 이는 오직
  `PartialEq`만인 `LearningNode`를 통해서만 비교된 적이 있었다.
- `diffusion_schedule`은 `betas` / `alphas_cumprod`를 `num_train_timesteps`에 걸쳐
  만들고, diffusers의 기본 `timestep_spacing = "leading"` 아래에서 `set_timesteps`가
  하는 것과 같은 방식으로 `n_steps`개의 추론 timestep을 서브샘플링한다:
  `(arange(0, n_steps) * (num_train // n_steps))[::-1]`. `beta_schedule`은
  `linear`(`torch.linspace`) 또는 `squaredcos_cap_v2`(`betas_for_alpha_bar`, cosine,
  `0.999`에서 상한)다.
- 분산은 `DDPMScheduler._get_variance`를 아래로 `1e-20`에서 clamp한 것이다: 기본값으로
  `fixed_small` `beta~_t = (1 - abar_prev)/(1 - abar_t) * beta_cur`, 선택 가능한
  `fixed_large` `beta_cur`. `DDIMScheduler`에는 `variance_type`이 없으므로 `Ddim`
  아래에서는 무시된다.
- `clip_sample`은 `pred_original_sample`을 `+-clip_sample_range`로 clamp하며, 이는
  아핀 `c1*x + c3*eps` 대신 diffusers의 분해(먼저 `x0`, 그다음 재결합)로 루프를
  강제한다. DDIM eta = 0 경로는 형태상 변하지 않고 이제 서브샘플링된 timestep들 위에서
  실행된다. `noise_<t>` 버퍼는 실제 timestep으로 키가 매겨진다.
- `prediction_type != Epsilon`은 `LowerError::Unsupported`다; `num_train_timesteps <
  n_steps`와 홀수인 conditioning 너비는 `LowerError::Shape`다.
- `es-data`의 LeRobot 변환은 여섯 개 모두를 그대로 통과시킨다; `variance_type`은
  LeRobot 필드가 아니며 diffusers의 `fixed_small`로 고정된다.

## oracle (오라클)

```
cargo test -p es-policy -p es-ir -p es-data --features es-ir/testing
ES_PYTHON=<venv-with-torch-and-diffusers>/Scripts/python cargo test -p es-policy -- --nocapture
```

- `python/ddpm_ref_check.py`는 독립적인 레퍼런스다: 스케줄러 설정이 주어지면 실제
  `DDPMScheduler` / `DDIMScheduler`를 만들고 `alphas_cumprod`, 서브샘플링된
  `timesteps`, 스텝별 `_get_variance`를 JSON으로 출력한다; 가중치와 입력도 주어지면,
  전체 샘플러를 `scheduler.step(model_output, t, sample)`을 통해 실행하고 그 청크를
  출력한다.
- `reference::tests::diffusion_schedule_matches_diffusers`는 `diffusion_schedule`을
  이것과 대조하는데, 여덟 개의 스케줄러 x `beta_schedule` x `variance_type` 조합
  전체에 대해 `Tolerance::TIER4_FP32`에서 비교하며, `timesteps`는 **정확히** 같아야
  한다.
- `reference::tests::torch_{ddpm,ddim}_matches_diffusers_step_loop`는 lowering된
  head를 `TorchRuntime`을 통해 실행하고 이를 `Tolerance::TIER4_FP32`에서 그 직접
  step 루프와 비교한다. DDPM의 ancestral 추출은 `randn_tensor` monkeypatch를 통해
  체크포인트의 `noise_<t>` 버퍼에서 오므로, 양쪽 다 같은 숫자를 소비한다.
- 둘 다 `RAN ... max_abs = ...` 또는 `SKIPPED ...: <why>`를 한 줄로 출력한다;
  인터프리터가 없거나, `torch` 또는 `diffusers`가 없으면 항상 skip이지 결코 pass가
  아니다.

## acceptance (수용 기준)

- 다섯 개의 torch-tier 테스트가 여기서 RAN했다(torch 2.14.0+cpu, diffusers 0.40.0):
  `torch_ddpm` 2.76e-7, `torch_ddim` 1.79e-7, `torch_flow_matching` 1.04e-7,
  `ddpm_matches_diffusers_step_loop` 6.86e-7, `ddim_matches_diffusers_step_loop`
  1.34e-6 `max_abs`, 모두 spec 8.9 tier-4 한계인 1e-5 아래다.
  `diffusion_schedule_matches_diffusers`: `alphas_cumprod` <= 2.39e-7,
  `_get_variance` <= 4.18e-7, `timesteps` 정확히 일치, 여덟 조합 전체에 대해.
- tier-4 설정은 100개의 학습 timestep 중 8개의 추론 스텝이므로, 서브샘플링 경로를
  가로지른다: `[84, 72, 60, 48, 36, 24, 12, 0]`, 생성된 소스에 문자 그대로
  단언된다.
- `cargo fmt --check`, `clippy -D warnings`와 `es-policy` / `es-ir` / `es-data`의
  전체 테스트 실행이 깨끗하다; `es-ir`는 spec 1.5 context budget 안에 머문다.

## forbidden (금지)

- 새 trait 없음(`INV-17`: 일곱 확장 지점은 닫혀 있다). `PolicyRuntime` 변경 없음.
- `sample` / `v_prediction`의, `DpmSolver`의, `eta > 0`의, `"leading"` 외의
  `timestep_spacing`의, U-Net denoiser의 새로운 lowering 없음 — 이들은 이후 패킷이며,
  각각은 근사가 아니라 여기서 명시적인 `Unsupported`다.
- `## context` 목록 밖의 편집 없음. `crates/es-data/src/lerobot_config.rs`는 진행
  중인 `Augment` 연결 수정(P-M2-R2)과 공유된다: diffusion-head 구성과
  `DiffusionConfig` 필드만 이 패킷의 것이다.
- `pickle` 없음, `torch.load` 없음(`INV-16`): 오라클의 가중치는 JSON 숫자로
  넘어간다.
- Golden 파일은 읽기 전용이다(spec 1.4).
