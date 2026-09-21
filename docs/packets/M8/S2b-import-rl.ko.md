# M8 S2b — `es policy import-rl`: PPO 액터가 어댑터 문서를 거쳐 번들이 된다

스펙: §14.4("RL 정책 가져오기" 문단: Python 쪽 절반 → `safetensors` + `import.json`, `es`
쪽 절반 → 문서들 + 번들; Semantic Mapping Report), §13.4("가져오기는 결코 추측하지 않는다"),
§8.9(동등성 계층), §28.11 파동 2, INV-16. **S2a**(활성 함수/스쿼시)와 **S2c**(실제 체크포인트와
`oracle-1000.npz`)에 의존. 선례: `es import lerobot-config`(`crates/es/src/cmd/import.rs`,
`es_data::lerobot_config::convert`)와 `es policy import-lerobot`
(`crates/es/src/cmd/policy.rs`, `es_policy::lerobot`). 오라클 계층:
`docs/design/rl-continuation.md` 4절. 확장할 설계 노트: `rl-continuation.md` 7절(측정됨)과
새 절 "임포터와 어댑터"(+ `.ko.md`).

## 질문

**brax, rsl_rl, rl_games가 학습한 액터를 Learning IR + Observation IR + 번들로 바꿀 수 있는가,
그 런타임 출력이 무작위 관측 1,000개 위에서 소스의 결정론적 행동을 재현하도록 — 모든 관절,
단위, 채널 매핑이 문서에 선언되고 모든 불일치가 이름으로 거절되도록?**

## 사양

* **Python 쪽 절반** `python/es/import_rl.py --from mujoco-playground|rsl-rl|rl-games
  --checkpoint <path> [--activation elu|swish|relu|tanh] --out <dir>` →
  `weights.safetensors`(중립 키 `mlp.<i>.weight|bias` `[out, in]`, `head.weight|bias`,
  `log_std`)와 `import.json`(`framework`, `versions`, `obs_dim`, `action_dim`, `hidden`,
  `activation`, `activate_output`, `squash`, `obs_mean`, `obs_std`, `log_std`,
  `action_scale`, `action_offset` — 프레임워크가 그것을 나르지 않으면 `null`이고, 그러면
  어댑터가 그것을 대신 공급해야 한다).
  - `mujoco-playground`: S2c의 `source.npz` / `meta.json`, 또는
    `brax.training.checkpoint.load`를 통한 orbax 체크포인트; 출력 Dense의 mean 절반;
    `squash = tanh`, `activate_output = true`, `activation = swish`.
  - `rsl-rl`: `torch.load(model_*.pt)`(옛 `model_state_dict`에 최상위 `std`가 있는 레이아웃과
    ≥ 5.0의 `actor_state_dict` 레이아웃 둘 다); `actor.<i>.weight`; 선택적 `obs_normalizer`
    (`EmpiricalNormalization`의 mean / var → std); 활성 함수는 `--activation`에서 온다
    (파일이 그것을 나르지 않는다 — 플래그 없이는 거절하라).
  - `rl-games`: `.pth`의 `model` 딕셔너리, `a2c_network.actor_mlp.<i>`, `a2c_network.mu`,
    `running_mean_std.running_mean|running_var`; 활성 함수는 `--activation`에서 온다.
  - pickle / orbax는 **오직 여기서만** 열린다(INV-16).
* **Rust 쪽 절반** `es policy import-rl --manifest <import.json> --weights <w.safetensors>
  --adapter <adapter.toml> --task <task.toml> --deployment <d.toml> --out <dir>`는
  `observation.toml`(어댑터 채널마다 `StateInput` → `Concat` → manifest로부터의
  `Normalize{MeanStd}`, manifest에 없으면 identity `Range`), `learning.toml`
  (`StateEncoder{Mlp hidden, activation, activate_output}` → `PolicyHead{Regression,
  horizon 1, squash}` → `Normalizer{Inverse, MeanStd: mean = offset, std = scale}`),
  `policy.esb`(pack 경로, 가중치는 로워링된 모듈의 `weight_keys`로 다시 매핑됨),
  `mapping-report.json` — §14.4의 Semantic Mapping Report: 관절마다, 관측 채널마다 한
  행(소스 인덱스, 우리 이름, 단위, 심각도)을 쓴다.
* **`adapter.toml`**(`deny_unknown_fields`): `[robot] name`; `[joints] source_order = [...]`
  (프레임워크의 것), `units = "rad" | "deg"`; `[action] kind = "position_target" | "torque"`,
  선택적 `scale` / `offset`(manifest를 덮어씀); `[observation] channels = [{ source = "qpos",
  slice = [0, 6], channel = "joint_pos" }, …]`이 각 슬라이스가 먹이는 Task IR
  `ObservationSpec` 채널을 이름 짓는다. 커밋된 것: `tests/fixtures/rl/adapter-so101.toml`.
* **거절, 이름을 대고**(`es-data`나 `convert`가 사는 곳의 `IMP-001` …): Task IR
  `ActionSpec.dim`과 다른 관절 수; 장면의 액추에이터에 없는 관절 이름; `units = deg`;
  `ActionSpec.space`와 맞지 않는 `kind`; `obs_dim`을 정확히 채우지 못하거나 선언되지 않은
  채널을 이름 짓는 관측 슬라이스.
* CI를 위한 합성 픽스처(테스트 시점에 Python 없이):
  `tests/fixtures/rl/import/{playground, rsl-rl,rl-games}/{import.json,weights.safetensors}` —
  15 → [8, 8] → 6, `import_rl.py --synth <framework>`로 한 번 생성되어 커밋됨(각각 몇 KB).

## context

```
python/es/import_rl.py
crates/es-data/src/rl_import.rs
crates/es-data/src/lib.rs
crates/es-ir-types/src/codes.rs
crates/es-data/tests/**
crates/es/src/cmd/policy.rs
crates/es/tests/cli.rs
tests/fixtures/rl/**
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M8/S2b-import-rl.md
docs/packets/M8/S2b-import-rl.ko.md
```

## 오라클

1. `cargo test -p es --test cli import_rl_synthetic_three_frameworks` — 각 픽스처 → 두
   문서가 검증되고, 번들이 `TorchRuntime`에서 열린다(`ES_PYTHON` 없이는 여는 것을 SKIP),
   `mapping-report.json`은 관절과 채널마다 한 행을 갖는다.
2. `cargo test -p es-data import_rl_refusals` — 이름으로 대는 다섯 개의 `IMP-0xx`.
3. 서버(`--ignored`, `ES_PYTHON`): S2c의 체크포인트 → 번들; `oracle-1000.npz` 위에서:
   (a) 임포터의 numpy→torch 재구성 대 우리 런타임 **비트 단위** f32; (b) JAX의 행동 대 우리
   런타임 최대 절대오차 ≤ 1e-5, 값이 기록됨. rsl_rl / rl_games의 경우: venv 안의 소스
   프레임워크에서 몇 초 학습된 작은 액터(또는 프레임워크 자신의 `save`로 저장된 무작위
   가중치 액터) → 우리 런타임 **비트 단위**.
4. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S2b-import-rl.md`.

## 수용 기준

오라클 1–4; 설계 노트의 새 절과 측정된 행들(가져온 번들의 해시, 최대 절대오차, 날짜, 경로).

## 금지

어댑터가 선언하지 않은 매핑을 추측하는 것; Rust 안의 pickle(INV-16); 새 노드 타입;
`es-policy/src/lerobot.rs`를 건드리는 것; `docs/ARCHITECTURE*.md`; `tests/golden/**`;
INV-17.
