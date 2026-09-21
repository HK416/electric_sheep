# M9 T2 — 증분-행동 소스 정책, 그리고 그것을 가져오는 어댑터

스펙: §14.4(어댑터가 `[action] kind`를 선언한다), §8.5(증분 단위), §28.12 파동 1, INV-16.
**T1**에 의존한다. 선례: S2c(`python/es/rl_source/`), S2b(`import_rl.py`,
`es policy import-rl`, `adapter-so101.toml`). 확장할 API 노트:
`docs/api-notes/brax-ppo-so101.md`(+ `.ko.md`), 증분 env를 위한 새 절.

## 질문

**같은 brax 스택이 행동이 틱당 관절 증분인 reach 정책을 학습할 수 있는가, 그리고 가져오기가
그것을 1,000개 관측에서 소스를 재현하는 `JointDelta` 번들로 들여올 수 있는가 — 코드가 아니라
어댑터가 행동이 증분이라고 말하면서?**

## 사양

* `python/es/rl_source/so101_reach_env.py`가 `--action delta`를 얻는다: `target_t =
  target_{t−1} + a · delta_scale`(`delta_scale`은 `meta.json`에 기록됨, 예: 0.05 rad),
  target₀ = 리셋 포즈; 관측, 보상, 성공, 타임아웃은 불변(`rl-continuation.md` 5절); 파생된
  MJX 장면도 불변. `train_brax_so101.py --action delta`는 같은 내보내기를 쓴다(`source.npz`,
  `action = { kind = "joint_delta", scale = … }`를 담은 `meta.json`, `oracle-1000.npz`).
* `import_rl.py`는 `action.kind`를 `import.json`을 통해 넘긴다; `adapter-so101-delta.toml`은
  `[action] kind = "joint_delta"`와 증분 단위를 선언한다; `es policy import-rl`은 kind가
  어댑터와 어긋나는 매니페스트를 거절하고(`IMP-004`가 이미 이 이름을 대고 있다)
  `action.space = JointDelta`와 증분 단위의 `Normalizer{Inverse}`를 가진 Deployment IR 참조를
  낸다.
* 소스 프레임워크에서 측정: 64 에피소드에 걸친 성공률, 그리고 틱당 명령 변화 분포(T3이
  클램프율을 견주는 숫자).

## context

```
python/es/rl_source/**
python/es/import_rl.py
crates/es-data/src/rl_import.rs
crates/es-data/tests/**
crates/es/tests/cli.rs
tests/fixtures/rl/adapter-so101-delta.toml
tests/fixtures/rl/import/playground-delta/**
docs/api-notes/brax-ppo-so101.md
docs/api-notes/brax-ppo-so101.ko.md
docs/design/rl-continuation.md
docs/design/rl-continuation.ko.md
docs/packets/M9/T2-delta-source-and-import.md
docs/packets/M9/T2-delta-source-and-import.ko.md
```

## 오라클

1. 서버: `train_brax_so101.py --action delta --seed 0`를 두 번 → `source.npz` 비트 단위(기록);
   64 에피소드에 걸친 성공률 기록(목표 0.8; 도달한 만큼 출하).
2. 서버(`--ignored`): 가져오기 → (a) 재구성 대 우리 런타임, 1,000행에서 비트 단위, (b) 우리
   런타임 대 JAX, `obs_scaled` 위에서 ≤ 1e-5; 번들의 Deployment IR이 `JointDelta`라고 말한다.
3. `cargo test -p es --test cli import_rl_delta_fixture` — 합성 `playground-delta` 픽스처 →
   문서 + 번들; kind 불일치는 이름으로 거절된다.
4. `cargo xtask ci`; `check-scope`.

## 수용 기준

오라클 1–4; api-note 절과 측정된 행(서버, 날짜, 경로와 함께).

## 금지

5절의 관측, 보상, 장면을 바꾸는 것; Rust 안의 pickle; 행동 kind를 추측하는 것(어댑터 아니면
거절); `docs/ARCHITECTURE*.md`; `tests/golden/**`.
