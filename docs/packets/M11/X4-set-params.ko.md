# M11 X4 — mass, friction, gain 추출이 물리에 닿는다

스펙: §28.14 규칙 4(모델 파라미터는 기존 `PhysicsBackend` trait의 메서드 하나를 통해 물리에 닿는다; 추출은
`(seed, env, episode, stream)`으로 키가 매겨지고 에피소드에 기록된다)와 규칙 1과 파동 1,
§6.3(`Randomization`), §5 "무작위화 대상", §17.2(능력이 없는 백엔드는 이름으로 거부한다), INV-17(새 trait
없음 — 일곱 중 하나에 메서드 하나는 허용). 설계 노트: `batch-domains.md`(+ko) 148줄 언저리의
한계("`PhysicsBackend`에는 모델 파라미터 API가 없다")는 이 패킷이 짓는 것으로 대체된다. 유형 B.

## the question

`body.<n>.mass`, `geom.<n>.friction`, `actuator.<n>.gain` 추출은 풀려서 `EpisodeMeta.param_scales`에
기록되지만 밀어 넣어지지는 않는다(`crates/es-env/src/randomize.rs:24`); MuJoCo CPU는 env 사이에
`MjModel` 하나를 공유하므로(`crates/es-physics-backend/python/mujoco_ref.py:39-47`) env별 파라미터는
오늘 존재할 수 없다. **`PhysicsBackend::set_params`로, 추출된 scale이 리셋 시점에 각 env의 모델에 닿고,
직접 한 MuJoCo 모델 편집을 비트 단위로 재현하며, 파라미터 대상을 선언하지 않은 모든 장면은 그대로 두는가?**

## spec

* `PhysicsBackend::set_params(&mut self, envs: &[u32], params: &[(Param, StableId, f64)]) ->
  Result<(), PhysicsError>` — 적재된 모델 값 상대 scale(`mass·s`, `friction[0..3]·s`, `gainprm[0]·s`,
  position actuator라면 `biasprm[1]·s`도 함께 — 서보가 서보로 남도록; 어느 것인지 문서에 적는다). 기본
  구현은 `PhysicsError::Unsupported("set_params")`를 돌려준다; `Capabilities`에 `Feature::ModelParams`가
  더해진다. `Param`이 그보다 위에 산다면 `es-physics-core`로 옮기거나(또는 거기서 re-export).
* MuJoCo CPU: `mujoco_ref.py`는 **env마다 `MjModel` 하나**를 유지한다(적재된 모델의 복사본, 첫
  `set_params`에서 지연 생성 — 그래서 한 번도 부르지 않는 실행은 모델 하나와 오늘의 바이트를 유지한다);
  `SetParams` 요청은 *원본* 모델에서 scale된 값을 쓰고(에피소드에 걸쳐 절대 누적하지 않음), 필드가
  요구할 때만 `mj_setConst`를 부른다(어느 것인지 문서화). `Env::reset`은 계획에 `Scale` 항목이 하나라도
  있으면 상태를 밀어 넣기 전에 리셋되는 env들에 `set_params`를 부른다; 추출 순서와 기록되는
  `param_scales`는 불변.
* MJWarp: 고정된 버전에서 `mujoco_warp`가 배치된 model 배열을 지원하면 world별 model 필드(에이전트가
  `docs/api-notes/mujoco-warp*.md`나 설치된 패키지를 확인하고 기록한다); 아니면 `ModelParams` 능력 없음,
  scale 대상을 가진 Task IR은 `mjwarp`에서 이름으로 거부. Newton: 선언 안 함.

## context

```
crates/es-physics-core/src/backend.rs
crates/es-physics-core/src/caps.rs
crates/es-physics-core/tests/**
crates/es-physics-backend/src/mujoco.rs
crates/es-physics-backend/src/mjwarp.rs
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/src/mapping.rs
crates/es-physics-backend/python/mujoco_ref.py
crates/es-physics-backend/python/mjwarp_ref.py
crates/es-physics-backend/tests/**
crates/es-env/src/randomize.rs
crates/es-env/src/env.rs
crates/es-env/tests/**
tests/fixtures/mjcf/**
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M11/X4-set-params.md
docs/packets/M11/X4-set-params.ko.md
```

## oracle

1. `cargo test -p es-physics-core set_params_default_is_unsupported`.
2. `ES_PYTHON=… cargo test -p es-physics-backend --test set_params` — env 둘짜리 장면에서 env 1의 body
   mass ×1.5, 한 geom의 friction ×0.5, 한 actuator gain ×1.2: 500스텝이 같은 모델 편집을 한 직접 파이썬
   `mujoco` 실행과 **비트 단위로 같다**(qpos, qvel); env 0은 편집 없는 실행과 비트 단위로 같다; 같은
   scale을 두 번 적용해도(리셋 두 번) 누적되지 않는다.
3. `cargo test -p es-env randomized_params_reach_the_backend` — env 4개에 걸쳐 `body.<n>.mass
   Uniform(0.8, 1.2)`를 가진 Task IR: 각 env에서 측정된 `body_mass`(테스트 훅이나 mujoco 응답으로 읽음)는
   `nominal × 기록된 scale`과 같다; `seed`는 정확히 재생된다.
4. 커밋된 task / scene / trajectory 골든은 모두 불변(데모는 scale 대상을 선언하지 않는다).
5. fmt, clippy `-D warnings`, check-scope, verify-goldens.

## acceptance

오라클 1~5가 로컬과 서버(CPU 큐)에서; `batch-domains.md` 절이 다시 쓰임(+ko); MJWarp 능력 결정이 그
증거와 함께 기록됨.

## forbidden

새 trait; `es-safety`; scale 누적; 파라미터 대상이 없는데 env별 모델 생성; 돌려보지 않고 MJWarp 지원을
주장하기.
