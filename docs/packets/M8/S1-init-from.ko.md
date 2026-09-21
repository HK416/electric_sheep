# M8 S1 — `es train`가 정책에서 시작한다: `[init] policy`, 그리고 `init.lock`이 무엇이 복사되었는지 말한다

스펙: §13.4("'이 정책에서 시작한다'는 lock에 적힌다"), §19.3(`base_model.lock`의 모양:
`training/` 안의 출처), §13.3(`policy_hash = H(base = previous checkpoint, training.lock)`),
§28.10 규칙 2(슬롯은 실제 값 아니면 `unset`), §28.11 파동 2. 선례: 패킷 M7/T5의
`base_model.lock`(`crates/es-data/src/training.rs`의 `Backbone`, `es/src/cmd/train.rs`의
`--init-backbone`, `train_act.py`의 `init_backbone`). 확장할 설계 노트:
`docs/design/training-recipe.md`(+ `.ko.md`), 새 절 "정책에서 시작하기".

## 질문

이어하기(S4c)와 그 뒤의 모든 파인튠은 이미 존재하는 가중치에서 시작한다. **레시피가 시작할
번들을 이름 짓고, 로워링된 모듈과 이름과 형태가 맞는 텐서만 정확히 복사하고, 나머지를
초기화하고, 그 모두를 `training_hash`에 들어가는 lock에 기록할 수 있는가 — 그래서
체크포인트로부터의 0스텝이 그 체크포인트를 비트 단위로 재현하도록?**

## 사양

* `training.toml`은 선택적 테이블 `[init] policy = "<bundle.esb>"`을 얻는다
  (`deny_unknown_fields`). IR 경로에서만; 외부(`lerobot`) 경로에서는 이름으로 거절된다
  (`lerobot-train`은 자기 자신의 `--policy.path`를 갖고 있고, 둘을 섞으면 출처를 날조하게
  된다).
* `es policy lower` 뒤 트레이너 전에: init 번들을 열고 그 `safetensors`를 읽어서(INV-16 —
  결코 다른 무엇도 아니다) 모든 텐서를 `contract.json`의 `weight_keys` / `weight_shapes`와
  비교한다: **copied** = 이름과 형태가 맞음; **initialised** = contract에 있지만 번들에는
  없음; **shape_mismatch** = 이름은 맞지만 형태가 다름(기록되지만 복사되지 않음). 복사된
  텐서가 0개면 이름으로 거절된다(`TRN-…`, 그 파일의 계열을 따른다): 아무것도 공유하지 않는
  정책에서 시작하는 것은 실수이지 웜스타트가 아니다.
* 복사된 텐서는 `--init-weights <safetensors>`로 트레이너에 닿는다(`--init-backbone` 옆의
  새 `train_act.py` 플래그; 나열된 이름만 정확히 로드하고 다른 키는 거절한다).
* `training/init.lock`: `{ source: <path as written>, policy_hash, learning_hash,
  copied: [...], initialised: [...], shape_mismatch: [{name, expected, found}] }`, 정렬된
  이름들. 이것은 §19.3의 슬롯이다: `training_hash`에 들어간다; `[init]`이 없으면 부재하므로
  기존의 모든 레시피의 `identity_hash` / `training_hash`는 움직이지 않는다(무엇에도 손대기
  전에 픽스처 레시피의 것을 고정하라).
* `[run] steps = 0`은 합법이고 "즉시 체크포인트"를 뜻한다: 트레이너는 옵티마이저 스텝 없이
  `0.esb`의 가중치를 쓴다. 이것이 오라클의 지레이자 실제 용례다(다시 패킹).
* `--dry-run`은 init 단계를 찍는다; 새 픽스처 레시피의 plan 골든은 추가분이다.

## context

```
crates/es-data/src/training.rs
crates/es-data/tests/**
crates/es/src/cmd/train.rs
crates/es/tests/cli.rs
python/es/train_act.py
tests/fixtures/visible-learning/training-init.toml
tests/golden/train/**
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M8/S1-init-from.md
docs/packets/M8/S1-init-from.ko.md
```

`training.rs`(테이블, lock, 해시 슬롯, 거절), `train.rs`(번들 열기, 비교, lock 쓰기, 플래그
전달), `train_act.py`(`--init-weights`, `steps = 0`), 새 픽스처 레시피, 골든(추가분만),
노트, 이 패킷.

## 오라클

1. `cargo test -p es-data init_lock_enters_training_hash_and_is_absent_without_init` —
   픽스처 레시피의 identity/training 해시가 고정되고 움직이지 않음; `[init]`이 있는
   레시피는 다르다.
2. `cargo test -p es --test cli train_init_from_bundle_zero_steps`(`ES_PYTHON`; 이유와 함께
   SKIP): T1의 40스텝 실행이 번들 `A`를 낸다; `[init] policy = A`, `steps = 0`인 레시피가
   모든 텐서가 `A`의 것과 비트 단위인 `0.esb`를 낸다; `init.lock`은 모든 이름을 `copied`로
   나열하고 initialised는 없다.
3. `cargo test -p es --test cli train_init_partial_and_refused` — Learning IR이 다른
   번들(`learning-pretrained.toml`의 그래프)에서의 init은 교집합을 복사하고 나머지를
   `initialised`로 나열하고 돈다; 텐서를 하나도 공유하지 않는 번들에서의 init은 이름으로
   거절된다; lerobot 경로의 `[init]`은 이름으로 거절된다.
4. 서버: `[init] policy = <U3's 20000.esb>`(`~/artifacts/plan-v/m7-u/U3/train/checkpoints/`),
   `steps = 0` → 비트 단위 U3; `policy_hash` 사슬이 노트에 기록됨.
5. `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S1-init-from.md`.

## 수용 기준

오라클 1–5. lock의 스키마와 세 거절을 이름으로 대는 노트의 새 절.

## 금지

pickle이나 `safetensors` 말고 어떤 로더든(INV-16); 형태가 다른 텐서를 복사하는 것(reshape는
추측이다); 기존 레시피의 해시를 움직이는 것; 외부 경로; 새 trait. `docs/ARCHITECTURE*.md`;
`tests/golden/**` 수정(추가분만).
