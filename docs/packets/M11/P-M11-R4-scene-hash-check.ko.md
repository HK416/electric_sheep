# M11 R4 — 실행은 Task IR이 고정한 장면이 아니면 거부한다

스펙: §5.3(`scene_hash`는 해시 체인의 입력이며, 같은 `execution_hash`는 같은 조건을 뜻한다),
§6(`SceneRef { path, scene_hash, asset_hash }`), §10.4, §3.5. I3(2026-09-24)에서 나온 것:
`es eval run --scene`은 넘겨받은 장면 파일을 무엇이든 불러온다. 서버에서 리치 A0 정책을 수정한 장면
네 개(감쇠 없음, frictionloss 없음 등)에서 돌렸는데, 네 실행 모두 커밋된 `execution_hash`
`08851281…`을 보고했다. 불러온 장면의 `SceneDesc::scene_hash()`를 Task IR의 `task.scene.scene_hash`와
비교하는 런타임 코드가 없다. `crates/`를 grep하면 선언된 값을 복사하는 Evaluation IR 잠금
(`es-eval/src/runner.rs`)만 나온다. 유형 B.

## 질문

**모든 실행기(`es eval run`, `es loop collect`, `es train`/`Rollout`, `es video`)가 거치는
`es_env::Env::new`에 검사 하나를 두어, 내용 해시가 Task IR이 선언한 `scene_hash`와 다른 장면을
거부할 수 있는가? 그러면서 커밋된 문서는 모두 커밋된 장면으로 그대로 돌고, 어떤 해시도 움직이지 않는가?**

## 명세

* `Env::new`는 `scene.scene_hash()`를 `task.scene.scene_hash`와 비교한다. 다르면 두 해시(짧은 hex)와
  장면 경로를 밝힌 `EnvError`를 돌려준다. 스펙에 이 경우의 오류 코드가 있으면 그것을 쓰고, 없으면
  코드를 추가하지 않는다. 메시지로 충분하며, 새 코드는 스펙 변경이므로 보고서에서 제안한다.
* 자리표시 `scene_hash`(예: `[7u8; 32]`)로 `TaskIr`를 만들어 `Env`를 돌리는 테스트는 이제 진짜 해시를
  선언해야 한다. 테스트 전용 도우미는 `task.scene.scene_hash = scene.scene_hash()`로 명시적으로 넣어도
  된다. 제품 코드에는 끄는 플래그가 없다. 수정한 장면으로 하는 진단 실행에는 그 실행만의 Task IR이
  필요하고, 바로 그것이 다른 조건이 되게 만든다.
* `asset_hash`는 범위 밖이다. 같은 방식으로 검사할 수 있는지는 보고서에 적는다.

## context

```
crates/es-env/src/env.rs
crates/es-env/src/lib.rs
crates/es-env/tests/**
crates/es-eval/tests/**
crates/es-data/tests/**
crates/es/tests/**
crates/es-py/tests/**
crates/es-py/src/**
docs/design/batch-domains.md
docs/design/batch-domains.ko.md
docs/packets/M11/P-M11-R4-scene-hash-check.md
docs/packets/M11/P-M11-R4-scene-hash-check.ko.md
```

## 오라클

1. `cargo test -p es-env scene_hash_mismatch_is_refused`: 커밋된 장면을 쓴 SO-101 과제는 만들어진다.
   같은 과제를 장면의 수정본(감쇠 값 하나를 바꾼 것)으로 만들면 거부되고, 오류가 두 해시를 밝힌다.
2. `cargo test -p es --test cli eval_run_refuses_an_edited_scene`: 커밋된 평가 문서로
   `es eval run --scene <수정본>`을 실행하면 그 메시지와 함께 0이 아닌 코드로 끝난다.
3. 기존 테스트가 모두 통과한다(`cargo xtask ci`). 커밋된 해시, 골든, `.estraj`, 보고서 중 어느 것도
   움직이지 않는다(verify-goldens).
4. fmt, clippy `-D warnings`, check-scope.

## 수용

오라클 1–4, 그리고 `batch-domains.md`(+ko)에 장면을 어디서 검사하는지 한 단락.

## 금지

끄는 플래그나 환경 변수; 커밋된 Task IR이나 장면 변경; `scene_hash`의 정의 변경.
