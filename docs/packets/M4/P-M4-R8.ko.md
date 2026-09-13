<!-- Korean translation of docs/packets/M4/P-M4-R8.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R8 — Python 빌더의 Rust 미러를 위한 공유 골든 벡터

M4 리뷰 should-fix S-14(`docs/reviews/M4.md`)를 닫는다: `python/es/builder.py:35`의
`stable_id`는 `es_core::StableId::from_path`(`crates/es-core/src/id.rs:19-24`)를 재구현하고,
`builder.py:476-486`의 `_feature_ty`/`_chunk_ty`는 `crates/es-ir/src/learning.rs`의 비공개
`feature()`/`chunk()`를 재구현한다. `id.rs:81`의 테스트는 자기 일관성만(동일 입력이 동일
id를 낸다는 것) 단언했으므로, 어느 쪽도 *다른* 쪽의 실제 바이트에 고정되어 있지 않았다;
어긋남이 생겨도 어디서도 테스트가 실패하지 않은 채 IR이 잘못된 바디에 대해 검증될 것이다.

Spec: 사양 §1.4(오라클 우선 — 공유 오라클이 없는 Python 미러는 설계 문제가 아니라 구현
공백이다), §5.3(해시 체인 무결성은 `StableId`가 어디서나 일치하는 데 의존한다),
§3.4(결정론). IR이나 해싱 동작에는 변화가 없다; 이 패킷은 누락된 고정값만 추가한다.

## context (범위)

```
crates/es-core/src/id.rs      (golden hex literal added to the existing test)
python/es/selfcheck.py        (new — recomputes the same vectors in Python)
python/es/README.md           (new — documents the Rust side as canonical)
docs/packets/M4/P-M4-R8.md    (new)
```

## spec (사양)

- `StableId::from_path("robot/arm/joint_1")`은 `blake3(path)`를 16바이트로 자른 것이며, hex
  형태는 `6d2e29e8077ed3c571f21602d29c7145`다. `id::tests::from_path_is_stable_and_distinct`
  (Rust)와 `python/es/selfcheck.py`의 `STABLE_ID_VECTOR`(Python) 양쪽에 리터럴로 고정되어
  있다 — 둘 다 같은 문자열에서 다시 계산하며 바이트 단위로 일치해야 한다.
- `_feature_ty(dim, tokens)`는 `feature()`를 미러링한다: `tokens == 0`이면 shape `[dim]`,
  아니면 `[tokens, dim]`이며, 항상 `Dimensionless`/`Policy`다. `(512, 0) -> shape [512]`
  (`FEATURE_TY_VECTOR`)에 고정되어 있다.
- `_chunk_ty(horizon, action_dim)`는 `chunk()`를 미러링한다: shape `[horizon, action_dim]`,
  `Normalized(-1, 1)`(`action_unit()`), `Policy` 프레임. ACT 자신의 `(100, 14)`
  (`CHUNK_TY_VECTOR`)에 고정되어 있으며, `docs/api-notes/lerobot-act.md`의
  `chunk_size`/`action_dim`과 일치한다.
- `python/es/selfcheck.py`는 독립적이고 의존성 없는 검사다: `python/es/builder.py`만
  임포트하여 위의 세 벡터를 단언한다. 테스트 프레임워크도 픽스처도 없다 — `assert` 세 개와
  print 하나뿐이며, `python/es`의 나머지 스타일과 일치한다.
- `python/es/README.md`는 `crates/es-core/src/id.rs`와 `crates/es-ir/src/learning.rs`가
  정본(canonical)임을 명확히 밝힌다; Python 함수들은 저작(authoring) 빌더가 모든 id와 포트
  타입마다 Rust를 셸아웃하지 않아도 되도록 존재하는 편의용 미러일 뿐이다.

## oracle (오라클)

```
cargo fmt -p es-core --check
cargo clippy -p es-core --all-targets -- -D warnings
cargo test -p es-core
PYTHONPATH=python <venv>/Scripts/python.exe -m es.selfcheck
```

(`es.selfcheck`는 `es.builder`를 임포트하고, 이는 다시 `es_native`를 임포트한다 — venv에
아직 설치되어 있지 않다면 `python/es`에서 `maturin develop --release --features python`으로
한 번 빌드한다.)

## acceptance (수용 기준)

- `cargo test -p es-core`는 고정된 hex 리터럴이 있는 상태로 통과한다.
- `python -m es.selfcheck`는 ACT 오라클에 쓰이는 것과 같은 venv에 대해 `OK`를 출력하고
  0으로 종료한다 (spec §1.4).
- `StableId::from_path`나 `stable_id` 중 하나(또는 `feature()`/`_feature_ty`,
  `chunk()`/`_chunk_ty`)에 의도적으로 한 글자를 수정하면 위 두 검사 중 정확히 하나가
  실패한다 — 이 벡터들이 장식이 아니라 실제로 결과를 좌우함을 확인한다.
- `docs/packets/M4/P-M4-R8.md`와 `python/es/README.md`는 존재하며 서로를 상호 참조한다.

## forbidden (금지)

- `crates/es-py/**`, `crates/es-ir/**` — pyo3 확장이나 IR crate의 동작 변경 없음; 이 패킷은
  기존 출력을 고정할 뿐이다.
- 다른 패킷의 범위(`es-env`, `es-eval`, `es-data`, `es-usd`, `es-script`, `es-gpu`,
  `es-render`, 그리고 S-15가 소유하는 범위를 넘어서는 `docs/api-notes/lerobot-act.md`).
