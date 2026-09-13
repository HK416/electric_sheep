<!-- Korean translation of docs/packets/M2/W6-python-builder.md. The English file is the working copy; regenerate this when it changes. -->

# M2 W6 (나머지) — Python authoring builder (`es-py`, spec 14.2)

Spec 14.1/14.2이 설명하는 Python authoring 프런트엔드를 구현한다: Python에서
접근 가능한 다섯 개(여기서는 네 개 — Evaluation IR은 범위 밖)의 타입 IR
builder이며, 에디터와 CLI가 쓰는 것과 같은 노드 팩토리로 뒷받침되므로
`es.Task(...)`/`.save(...)`와 손으로 쓴 `task.toml`은 동일한 IR을 만들어낸다.
Design note: `docs/design/python-builder.md`. API digest: `docs/api-notes/pyo3.md`.

## context (범위)

```
crates/es-py/**                         (신규 crate, layer 11)
Cargo.toml                              (`es-py` member 항목, `pyo3` 워크스페이스 의존성 추가)
python/es/**                            (신규: __init__.py, builder.py, math.py, pyproject.toml,
                                          examples/pick_place.py)
docs/api-notes/pyo3.md                  (신규)
docs/design/python-builder.md           (신규)
docs/packets/M2/W6-python-builder.md    (신규)
```

## spec (사양)

- **`crates/es-py/src/builder.rs`**: 언어 중립적인 core, `pyo3` 없음, 기본으로
  컴파일되고 테스트됨(feature 없이 `cargo test -p es-py`). `TaskBuilder`,
  `ObservationBuilder`, `LearningBuilder`는
  `TaskNodeRegistry`/`LearningNodeRegistry`/`ObservationNode` 자체의 태그된
  enum을 `add(kind, params: serde_json::Value) -> Result<NodeId,
  Diagnostic>`, `connect`, 그 IR의 경계 선언을 위한 `declare_*` 메서드,
  `build(self) -> Result<Ir, Vec<Diagnostic>>`(IR을 조립하고 자신의
  `validate()`를 호출함), 그리고 `es_ir::serial`을 쓰는 연관 함수
  `save_toml(&Ir, &Path)`로 감싼다. `DeploymentBuilder`는 대신 직접적인
  setter다(`DeploymentIr`는 그래프가 아니라 구조체다 — design note 참고).
  이 파일 안 어디에도 node-kind별 match arm이 없다.
- **`crates/es-py/src/pybind.rs`** (`#[cfg(feature = "python")]`): pyo3
  `es_native` 모듈. `Task`/`Observation`/`Learning`/`Deployment` pyclass는
  일반적인 `add`/`connect`/`declare_*`/`build`/`save`에 더해, spec 14.2가
  이름 붙인 소수의 설탕 메서드(`Task.observe`, `.reward`, `.terminate`;
  정적 편의 생성자로서의 `Learning.act`)를 노출한다 — 이들 모두 여전히
  `add`/`connect`를 호출할 뿐이다. 오류는 `(display_text,
  diagnostics_json)`을 싣는 `EsDiagnosticError`가 된다.
- **`python/es/`**: `es_native` 위에 지어진, spec 14.2 스니펫의 이름과 호출
  shape에 맞춘 순수 Python 패키지(`Task`, `Observation`, `Learning`,
  `Deployment`, 연산자 오버로드를 가진 `Expr`, `es.math.norm`).
  `pyproject.toml`(`[tool.maturin]`, `crates/es-py`로의 `manifest-path`,
  `features = ["python"]`)이 `maturin develop`이 읽는 것이다.
  `examples/pick_place.py`는 spec 14.2 스니펫을 재현하며 `task.toml` /
  `observation.toml` / `learning.toml` / `deployment.toml`(spec 14.3
  포맷)을 쓴다.
- **`pyo3`는 선택적이며 기본적으로 꺼져 있다**(`es-py`의 `python` feature,
  `dep:pyo3`): 기본 워크스페이스 빌드는 결코 libpython을 링크하지 않는다.
  `maturin`이 이를 명시적으로 활성화한다.
- 제약: 영어만, `BTreeMap`만(직접 필요한 것은 없다 — `es-ir` 타입에서
  물려받음), 새 trait 없음(`INV-17`: `TaskNodeFactory`/`LearningNodeFactory`는
  재사용될 뿐 확장되지 않는다), `builder.rs` + `pybind.rs`에 걸쳐 대략
  1000줄의 새 Rust 코드.

## oracle (오라클)

```
cargo fmt -p es-py --check
cargo clippy -p es-py --all-targets -- -D warnings
cargo test -p es-py
cargo xtask layering
```

- `crates/es-py/src/builder.rs` 단위 테스트(Python 없이): `add`/`connect`로
  최소한의 유효한 Task 그래프를 빌드하고 `save_toml`/`task_from_toml`로
  왕복시키기; 알 수 없는 kind가 패닉하는 대신 `Diagnostic`으로
  `FACTORY-001`을 보고하기; `declare_obs`가 `ObsChannel`을 왕복시키기;
  레지스트리 없이 `ObservationIr`을 빌드하기; 필요한 상태(policy 핸들;
  모든 `DeploymentIr` 필드)가 빠졌을 때 `LearningBuilder`/`DeploymentBuilder`
  둘 다 패닉하는 대신 명확한 진단을 보고하기.
- `crates/es-py/tests/python_example.rs`: `ES_PYTHON`이 설정되지 않으면
  **SKIP**한다(shell out할 `es` CLI가 아직 존재하지 않는다 —
  `CLAUDE.md`). `ES_PYTHON=<es가 설치된 python>`이면
  `python/es/examples/pick_place.py`를 실행하고, 그것이 쓰는 네 개의 TOML
  파일을 `es_ir::serial::*_from_toml`로 파싱하며, 각 IR 자신의
  `.validate()`가 오류 0개를 보고함을 단언한다 — 언젠가 `es ir validate`
  CLI가 호출하게 될 것과 같은 validator다.

## acceptance (수용 기준)

- `cargo fmt -p es-py --check && cargo clippy -p es-py --all-targets -- -D
  warnings && cargo test -p es-py && cargo xtask layering` 모두 초록색.
- `cargo check --workspace --all-targets`는 `es-py`를 깨끗하게 컴파일한다
  (동시에 편집 중인 다른 crate에서의 워크스페이스 전역 깨짐은 이 패킷의
  통제와 게이트 밖이다).
- 이 세션에서 라이브로 검증됨: `maturin develop --release`(저장소 루트의
  `.venv`로, `python/es`에서)가 `es`와 `es_native` 확장을 빌드하고
  설치한다; `python/es/examples/pick_place.py`를 실행하면
  `task.toml`/`observation.toml`/`learning.toml`/`deployment.toml`이
  만들어지고, `ES_PYTHON=<venv python> cargo test -p es-py --test
  python_example`가 통과한다(넷 모두 오류 0개로 검증됨).

## forbidden (금지)

- `crates/es-py`(신규)를 제외한 어떤 crate든, 그리고 루트 `Cargo.toml`에
  나열된 두 편집(`es-py` member 항목과 `pyo3` 워크스페이스 의존성 줄) —
  그 외 모든 crate는 동시에 진행 중인 패킷이 소유한다.
- Evaluation IR — spec 14.2 자체의 스니펫도 이를 builder화하지 않는다;
  여기서는 범위 밖이다.
- Cross-IR 검사(`es-ir::cross`, spec 11.1)를 클라이언트 쪽에서 재현하는
  것 — 그것은 이 builder가 아니라 컴파일러의 일이다.
- 커밋하는 것.
