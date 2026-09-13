<!-- Korean translation of docs/api-notes/pyo3.md. The English file is the working copy; regenerate this when it changes. -->

# pyo3 (고정된 API 다이제스트)

`crates/es-py`(M2 W6, spec 14.2 Python authoring builder)를 위해 고정됨. 버전과
feature 집합은 워크스페이스 `Cargo.toml`에 있다; 이 노트는 API 표면 자체에 대해
기억해 둘 것이지, `docs.rs/pyo3/0.29.2`를 대신하는 것이 아니다.

## Pin

```toml
# workspace Cargo.toml
pyo3 = { version = "0.29.2", features = ["extension-module", "abi3-py312"] }
```

- **`abi3-py312`**: 3.12 시점의 CPython 안정 ABI에 대해 컴파일한다. 그러면 휠
  하나(`*-cp312-abi3-*.whl`)가 3.12, 3.13, 3.14, ...에서 로드된다 —
  마이너 버전마다 다시 빌드할 필요가 없다. 이 저장소가 고정한 Python(3.12
  시스템 + `.venv`)과 일치한다. 이 고정을 잃는 것(`abi3-py3xx` 태그를 올리는
  것)은 일상적인 버전 업이 아니라 호환성 결정이다.
- **`extension-module`**: pyo3에게 libpython을 지연 링크하라고 말한다(심볼은
  빌드 시점이 아니라 import 시점에 호스트 인터프리터로부터 온다). Python이
  로드하는 `cdylib`에는 필요하지만, Rust 바이너리 *안에* Python을 임베드하는
  데는 맞지 않다(이 패킷이 하는 일이 아니다).
- 두 feature 모두 *워크스페이스* 수준에서 요청되지만(그래서 `cargo
  metadata`/lockfile이 하나뿐인 의존자에 대해 단일 feature로 남는다),
  `es-py` 자신의 `python` feature가 켜져야만 빌드에 도달한다
  (`pyo3 = { workspace = true, optional = true }` + `python = ["dep:pyo3"]`).
  기본 `cargo build`/`cargo test --workspace`는 이를 절대 활성화하지 않으므로,
  CI는 워크스페이스의 나머지를 빌드하기 위해 libpython 헤더나 Python
  인터프리터가 결코 필요하지 않다(CLAUDE.md, spec 2.1).

## 여기서 사용되는 모듈 shape

- `#[pymodule] fn es_native(m: &Bound<'_, PyModule>) -> PyResult<()>` 하나
  (0.21+의 "Bound" API — 0.21 이전의 `&PyModule` 시그니처는 사라졌다). 이
  이름은 maturin의 `module-name`의 마지막 경로 세그먼트와 **반드시**
  일치해야 하는데(`python/es/pyproject.toml`은 `module-name =
  "es.es_native"`를 설정한다), 매크로가 만들어내는 `PyInit_<name>` C 심볼이
  함수가 어디에 선언되었는지나 `pub`인지가 아니라 함수 이름에서 도출되기
  때문이다.
- 모든 builder wrapper(`Task`, `Observation`, `Learning`, `Deployment`)에
  `#[pyclass(unsendable)]`: pyo3는 평범한 `#[pyclass]`가 `Send + Sync`일
  것을 요구한다(원리적으로 Python의 GIL-해제 경계를 안전하게 건널 수 있도록);
  이들이 감싸는 노드 팩토리(`Box<dyn TaskNodeFactory>` 등)는 둘 다 아니며,
  authoring 스크립트는 오직 한 스레드에서만 한 builder를 건드리므로,
  아무것도 필요하지 않은 곳에 `Mutex` 래핑을 추가하는 대신
  `unsendable`(검사를 건너뛰고, 생성된 것과 다른 OS 스레드에서 사용되면
  패닉함)이 정직한 선택이다.
- 소비하는 메서드(`build`, `save`)는 값으로서의 `self`가 아니라 `.take()`를
  쓰는 `Option<Builder>` 필드로 모델링된다: `#[pymethods]`는
  `#[pyclass(unsendable)]` 타입 인스턴스에 대해 값으로 receiver를 받을 수
  없는데, Python 쪽이 여전히 그 참조를 갖고 있기 때문이다.
- 커스텀 예외: `pyo3::create_exception!(es_native, EsDiagnosticError,
  PyException)`. `(display_text, diagnostics_json)`이라는 2-튜플과 함께
  발생시키므로, 호출자는 이를 그대로 출력하거나
  `json.loads(e.args[1])`로 구조화된 `Diagnostic`을 얻을 수 있다.
- 파라미터는 손으로 순회하는 `PyObject` 트리가 아니라 **JSON 텍스트**로
  FFI 경계를 건넌다: Python의 표준 라이브러리 `json.dumps`/`json.loads`가
  이미 정확히 그 변환을 하며, `pythonize`/`serde-pyobject` 스타일의 crate는
  이 패킷의 승인된 의존성 목록(`pyo3`뿐)에 없다 —
  `crates/es-py/src/pybind.rs`의 모듈 문서 주석 참고.

## 이 패킷을 만들면서 걸린 함정

- `#[pymethods]` `impl` 블록은 그 안의 **모든** 함수를 Python에 노출하며,
  private 헬퍼로 의도된 것도 예외가 아니다 — 블록 안에는 `pub(crate)`
  스타일의 옵트아웃이 없다. `fn inner(&mut self) -> PyResult<&mut Builder>`
  같은 헬퍼(유효한 Python 반환 타입이 아니다 — `Builder`는 `pyclass`가
  아니다)는 대신 두 번째의, 평범한 `impl Task { .. }` 블록에 살아야 한다.
- `unsendable` 없는 `#[pyclass]`는 런타임이 아니라 컴파일 타임에, 긴
  trait-resolution 체인의 여러 `note:` 아래에 파묻힌 근본 원인을 가진
  `Send`/`Sync` trait-bound 오류로 실패한다 — `Box<dyn Trait>`를 감싸는
  어떤 pyclass든 가장 먼저 확인해 볼 가치가 있다.
