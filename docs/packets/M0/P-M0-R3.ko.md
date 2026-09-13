<!-- Korean translation of docs/packets/M0/P-M0-R3.md. The English file is the working copy; regenerate this when it changes. -->

# P-M0-R3 — MJCF `<default>` 클래스 그래프 강화(hardening)

Spec: §3.1 (컨벤션), §17.2 (백엔드 의미 매핑), §4.3 (MuJoCo 계열 백엔드). M0 리뷰
(`docs/reviews/M0.md`, `crates/es-assets/src/mjcf/mod.rs`에 대한 두 건의 Should-fix 항목)의
후속 작업. 리뷰 등급 A. P32를 기반으로 한다.

## context

```
crates/es-assets/src/mjcf/mod.rs
tests/fixtures/mjcf/default_cycle.xml
tests/fixtures/mjcf/default_duplicate.xml
docs/packets/M0/P-M0-R3.md
```

## spec

MJCF는 신뢰할 수 없는(untrusted) 입력이므로, 클래스 그래프는 생성 시점에 비순환(acyclic)이
되도록 만들어야 하며, 그 위를 순회하는 과정 역시 어차피 유계(bounded)여야 한다.

- **중복된 클래스 이름은 오류다**, MuJoCo와 동일하게 맞춘다. `MjcfError::DuplicateClass
  { line, class }`. 이전에는 두 번째 `<default class="x">`가 속성을 병합하면서 첫 번째의
  부모 링크를 덮어써서, 이전 상속 관계가 아무런 진단 없이 사라졌다. 경고가 아니라 오류로
  처리하기로 한 이유는, 조용히 바뀐 부모 링크가 호출자가 알아챌 방법 없이 `scene_hash`를
  바꾸기 때문이다.
- `main`은 유일한 예외다: 이는 암묵적인 루트 클래스이므로, 반복되는 **최상위(top-level)**
  `<default>` 섹션은 이전과 정확히 동일하게 여전히 그 안으로 병합된다.
- 따라서 **중첩된(nested)** `<default>`는 아직 아무도 선언하지 않은 클래스 이름을 가져야
  한다. 이름이 없는 중첩 `<default>`는 기본값으로 `main`이라는 이름을 갖게 되어 루트
  클래스를 그 자신의 자손으로 만들어버리므로, 같은 중복 검사에 의해 거부된다
  (`duplicate default class \`main\``). 중첩된 `<default class="main">`도 마찬가지다.
- 이 규칙들이 합쳐져 클래스 그래프는 `main`을 루트로 하는 포레스트(forest)가 된다: 중첩된
  클래스의 부모는 그것을 엄격히 둘러싸는(strictly-enclosing) 클래스이며, 어떤 이름도
  반복되지 않는다.
- `resolve`는 어쨌든 한계를 둔다 — 클래스 테이블보다 긴 부모 체인은 어떤 클래스를
  재방문했다는 뜻이며, 프로세스가 메모리 부족에 이를 때까지 `chain`에 계속 push하는 대신
  `MjcfError::ClassCycle { line, class }`를 반환한다. 이는 안전장치(backstop)이며, 파싱된
  파일로부터는 도달할 수 없다.
- 새로 추가된 두 variant 모두 다른 모든 `MjcfError`와 마찬가지로 line 정보를 담는다.

## oracle

```
cargo test -p es-assets mjcf
```

## acceptance

- `default_cycle.xml`(리뷰에서 나온 `<default class="a"><default class="a"/></default>`)는
  `line 6: duplicate default class \`a\``를 반환한다 — 멈추지도(hang), OOM이 나지도 않는다.
- `default_duplicate.xml`은 `line 14: duplicate default class \`x\``를 반환한다.
- 두 픽스처 테스트 모두 워커 스레드 안에서 파싱하며, 5초 이내에 결과가 오지 않으면 테스트를
  실패시킨다 — 그래서 무제한 순회로의 회귀(regression)가 CI를 멈추게 하는 대신 실패로
  보고된다.
- `an_unnamed_nested_default_is_rejected`: `class`가 없는 중첩 `<default>`는 오류다.
- `repeated_top_level_defaults_still_merge_into_main`: 최상위 `<default>` 섹션 두 개는
  오류도 경고도 없이 파싱된다.
- `a_cyclic_class_graph_cannot_loop_resolve`: 직접 구성한 `a -> b -> a` 클래스 테이블에서
  `resolve`가 `ClassCycle`을 반환한다.
- `arm2.xml`과 기존의 다른 모든 MJCF 픽스처는 여전히 같은 값으로 파싱되며, 뮤테이션 퍼저
  (`fuzz_mjcf_importer_never_panics_on_mutated_fixtures`)도 새로 추가된 두 픽스처를 함께
  커버한다.

## forbidden

`context` 밖의 모든 파일. 기존 픽스처 변경. `crates/es-assets/src/urdf.rs`, `gltf.rs`,
`scene.rs`, 또는 `mjcf` 서브모듈(`attrs.rs`, `elements.rs`, `orient.rs`). trait나 XML
의존성 추가. `HashMap` / `HashSet`(§3.4). 참조된 파일을 resolve하거나 읽는 것.
