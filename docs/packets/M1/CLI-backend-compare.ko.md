<!-- Korean translation of docs/packets/M1/CLI-backend-compare.md. The English file is the working copy; regenerate this when it changes. -->

# CLI-backend-compare — `es backend compare`

Spec: §17.2 (백엔드 시맨틱 매핑, `es backend compare`), §17.3 (결정성 등급),
§14.4 (severity: error인 미매핑 행은 실행을 막는다), §3.5 (결정성 등급 비교 지표).

## context (범위)

```
crates/es/Cargo.toml
crates/es/src/cmd/mod.rs
crates/es/src/cmd/backend.rs
crates/es/src/main.rs
crates/es/tests/cli.rs
docs/packets/M1/CLI-backend-compare.md
```

## spec (사양)

`es backend compare`(`es-physics-backend::mapping`에 대한 CLI 배선이며, 실제 작업은 이미
`mapping_report`/`compare_backends`가 수행한다)는 씬(`--scene <file.xml|urdf>`, 또는
Task IR의 `SceneRef::path`를 읽는 별칭인 `--task <task.toml>`)과 콤마로 구분된
`BackendKind::{mujoco-cpu,mjwarp,newton,physx}`에서 가져온 `--backends` 목록을 받는다.

- 파일 확장자에 맞는 임포터로 씬을 파싱한다: `.xml`은 `es_assets::parse_mjcf`,
  `.urdf`는 `PackageResolver::from_env()`를 사용한 `es_assets::urdf::parse_urdf`.
  `--task`는 `SceneRef`를 읽는다; 해시 전용 참조(`path`가 비어 있음)는 사용자에게
  `--scene`을 직접 넘기라고 알리는 런타임 에러다.
- 요청된 모든 백엔드에 대해 `mapping_report(scene, kind)`를 무조건 먼저 출력한다 —
  어댑터가 없는 백엔드(`newton`, `physx`)나 사용 불가능한 것으로 판명된 백엔드에
  대해서도 실행되어야 한다. 시맨틱 매핑 테이블이 항상 보이도록 하기 위해서다.
- 그 다음, 요청된 백엔드마다: 매핑 리포트가 `blocked`(spec 14.4)인 경우 가용성이
  결코 검사되지 않고, 차단하는 기능들을 명시하며 `SKIPPED (blocked by mapping
  report, ...)`로 보고된다. 그렇지 않으면 `MuJoCoCpuBackend::is_available()` /
  `MjWarpBackend::is_available()`이 `available`인지 `SKIPPED (<reason>)`인지를
  결정한다; `newton`/`physx`는 항상 `SKIPPED (not implemented (M2/M3))`를
  출력한다(어댑터가 존재하지 않는다).
- 최종적으로 `available`이 된 모든 백엔드 쌍에 대해: 둘 다 인스턴스를 생성하고,
  `LoadConfig { n_envs: --envs, seed: --seed, rate: None }`으로 각각을 미리
  로드하며(그래서 `model_info()`가 `None`일 때만 로드하는 `compare_backends`가
  자신의 `LoadConfig::default()` 대신 이 값을 집어 쓴다), 로드된 `ModelInfo::nu`로부터
  제어 시퀀스를 구성하고(기본값은 전부 0, 또는 `--ctrl-random`일 때는 시드된
  유사난수 — 작은 인라인 splitmix64, `rand` 의존성 없음), `compare_backends`를
  실행하고, 그 `Display` 테이블을 출력한다(양쪽의 매핑 리포트를 그 타입 자체의
  `Display` impl로 반복 출력한다 — `es-physics-backend`를 건드려 바꾸는 대신 받아들인
  중복이며, 그렇게 하는 것은 이 패킷의 범위 밖이다).
- 종료 코드: 요청된 모든 백엔드가 비교를 실행했거나 환경적 이유(사용 불가 / 미구현)로
  스킵되었으면 0; 요청된 백엔드의 매핑 리포트가 `blocked`(spec 14.4)이거나, 어떤
  비교의 `max_dqpos`가 `--tol`(기본값 `1e-6`)을 초과하면 1; 사용법 오류(누락된
  `--backends`/`--scene`/`--task`, 알 수 없는 백엔드 이름, 잘못된 플래그 값)면 2.
- 사용 불가능하거나 차단된 백엔드에서 결코 패닉하지 않는다 — 둘 다 `Err`가 아니라
  평범한 `SKIPPED` 행이다.

## oracle (오라클)

```
cargo fmt -p es --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es
```

## acceptance (수용 기준)

- `backend compare --scene tests/fixtures/mjcf/pendulum.xml --backends mujoco-cpu,mjwarp`는
  두 백엔드의 매핑 리포트를 모두 출력한다; `mjwarp`는 `blocked`이며(픽스처의
  `option cone="elliptic"`는 `mjwarp`에서 피라미달 전용 매핑이 없다, spec 17.2) 항상
  `SKIPPED`로 보고되므로, Python `mujoco`가 설치되어 있는지 여부와 무관하게 프로세스는
  1로 종료한다.
- 같은 파일을 `--backends mujoco-cpu`만으로 실행하면(차단되지 않음) 0으로 종료하고,
  `mujoco` 패키지가 없는 CI Python 환경에서는 `mujoco-cpu`가 환경적 이유로 `SKIPPED`로
  보고된다.
- `--backends not-a-real-backend`는 2로 종료한다: 백엔드 이름은 씬 파일이 읽히기
  전, 플래그 파싱 중에 검증된다.
- 어떤 서브커맨드 경로도 CLI 입력, 파일 내용, 스폰된 백엔드의 출력에 대해
  `.unwrap()`/`.expect()`를 호출하지 않는다.

## forbidden (금지)

- `crates/es-ir/**`, `crates/es-compile/**`, `crates/es-runtime-embedded/**`,
  `xtask/**`, 루트 `Cargo.toml`, 그리고 모든 `docs/ARCHITECTURE*.md` — 동시에 진행
  중인 다른 패킷들의 소유다.
- `crates/es-physics-backend/**`, `crates/es-physics-core/**`, `crates/es-assets/**` —
  이 패킷은 이들의 공개 API(`mapping_report`, `compare_backends`, `BackendKind`,
  `MuJoCoCpuBackend`, `MjWarpBackend`, `parse_mjcf`, `urdf::parse_urdf`)만
  소비한다.
- `clap`이나 그 밖의 CLI 인자 파싱 크레이트; `--ctrl-random`을 위한 `rand`나 그 밖의
  RNG 크레이트(인라인 시드 splitmix64로 충분하다).
- `HashMap`/`HashSet` — 저장소 관례에 따라 `BTreeMap`/`BTreeSet` 또는 작은 `Vec`만
  허용된다.
- 커밋.
