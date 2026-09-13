<!-- Korean translation of docs/packets/M1/CLI-backend-compare.md. The English file is the working copy; regenerate this when it changes. -->

# CLI-backend-compare — `es backend compare`

Spec: §17.2 (백엔드 의미 매핑, `es backend compare`), §17.3 (결정성 계층),
§14.4 (`severity: error`인 미매핑 행은 실행을 차단한다), §3.5 (결정성-계층
비교 지표).

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

`es backend compare`(`es-physics-backend::mapping`을 위한 CLI 연결이며,
실제 작업은 이미 그것의 `mapping_report`/`compare_backends`가 한다)는 씬
(`--scene <file.xml|urdf>`, 또는 Task IR의 `SceneRef::path`를 읽는 별칭인
`--task <task.toml>`)과 `BackendKind::{mujoco-cpu,mjwarp,newton,physx}`에서
뽑은 쉼표로 구분된 `--backends` 목록을 받는다.

- 파일 확장자에 맞는 importer로 씬을 파싱한다: `.xml`에는
  `es_assets::parse_mjcf`, `.urdf`에는 `PackageResolver::from_env()`를
  가진 `es_assets::urdf::parse_urdf`. `--task`는 `SceneRef`를 읽는다;
  해시만 있는 ref(`path`가 비어 있음)는 사용자에게 `--scene`을 직접
  넘기라고 알리는 런타임 오류다.
- 요청된 모든 백엔드에 대해 `mapping_report(scene, kind)`를 무조건 먼저
  출력한다 -- 이는 어댑터가 없는 백엔드(`physx`)나 사용 불가능한 것으로
  밝혀진 백엔드에 대해서도 실행되어야 하며, 그래야 의미-매핑 표가 항상
  보인다.
- 그다음, 요청된 백엔드마다: `blocked`(spec 14.4)인 매핑 리포트는 가용성
  검사를 전혀 받지 않고 차단하는 feature들을 지목하며
  `SKIPPED (blocked by mapping report, ...)`로 보고된다. 그렇지 않으면
  `MuJoCoCpuBackend::is_available()` / `MjWarpBackend::is_available()` /
  `NewtonBackend::is_available()`이 `available`인지 `SKIPPED (<reason>)`인지를
  결정한다; `physx`는 항상 `SKIPPED (not implemented (M2/M3))`를 출력한다
  (어댑터가 존재하지 않는다).
- `available`로 끝난 모든 백엔드 쌍에 대해, 두 인스턴스를 생성하고, 각각을
  `LoadConfig { n_envs: --envs, seed: --seed, rate: None }`로 미리 로드한
  뒤(그래서 `model_info()`가 `None`일 때만 로드하는 `compare_backends`가
  자신의 `LoadConfig::default()` 대신 이것을 집어 든다), 로드된
  `ModelInfo::nu`로부터 제어 시퀀스를 만들고(기본적으로 전부 0이거나,
  `--ctrl-random` 아래에서는 시드된 의사난수 -- `rand` 의존성 없는 작은
  인라인 splitmix64), `compare_backends`를 실행하고, 그 `Display` 표를
  출력한다(각 쪽의 매핑 리포트를 그 타입 자체의 `Display` 구현으로 반복해
  담는데 -- `es-physics-backend`에 손대어 이를 바꾸는 대신 받아들여진
  중복이며, 이는 이 패킷의 범위 밖이다).
- 종료 코드: 요청된 모든 백엔드가 비교를 실행했거나 환경상의 이유(사용
  불가 / 미구현)로 건너뛰어졌으면 0; 요청된 어떤 백엔드의 매핑 리포트든
  `blocked`이거나(spec 14.4), 어떤 비교의 `max_dqpos`든 `--tol`(기본값
  `1e-6`)을 초과하면 1; 사용법 오류(빠진 `--backends`/`--scene`/`--task`,
  알 수 없는 백엔드 이름, 잘못된 플래그 값)면 2.
- 사용 불가능하거나 차단된 백엔드에 대해 절대 패닉하지 않는다 -- 둘 다
  `Err`가 아니라 평범한 `SKIPPED` 행이다.

## oracle (오라클)

```
cargo fmt -p es --check
cargo clippy -p es --all-targets -- -D warnings
cargo test -p es
```

## acceptance (수용 기준)

- `backend compare --scene tests/fixtures/mjcf/pendulum.xml --backends
  mujoco-cpu,mjwarp`는 두 백엔드의 매핑 리포트를 출력한다; `mjwarp`는
  `blocked`이며(픽스처의 `option cone="elliptic"`은 `mjwarp`에서
  pyramidal 전용 매핑을 갖지 않는다, spec 17.2) 항상 `SKIPPED`로
  보고되므로, Python `mujoco`가 우연히 설치되어 있든 아니든 프로세스는
  1로 종료한다.
- 같은 파일에 `--backends mujoco-cpu`만 있으면(차단되지 않음) 0으로
  종료하며, `mujoco` 패키지가 없는 CI Python에서는 `mujoco-cpu`가 환경상의
  이유로 `SKIPPED`로 보고된다.
- `--backends not-a-real-backend`는 2로 종료한다: 백엔드 이름은 씬 파일이
  읽히기 전에, 플래그 파싱 중에 검증된다.
- 어떤 서브커맨드 경로도 CLI 입력, 파일 내용, 또는 spawn된 백엔드의
  출력에 대해 `.unwrap()`/`.expect()`를 호출하지 않는다.

## forbidden (금지)

- `crates/es-ir/**`, `crates/es-compile/**`, `crates/es-runtime-embedded/**`,
  `xtask/**`, 루트 `Cargo.toml`, 그리고 어떤 `docs/ARCHITECTURE*.md`든 --
  동시에 진행 중인 패킷들이 소유한다.
- `crates/es-physics-backend/**`, `crates/es-physics-core/**`,
  `crates/es-assets/**` -- 이 패킷은 이들의 공개 API
  (`mapping_report`, `compare_backends`, `BackendKind`, `MuJoCoCpuBackend`,
  `MjWarpBackend`, `parse_mjcf`, `urdf::parse_urdf`)만 소비한다.
- `clap`이나 다른 CLI-인자-파싱 crate; `--ctrl-random`을 위한 `rand`나
  다른 RNG crate(인라인으로 시드된 splitmix64 하나면 충분하다).
- `HashMap`/`HashSet` -- 저장소 관례에 따라 `BTreeMap`/`BTreeSet`이나
  작은 `Vec`만.
- 커밋하는 것.
