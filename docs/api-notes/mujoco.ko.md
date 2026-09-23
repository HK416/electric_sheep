<!-- Korean translation of docs/api-notes/mujoco.md. The English file is the working copy; regenerate this when it changes. -->

# `mujoco` (Python), 3.13.0에 고정

`MuJoCoCpuBackend`(spec 17.1)의 참조 프로세스인
`crates/es-physics-backend/python/mujoco_ref.py`가 사용하는 표면. CPython 3.12 /
3.13 (Windows x86-64)에서 `mujoco==3.13.0` 휠에 대해, 크레이트 자체의 오라클
테스트를 실행해 확인했다. 워크스페이스의 고정 의존성은 아니다 — 패키지는
선택적이며, `MuJoCoCpuBackend::is_available()`이 부재를 보고하면 CI는 실패하는
대신 오라클을 건너뛴다 (spec 2.4 — 런타임 경로의 어떤 것도 Python을 필요로 하지
않는다).

3.x 라인이면 무엇이든 동작해야 한다. CI 이미지가 설치하는 버전이 바뀌면 이 파일을
갱신한다.

## Calls used

| Call | 사용된 시그니처 | 비고 |
|---|---|---|
| `mujoco.MjModel.from_xml_string(xml)` | `(str) -> MjModel` | 모델이 잘못되면 `ValueError`를 발생시키며, 메시지는 `PhysicsError::Backend`로 그대로 전달된다. |
| `mujoco.MjData(model)` | `(MjModel) -> MjData` | 환경당 하나씩; `n_envs`는 이들의 리스트를 들고 있는 방식으로 에뮬레이션된다. |
| `mujoco.mj_step(model, data)` | `(MjModel, MjData) -> None` | 물리 틱 하나. `step(n)`당 `n`번 호출된다. |
| `mujoco.mj_forward(model, data)` | `(MjModel, MjData) -> None` | 리셋이나 상태 쓰기 뒤에 호출해, `sensordata` / `xpos` / `xquat`이 `qpos`와 맞도록 한다. |
| `mujoco.mj_resetData(model, data)` | `(MjModel, MjData) -> None` | 모델의 초기 상태로 되돌린다. |
| `mujoco.mj_id2name(model, objtype, id)` | `(MjModel, mjtObj, int) -> str \| None` | 이름 없는 요소면 `None`; emitter는 항상 이름을 쓰므로 여기서 `None`이 나오면 버그이고 `PhysicsError::Protocol`로 드러난다. |
| `mujoco.mjtObj.mjOBJ_JOINT` / `mjOBJ_ACTUATOR` / `mjOBJ_SENSOR` / `mjOBJ_BODY` | enum | 이 네 네임스페이스만 조회한다. |

## Fields read

`MjData`: `qpos`, `qvel`, `act`, `ctrl`, `sensordata`, `xpos`, `xquat` — 모두
`float64`의 `numpy` 배열. `xpos`는 `(nbody, 3)`, `xquat`는 `(nbody, 4)`이며,
**`xquat`는 wxyz**이므로 wire를 건너기 전에 spec 3.1의 xyzw로 재정렬한다.

`MjModel`: `nq`, `nv`, `nu`, `nsensordata`, `nbody`, `njnt`, `nsensor`, `jnt_type`,
`jnt_qposadr`, `jnt_dofadr`, `sensor_adr`, `sensor_dim`, `opt.timestep`.

`jnt_type`은 `mjtJoint`이다: `0 = free` (qpos 7 / dof 6), `1 = ball` (4 / 3), `2 =
slide` (1 / 1), `3 = hinge` (1 / 1). 스크립트는 주소 차이로부터 너비를 유도하는
대신 이 표를 직접 들고 있는다.

## Behaviour worth pinning

- **센서 노이즈는 기본적으로 꺼져 있다.** 센서의 `noise`와 `cutoff`는 MJCF에
  쓰이지만, `MuJoCo`는 `sensornoise` enable 플래그가 있을 때만 노이즈를 적용하며,
  이 어댑터는 그 플래그를 설정하지 않는다. 백엔드 capabilities(spec 17.2)에
  `BackendQuirk`로 선언되어 있다.
- **`autolimits`는 3.x에서 기본값이 true**이므로, `limited` 속성이 없는 joint
  `range`는 제한된 것으로 취급된다. Emitter는 이를 그대로 활용해 `limited`를 쓰지
  않는다.
- **`fullinertia`는 관성 프레임을 덮어쓴다:** `MuJoCo`가 이를 고유분해
  (eigendecompose)해 프레임을 그 결과로 설정하므로, `quat`과 `fullinertia`는
  함께 쓸 수 없다. Emitter는 관성이 대각(diagonal)이면 `quat` + `diaginertia`를,
  프레임이 항등(identity)이면 `fullinertia`를 쓰고, 나머지 경우는 회전을 잃는 대신
  이름을 명시해 거부한다.
- 평면(plane)의 세 번째 `size` 성분은 렌더링 격자 간격이며 양수여야 한다.

## Meshes (packet M10/W2a)

출처: MuJoCo 3.13 [XML reference, `asset/mesh`](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh),
[STL (file format) — Wikipedia](https://en.wikipedia.org/wiki/STL_(file_format)),
[fabbers.com STL format](https://www.fabbers.com/tech/STL_Format),
[Wavefront .obj file — Wikipedia](https://en.wikipedia.org/wiki/Wavefront_.obj_file).
`target/plan-w/w2a-research.md`로 수집했고, 여기서는 실제 파일에 대해 다시 확인했다.

### Binary STL

`80바이트 헤더 ‖ u32 LE 삼각형 개수 ‖ n × 50바이트`이며, 각 facet은
`f32 3개 법선 ‖ f32 3×3 정점 ‖ u16 attribute 개수`로 전부 리틀엔디언이다. 따라서
바이너리 파일은 정확히 `84 + 50·n` 바이트이고, `es_assets::stl::parse`는 이것을
판별 기준으로 쓴다. 스펙은 헤더가 "`solid`로 시작해서는 안 된다"고 말하지만 이는
내보내기 도구가 깨뜨리는 관례일 뿐이므로, 첫 단어만 보는 판별은 실제 파일을
잘못 읽는다. ASCII는 기본값이 아니라 대체 경로다.

저장된 법선은 참고용이며 흔히 `0 0 0`이다. 진실은 감김 순서(winding)이고, 바깥에서
볼 때 반시계 방향이다. 리더는 법선을 보관하지 않고 — 렌더러가 유도한다 — 대신
0이 아닌 저장 법선이 `(b−a)×(c−a)`와 어긋나면 facet의 두 번째와 세 번째 정점을
맞바꾼다.

### ASCII STL

`solid [name]` / `facet normal nx ny nz` / `outer loop` / `vertex x y z` 세 개 /
`endloop` / `endfacet` / `endsolid`. 공백은 자유 형식이다. 리더는 `normal`과
`vertex` 토큰만 보고 나머지는 무시하는데, 문법에는 충분하며 레이아웃 변형에 강하다.

### OBJ

`v x y z` (네 번째 숫자는 작성 도구에 따라 `w`이거나 색이며, 여기서는 둘 다 무시)와
네 가지 형태의 `f` — `a`, `a/vt`, `a//vn`, `a/vt/vn`. 인덱스는 1부터 시작하고, 음수는
최종 개수가 아니라 **파일에서 지금까지 본** 정점 수에서 거꾸로 센다. 면은 정점이
셋보다 많을 수 있으며 첫 정점을 중심으로 팬 삼각분할한다. `vt`, `vn`, `o`, `g`, `s`,
`mtllib`, `usemtl`과 주석은 무시한다 — 재질 없는 메시도 올바른 형상이다.

### `<asset><mesh>`

| 속성 | 기본값 | 이 파이프라인 |
|---|---|---|
| `file` | — | Importer가 `AssetRef.path`로 읽는다. **내보내지 않는다** — `scene_to_mjcf`는 `vertex`/`face`를 인라인으로 쓴다. `PhysicsBackend::load`는 `&SceneDesc`만 받고 경로 통로가 없으며 (INV-17), 해시되는 구조체 안의 기계 경로는 spec 5.3이 금지하는 바로 그것이다. |
| `vertex` / `face` | — | Emitter가 쓰는 것. `face`는 0-based, 반시계, `0..nvert-1`. `face`를 아예 빼면 MuJoCo가 점들의 볼록 껍질로 메시를 만든다. Emitter는 항상 쓴다. |
| `scale` | `1 1 1` | 기본값이 아니면 `MjcfError::Unsupported`로 이름을 명시해 거부한다. 굽기(baking)는 후속 항목으로 명시했다. |
| `refpos` / `refquat` | `0 0 0` / `1 0 0 0` | 마찬가지로 이름을 명시해 거부한다. 둘 다 MuJoCo가 보기 전에 정점을 변환한다. |
| `inertia` | **`legacy`** | 기본값 그대로 둔다. 아래 참조. |
| `maxhullvert` | `-1` (무제한) | 표현하지 않으며 `report_unknown`이 경고한다. `so101.xml`은 모델 전역 `128`, 그리퍼 메시 셋에 `64`를 설정하므로, 내보낸 SO-101은 상류가 제한하는 자리에서 qhull을 무제한으로 돌린다. |
| `normal` | — | STL 메시에는 아예 줄 수 없다. MuJoCo가 STL의 법선을 직접 생성한다. |
| `texcoord` | — | STL에서는 나오지 않는다. 텍스처는 M7 R6. |

**충돌은 `inertia`와 무관하게 언제나 메시의 볼록 껍질을 쓴다.** 렌더러는 표면을
그리므로 오목한 메시에서는 그려지는 것과 충돌하는 것이 다르다. 이는 암묵적으로
두는 대신 `Feature::ContactMesh`의 `BackendQuirk`로 선언한다.

### `inertia="legacy"` 대 `"exact"`, 측정값

문서는 `legacy`가 *비볼록* 메시의 부피를 과다 계산하므로 "권장하지 않는다"고 하며,
하위 호환을 위해서만 기본값으로 유지한다. 볼록하고 닫힌 메시에는 과다 계산할 것이
없고, `mesh_box::legacy_and_exact_inertia_agree_on_a_convex_closed_box`가 이를
확인한다. ±0.05 m 상자는 둘 다에서 **비트 단위로 동일한** `body_mass`와
`body_inertia`로 컴파일된다. 여기서 오차의 원인은 알고리즘이 아니다.

*실제* 원인은 `f32`다. 메시 정점은 파일에서도, `MeshData`에서도, 콘텐츠 해시에서도
`f32`인 반면 프리미티브의 `size`는 `f64`다. `0.05`에 가장 가까운 `f32`는
`0.05000000074505806`으로 상대 1.49e-8이고 부피는 그것이 셋이므로, 메시 상자는
동일한 프리미티브 상자보다 4.47e-8 무겁다 (측정: 1.0000000447034845 kg 대
1.0000000000000002 kg). 대각 관성은 7.45e-8 어긋난다. 그럼에도 두 물체는 1 kHz로
2,000 스텝 뒤 7.5e-10 m 차이로 0.049892 m에 정지한다. 메시 대 프리미티브 질량
비교에서 ~1e-7보다 빡빡한 허용오차는 `f64` 정점을 요구하는 것인데, STL도
`MeshData`도 그것을 담지 않는다.

**두 플랫폼, 같은 비트 (2026-09-23).** `mesh_box` 오라클 네 개, `mesh_load`, 그리고 `--ignored`인
`so101_provenance`, `panda_provenance`, `menagerie_meshes`를 오라클 서버(Linux x86-64,
`~/venvs/es-lerobot-cuda`의 MuJoCo 3.13)에서도 돌렸다: 모두 통과했고, 출력된 모든 수 — 두 질량, 관성
여섯 개, 두 정지 높이(`0.0498922453248334` / `0.04989224457977534`) — 가 Windows 실행과 자릿수까지
동일하다.

### MuJoCo Menagerie

저장소 전체 라이선스는 없다. 루트 `LICENSE`가 로봇마다 한 블록씩 이어 붙인 것이고
GitHub은 `NOASSERTION`으로 보고한다. `franka_emika_panda`와 `robotstudio_so101`은
Apache-2.0, `universal_robots_ur5e`는 ROS Industrial Consortium의 3-clause BSD다.
아무것도 벤더링하지 않는다 — `tests/fixtures/mjcf/*.PROVENANCE.json`이 커밋과 각
파일의 blake3를 고정하고, 테스트는 `meshdir`가 해석되도록 상류 디렉터리 구조 그대로
`target/menagerie/<commit>/`로 받아온다. Raw URL은
`https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/<commit>/<path>`.

| 모델 | 커밋 | 파일 | 측정값 |
|---|---|---|---|
| `robotstudio_so101` | `ac6b2b09983786f3036cab1000221017fa2193b4` | 바이너리 STL 19개 (17.2 MB); `so101.xml`은 그중 18개를 선언 | 삼각형 362,996개. 모든 물체의 질량이 상류를 직접 적재한 것과 비트 단위로 같다 — 메시에서 유도되는 `camera_mount` 0.012 kg 포함. 인라인 MJCF 11 MB |
| `franka_emika_panda` | `822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7` | `panda.xml` + 충돌 STL 8개 + 시각 OBJ 59개 | 삼각형 136,590개. 리더와 해시뿐이다. `panda.xml`에는 손가락 관절을 묶는 `<tendon><fixed>`가 있고 `scene_to_mjcf`는 tendon을 이름을 명시해 거부한다. `link5`에는 STL이 없다 — 충돌 형상이 `link5_collision_*.obj` 셋이다. |

OBJ 로봇을 MuJoCo로 적재하고 싶어지면 `universal_robots_ur5e`가 OBJ 전용 후보다:
OBJ 19개, `<tendon>` 없음, `<equality>` 없음. 두 집합의 어떤 Menagerie 모델도
기본값이 아닌 `<mesh scale>`을 쓰지 않는다.

**명시한 후속 항목:** `<mesh scale>` 굽기; 인라인 텍스트가 너무 느려지면 `file=`
경로 통로; glTF의 `MeshData`를 같은 방식으로 `scene.meshes`에 싣기; 오늘 미검증인
MJWarp의 `Feature::ContactMesh`; 텍스처와 재질 (M7 R6).

## Protocol

`python -c "<embedded script>"`, 양방향으로 줄당 JSON 객체 하나. 요청은 `cmd`를
담고, 응답은 `{"ok": true, ...}` 또는 `{"ok": false, "error": "<ExcType>:
<message>"}`다 — 모델링 오류는 프로세스가 죽는 것이 아니라 값이다. 배열은 숫자의
JSON 리스트이며 env-major다. Python float에 대한 `json.dumps`와
`float_roundtrip`을 쓴 `serde_json`은 둘 다 최단 왕복(shortest-round-trip)이므로
`f64` 값은 정확히 그대로 건너간다 — 비트 단위 run-to-run 테스트가 이에 의존한다.

| Request | Reply |
|---|---|
| `{"cmd":"load","mjcf":str,"n_envs":int,"timestep":float\|null,"seed":int}` | `nq, nv, nu, nsensordata, nbody, joints[{name,qpos:[adr,dim],dof:[adr,dim]}], actuators[name], sensors[{name,adr,dim}], bodies[name]` |
| `{"cmd":"reset","envs":[int]\|null,"state":{...}\|null}` | `{}` |
| `{"cmd":"set_ctrl","ctrl":[float]}` | `{}` |
| `{"cmd":"step","n":int}` | `{"nonfinite":[env]}` |
| `{"cmd":"state"}` | `qpos, qvel, act, sensordata, xpos, xquat` |
| `{"cmd":"set_state","state":{"qpos":[],"qvel":[],"act":[]}}` | `{}` |
| `{"cmd":"quit"}` | *(응답 없음; 프로세스가 종료된다)* |

`ES_PYTHON`이 인터프리터를 선택한다; 설정되지 않으면 `python`, 그다음 `python3`을
시도한다.
