<!-- docs/api-notes/isaac-sim.md의 한국어 번역. 영어 파일이 원본이며, 바뀌면 이 파일도 갱신한다. -->

# Isaac Sim — 설치, 헤드리스 스크립팅, MJCF/URDF 임포트, PhysX 결정성

**M11 X2**(`docs/packets/M11/X2-adapter-v2.md`)와 M8 리뷰가 남긴 미해결 항목 R8("Isaac Lab을
두 번째 소스로 — `es-usd` 씬, 동일한 어댑터 모양", `docs/reviews/M8.md`)을 위한 고정 다이제스트.
이 워크스페이스의 어떤 크레이트도 아직 Isaac Sim에 의존하지 않는다: 어떤 크레이트도 이를
임포트하지 않고, 어떤 venv도 이를 설치하지 않는다. 이 문서는 웹 리서치 준비 자료이며,
`docs/api-notes/mujoco.md`처럼 우리가 직접 검증한 오라클이 아니다 — 아래의 모든 주장은
`verified (fetched)`(2026-09-23에 인용된 페이지에서 읽음) 또는 `unverified`(1차 출처에서
찾지 못했거나, 2차/커뮤니티 출처에서만 찾음)로 태그되어 있다. 영어 원본: `isaac-sim.md`.

2026-09-23에 가져온 출처:

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html>
- <https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.0.0/importer_exporter/ext_isaacsim_asset_importer_mjcf.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/namespace_isaacsim__asset__importer__mjcf.html>
- <https://docs.isaacsim.omniverse.nvidia.com/5.0.0/core_api_tutorials/tutorial_core_hello_world.html>
- <https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html>
- <https://github.com/isaac-sim/IsaacSim>, <https://github.com/isaac-sim/IsaacLab/releases>

## 1. 고정 버전, 설치, EULA

| | | 상태 |
|---|---|---|
| 고정 버전 | **5.1.0** (2026-01-28 출시) | verified (fetched), pip_installation.html과 install_python.html 모두 `==5.1.0`을 대상으로 함 |
| 더 최신 릴리스 | Isaac Sim 6.0 GA(2026, 정확한 날짜는 가져오지 않음)와 "Isaac Sim 6.1을 위해 빌드된" Isaac Lab **v3.0.0-EA**(2026-09-16), GA는 2026년 10월 말 목표 | Isaac Lab 릴리스 노트 이상은 unverified; 짝을 이루는 Isaac Lab 릴리스 자체가 최신 안정판이 아니라 Early Access이므로 여기서는 **고정하지 않는다** |
| Python | **3.11** (Isaac Sim 5.x). 3.10/3.12는 pip 해석 실패("could not find a version that satisfies isaacsim") | verified (fetched) |
| pip 설치 | `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com` | verified (fetched), Isaac Lab과 Isaac Sim 설치 페이지가 일치 |
| GLIBC | **2.35+** (`manylinux_2_35_x86_64` 휠 태그); Ubuntu 20.04의 GLIBC 2.31로는 pip 설치 불가, 바이너리 설치만 가능 | verified (fetched) |
| Ubuntu | 22.04 / 24.04 | verified (fetched, requirements.html) |
| NVIDIA 드라이버 | Linux **580.65.06+**, Windows **580.88+** (최소 등급; "good"/"ideal" 등급도 같은 숫자) | verified (fetched, requirements.html) |
| GPU | 최소 등급은 GeForce **RTX 4080**를 명시; "ideal"은 RTX 5080 / RTX PRO 6000 Blackwell을 명시; **RT 코어가 없는 GPU(A100, H100)는 명시적으로 미지원** | verified (fetched); RTX 4090은 등급 표에 이름이 없음 |
| RAM / VRAM | 최소 32GB / 16GB, ideal 64GB / 48GB | verified (fetched) |
| 디스크 | 최소 50GB SSD, "ideal"은 최대 1TB NVMe | verified (fetched) |
| 헤드리스, 디스플레이 없음 | `SimulationApp({"headless": True})`가 문서화된 standalone 진입점(§2)이며, Isaac Lab 자체의 모든 CI/클러스터 워크플로가 쓰는 `train.py --headless`가 이 방식을 쓴다 | verified (fetched, Isaac Lab 문서 전반에서 쓰이는 패턴); requirements 페이지 자체에서 "디스플레이가 전혀 없어도 동작한다"는 직접적인 문장은 찾지 못함 |
| EULA | 환경 변수 **`OMNI_KIT_ACCEPT_EULA`**, 값 `YES`/`Y`/`1`(대소문자 무관), `import isaacsim` 전에 설정; 설정하지 않으면 첫 실행 시 런타임 프롬프트가 막는다. 라이선스: **NVIDIA Omniverse License Agreement**, <https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html> | verified (fetched) |

**우리 GPU 서버**(`docs/api-notes/gpu-server.md` 급의 박스, RTX 4090 24GB) 기준: VRAM(24GB)은
최소 16GB를 넘고, 세대 기준으로 RTX 4080(최소)과 RTX 5080(good) 소비자 등급 사이에 위치한다.
4090은 A100/H100처럼(RT 코어 없음) 배제된 것이 아니라 단지 NVIDIA의 이름 붙은 등급 표에 없을
뿐이다. 숫자상의 하한을 만족하는 것과 별개로 *검증된(validated)* 하드웨어인지는
**unverified**다 — 벤더의 목록은 호환성 타겟이지 전수 허용 목록이 아니며, RTX 4090 사용에 대한
커뮤니티 보고는 흔하지만 여기서 인용하는 1차 출처는 아니다.

컨테이너 설치도 문서화된, 헤드리스에 친화적인 또 다른 경로다: `docker run … -e
"ACCEPT_EULA=Y" …` (verified, 검색으로 링크된 컨테이너 설치 페이지에서 가져왔으나 이 노트에서
별도로 재확인하지는 않음 — `pip_installation.html`의 상호 참조에서 가져온 것).

## 2. 최소 헤드리스 standalone 스크립트

**verified (fetched)**, Isaac Sim 5.0/5.1 Hello-World 튜토리얼 + Core API Articulation 문서.
아래 모듈명은 **현재** 이름이다; 4.0 이전 이름(`omni.isaac.core`,
`omni.isaac.core.articulations.Articulation`)은 `isaacsim.*` 네임스페이스로 개명되었고,
5.x에서는 사용이 권장되지 않는 별칭일 뿐 주 API가 아니다.

```python
# 1. SimulationApp은 다른 어떤 isaacsim/omni import보다도 먼저 존재해야 한다.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

world = World()                       # USD 스테이지 + 물리 컨텍스트를 감싼다
world.scene.add_default_ground_plane()

# ... 스테이지에 로봇 프림을 추가 (에셋 참조, 또는 MJCF/URDF 임포터, §3) ...
# from isaacsim.core.prims import Articulation
# robot = world.scene.add(Articulation(prim_paths_expr="/World/Robot", name="robot"))

world.reset()                         # 첫 step 전에 필요; scene에 추가된 프림들을 초기화

for _ in range(500):
    action = ArticulationAction(
        joint_positions=np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]),
    )
    # robot.apply_action(action)      # 토크 제어라면 joint_efforts=...
    world.step(render=False)          # 물리 틱; render=False는 렌더 패스를 건너뜀

# joint_pos = robot.get_joint_positions()
# joint_vel = robot.get_joint_velocities()

world.reset()                         # 씬의 초기 상태로 되돌림
simulation_app.close()
```

비고, 별도 표시가 없으면 모두 **verified (fetched)**:

- `World()`는 `SimulationContext` + `Scene` 위의 싱글턴 래퍼다; `world.scene.add(...)`로
  프림 래퍼를 등록하면 `world.reset()`이 이를 초기화한다.
- `ArticulationAction(joint_positions=…, joint_velocities=…, joint_efforts=…,
  joint_indices=…)` — `joint_indices`를 넘기면 그 부분집합만 대상; 넘기지 않으면 배열은
  articulation의 전체 DOF 개수와 **순서**를 맞춰야 한다(§3의 순서 문제가 여기도 적용된다:
  이것이 USD 저작 순서인지 MJCF/URDF 소스 순서인지 임포트 이후에는 unverified — 로봇마다
  임포터가 정하는 것으로 취급하고 에셋별로 확인할 것).
- `isaacsim.core.api.World`와, MJCF 임포터 자체 예제(`stage_utils.open_stage`)에서 보이는
  더 새로운 `isaacsim.core.experimental.*` 네임스페이스: 5.1은 둘 다 제공한다; `core.api`는
  튜토리얼에 문서화된 쪽이자 위에서 쓴 쪽이고, `core.experimental`은 더 새로운 것으로, 6.x에서
  어느 쪽이 어느 쪽을 대체할 예정인지는 **unverified**.

## 3. MJCF 임포터

익스텐션 **`isaacsim.asset.importer.mjcf`**(4.0 이전 이름은 `omni.isaac.mjcf_importer`).
기본적으로 활성화되어 있음; GUI에서는 `File > Import`, 스크립트로는:

```python
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

omni.usd.get_context().new_stage()
import_config = MJCFImporterConfig(mjcf_path="/path/to/model.xml")
importer = MJCFImporter(import_config)
output_usd_path = importer.import_mjcf()          # 기본적으로 소스 옆에 .usd를 씀
result, stage = stage_utils.open_stage(output_usd_path)
```

익스텐션 자체 튜토리얼 페이지에서 **verified (fetched)**.

관측된 `MJCFImporterConfig` 필드: `mjcf_path`(파일 또는 디렉터리), `robot_type`(스키마 힌트:
Default/Manipulator/Humanoid/Wheeled/…), `import_scene`(MJCF 자체의 `<option>`/조명을 시뮬레이션
설정으로 가져올지), `merge_mesh`, `collision_type`(Convex Hull / Convex Decomposition /
Bounding Sphere / Bounding Cube), `allow_self_collision`, `fix_base`, `link_density`,
`run_asset_transformer` — **verified (fetched)**, 이름만; 필드별 기본값은 개별적으로
확인하지 않음(**unverified**).

**무엇을 파싱하는가**, C++/Python 바인딩 네임스페이스의 클래스 목록에서 읽음(`verified
(fetched)`, `namespace_isaacsim__asset__importer__mjcf.html` — 클래스가 존재한다는 것은 해당
요소가 USD로 라운드트립된다는 근거이고, 없다는 것은 조용히 버려진다는 근거이지 증거는 아니다.
다른 클래스로 접혀 들어갔을 수도 있기 때문이다):

| MJCF 요소 | 임포터 클래스 | 해석 |
|---|---|---|
| `<body>`, `<inertial>` | `MJCFBody`, `MJCFInertial` | 지원 |
| `<joint>` | `MJCFJoint` | 지원, **단서 있음**: USD/PhysX 조인트는 엄격히 두 바디 간이다; 같은 부모에 여러 조인트를 가진 MuJoCo 바디(흔한 다중 DOF 패턴)는 하나의 PhysX **D6** 조인트로 축약되며, 실제 버그 리포트에 따르면 축별 조인트 메타데이터가 손실된다(`isaac-sim/IsaacLab#6854`, 이슈 제목 이상은 unverified) |
| `<geom>` | `MJCFGeom` | 지원 — STL을 통한 메시 지오메트리(`MJCFMesh`) 포함 |
| `<contact>` (`<pair>`, `<exclude>`) | `MJCFContact`, `ContactNode` | 지원 — `contype`/`conaffinity` 류의 필터링을 읽는 클래스가 바로 이것; 속성별로 PhysX 충돌 그룹에 정확히 어떻게 매핑되는지는 **unverified** |
| `<tendon>` | `MJCFTendon` | 클래스가 존재 → 어느 정도 지원; 고정(fixed) 텐던과 공간(spatial) 텐던이 둘 다 임포트되는지, 아니면 PhysX에 네이티브 대응물이 있는 고정 텐던만 되는지는 **unverified** |
| `<equality><connect>` | `MJCFEqualityConnect` | 클래스가 존재 → `connect`(용접형) 등가 제약은 읽힘; 다른 등가 타입(`joint`, `tendon`, `distance`, 비항등 오프셋을 가진 `weld`)은 **unverified**이며, 2023년경 포럼 보고에 따르면 MuJoCo/Isaac Gym 시절의 equality-connect 지원에 버그가 있었다(이 익스텐션에서 수정되었는지는 unverified) |
| `<actuator>` | `MJCFActuator` | 지원 — 클래스는 존재하지만, **`position`/`velocity`/`motor`/`general` 액추에이터 타입이 서로 다른 PhysX 조인트 드라이브 stiffness/damping으로 매핑되는지, 아니면 모두 한 종류의 드라이브로 평탄화되는지는 unverified**; 에셋별로 확인하기 전까지는 모든 액추에이터를 "어떤 조인트 드라이브가 붙었다" 정도로만 취급할 것 |
| `<site>` | `MJCFSite` | 지원 (보통 마커용 Xform이 됨) |
| `<material>`, `<texture>` | `MJCFMaterial`, `MJCFTexture` | 지원 |
| `<sensor>` | 네임스페이스에 `MJCFSensor` 클래스 없음 | **해석: 임포트되지 않음.** 가져온 네임스페이스 목록에 센서 관련 클래스가 전혀 없다; MJCF 센서(framepos, jointpos, touch, …)는 이 익스텐션이 쓰는 USD 대응물이 없을 가능성이 크다. 명시적인 부정으로서는 unverified — 클래스 부재는 시사적일 뿐 문서화된 제외 목록은 아니다 |
| `friction` (geom `friction="μ_t μ_r μ_s"`) | — | MuJoCo의 세 마찰 계수(미끄럼/비틀림/구름) 전부가 PhysX의 단일 동적+정적 마찰 쌍에 매핑되는지, 첫 번째만 쓰이는지는 **unverified** |

알려진 제한, **verified (fetched)**: "링크나 조인트 이름의 특수문자는 지원되지 않으며
밑줄로 치환된다; 그 결과 이름이 밑줄로 시작하게 되면 `a`가 앞에 붙는다" (USD 프림 이름
유효성 규칙).

알려진 제한, **verified (fetched, GitHub 이슈 제목)**: 다중 DOF 조인트가 하나의 D6로
축약되면서, 원래의 축별 의미(드라이브 게인, 리밋)를 복원하려면 하류 리더가 필요로 할 축별
메타데이터가 손실된다.

## 4. URDF 임포터

익스텐션 **`isaacsim.asset.importer.urdf`**, 형태는 유사하다: `URDFImporterConfig` /
`URDFImporter.import_urdf()`. 문서에 이름이 언급된 설정 필드: `urdf_path`, `usd_path`,
`collision_from_visuals`, `merge_mesh`, 그리고 각 조인트의 드라이브 타입(Position/Velocity)과
강도(stiffness)를 명시적으로 설정하는 조인트 드라이브 후처리 단계 — **verified (fetched)**,
이름만; 이 노트는 더 깊이 들어가지 않는다. URDF는 M11 X2에서 MJCF만큼 중요하지 않기 때문이다
(해당 패킷의 "두 번째 소스"는 Isaac Lab **정책**이고, Isaac Lab 자체의 로봇들은 이미 USD이므로
임포터는 주요 경로가 아니라 대체 경로다 — M11 X2가 URDF 임포트를 전혀 필요로 하지 않을 수도
있으며 이는 unverified).

## 5. PhysX 결정성

**verified (fetched)**, `IsaacLab/…/features/reproducibility.html`:

- 동일한 하드웨어와 동일한 Isaac Sim/PhysX 버전이 주어지면, 강체(rigid-body)와 articulation
  씬은 실행마다 비트 단위로 재현된다. **하드웨어를 넘나드는** 재현성은 보장되지 **않는다** —
  GPU/CPU가 다르면 부동소수점 반올림 경로가 다르다.
- 비강체(천, 소프트바디, deformable)에 대해서는 PhysX가 어떤 결정성 보장도 제공하지 **않는다**.
- `enable_enhanced_determinism`: `SimulationCfg`/`PhysxCfg` 플래그, **기본값 `False`**,
  "성능을 희생해 향상된 결정성을 켜고 끈다" — 정확한 PhysX 레벨 메커니즘(추가 동기화 배리어인지,
  다른 솔버 코드 경로인지)은 **unverified**.
- GPU 작업 스케줄링 자체가 한 카드 안에서도 결정성 리스크다: "런타임에 시뮬레이션 파라미터를
  바꾸면 연산이 일어나는 순서가 달라질 수 있다"고 문서는 말하며, 최하위 비트 드리프트가
  누적될 수 있다 — 문서화된 완화책은 도메인 랜덤화를 GPU 파이프라인에서 에피소드 중간이 아니라
  스텝을 시작하기 전 설정 시점에만 적용하는 것이다.
- **솔버**: PhysX 5의 기본 `solver_type`은 **TGS**(Temporal Gauss-Seidel, 값 `1`)이며,
  대안은 **PGS**(Projected Gauss-Seidel, 값 `0`)다. 반복 횟수 필드(`min_position_iteration_count`
  및 그 속도/GPU 버퍼 크기 대응물)가 `PhysxCfg`에 존재하지만, `solver_type` 외의 정확한 필드명/
  기본값은 개별적으로 확인하지 않았다 — **unverified**, 특정 기본값에 의존하기 전에
  `isaaclab.sim.simulation_cfg` 소스를 직접 확인할 것.
- **GPU vs CPU 파이프라인**: PhysX 5의 GPU 가속 시뮬레이션이 기본값이며 `SimulationCfg`에서
  `device="cpu"`로 강제로 끌 수 있다; GPU 모드는 미리 크기를 정한 버퍼가 필요하다(동적으로
  키울 수 없다) — 결정성 문제와는 별개의 운영상 함정이다 — **verified (fetched)**.
- **dt / substeps**: Isaac Lab은 명시적 `substeps` 필드를 `sim.dt` + `decimation`으로
  대체했다: 예를 들어 `dt = 1/60`에 2단계 decimation이면 "물리를 `dt = 1/120`로 제어 틱당
  두 번 스텝하는 것과 동등하다" — **verified (fetched)**; reach 태스크의 구체적인 수치는
  `docs/api-notes/isaac-lab.md` §6 참고.

## 6. 아직 모르는 것

- MJCF로 임포트한 씬을 TGS를 쓰는 GPU PhysX에서 돌렸을 때, 어떤 백엔드 간 비교라도 성립할
  만큼 **MuJoCo CPU** 수치에 근접하게 재현하는지 — 이 노트를 위해 오라클을 돌리지는 않았다
  (이것은 측정 결과인 `docs/api-notes/mujoco.md`의 메시 절과 달리 리서치 다이제스트다).
- 정확한 액추에이터 타입→드라이브 매핑(§3)과 정확한 마찰 계수 매핑 — "두 번째 소스" 씬을
  Isaac Lab의 네이티브 USD 에셋을 직접 읽는 대신 MJCF 임포트로 만들 경우 M11 X2의 어댑터에
  둘 다 중요하다.
- Isaac Sim 6.0/6.1과 Isaac Lab 3.0의 멀티백엔드(Warp/Newton) 아키텍처는 릴리스 노트
  헤드라인에서만 보였다; 이 노트 작성 시점에 3.0이 Early Access이고 이 프로젝트 자체의 Newton
  백엔드(`docs/api-notes/newton.md`)가 어느 쪽이든 더 관련성 높은 참고 자료이므로, API 표면은
  아무것도 가져오지 않았다.
