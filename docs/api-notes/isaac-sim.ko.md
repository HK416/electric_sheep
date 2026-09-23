<!-- docs/api-notes/isaac-sim.md의 한국어 번역. 영어 파일이 원본이며, 바뀌면 이 파일도 갱신한다. -->

# Isaac Sim — 설치, 헤드리스 스크립팅, MJCF/URDF 임포트, PhysX 결정성

**M11 X2**(`docs/packets/M11/X2-adapter-v2.md`)와 M8 리뷰가 남긴 미해결 항목 R8("Isaac Lab을
두 번째 소스로 — `es-usd` 씬, 동일한 어댑터 모양", `docs/reviews/M8.md`)을 위한 고정 다이제스트.
어떤 크레이트도 Isaac Sim을 임포트하지 않는다. M11 I0(2026-09-23)부터 서버의 venv 하나
(`~/venvs/es-isaac`)가 이를 설치하고 `python/physx_smoke.py`가 이를 구동한다. §1–§6은 웹
리서치 준비 자료다 — 거기의 모든 주장은 `verified (fetched)`(2026-09-23에 인용된 페이지에서
읽음) 또는 `unverified`(1차 출처에서 찾지 못했거나, 2차/커뮤니티 출처에서만 찾음)로 태그되어
있다. **§7은 우리가 직접 측정한 것**이며(M11 I0, `docs/packets/M11/I0-isaac-sim-install.md`),
§1–§6과 어긋나는 곳에서는 §7이 우선한다. **§8은 그 위에 만든 `PhysXBackend`**(M11 I1)이며,
이것도 측정한 것이다. 영어 원본: `isaac-sim.md`.

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

익스텐션 자체 튜토리얼 페이지에서 **verified (fetched)**. **측정으로 바로잡음(§7):** 5.1.0
휠의 익스텐션(`isaacsim.asset.importer.mjcf` 2.5.13)에는 `MJCFImporter` /
`MJCFImporterConfig`가 없다. 파이썬 API는 Kit 명령 두 개, `MJCFCreateImportConfig`와
`MJCFCreateAsset(mjcf_path, import_config, prim_path, dest_path="")`이다(익스텐션 자체의
`docs/api.rst`와 테스트가 이를 쓴다). 위의 클래스 이름은 이후 릴리스의 것이다.

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

- ~~MJCF로 임포트한 씬이 MuJoCo CPU 수치를 재현하는지~~ — §7에서 측정했다: 재현하지 않는다.
  SO-101에서 100스텝 동안 최대 1.73 rad 차이가 나며, 이를 설명하는 임포터 행을 §7이 적는다.
  I1의 수리(§8)를 거치면 같은 제어에서 500스텝 동안 0.30 rad 차이다.
- 정확한 액추에이터 타입→드라이브 매핑(§3)과 정확한 마찰 계수 매핑 — `<position>`
  액추에이터와 지오메트리 `friction`은 §7.3에서 측정했다. `velocity`/`motor`/`general`
  액추에이터와 텐던/등식 제약은 여전히 측정하지 않았다(그것을 가진 픽스처가 없다).
- Isaac Sim 6.0/6.1과 Isaac Lab 3.0의 멀티백엔드(Warp/Newton) 아키텍처는 릴리스 노트
  헤드라인에서만 보였다; 이 노트 작성 시점에 3.0이 Early Access이고 이 프로젝트 자체의 Newton
  백엔드(`docs/api-notes/newton.md`)가 어느 쪽이든 더 관련성 높은 참고 자료이므로, API 표면은
  아무것도 가져오지 않았다.

## 7. 오라클 서버에서 측정한 것 (M11 I0, 2026-09-23)

이 절의 모든 것은 우리가 직접 돌렸다. 산출물(로그, `pip freeze`, 스테이지 리포트, 모든
궤적, `run_all.sh`)은 RTX 4090 서버의 `~/artifacts/plan-x/i0/` 아래에 있다.

### 7.1 EULA, 설치, 버전, 비용

- **EULA:** 소유자가 **2026-09-23**에 NVIDIA Omniverse License Agreement / Isaac Sim EULA를
  수락했다(I0 시작 전에 M11 오케스트레이터가 기록). 모든 실행은
  `OMNI_KIT_ACCEPT_EULA=YES`를 설정한다.
- **호스트:** Ubuntu **26.04.1**(NVIDIA의 22.04/24.04 목록에 없음), GLIBC 2.43, 커널 7.0.0,
  드라이버 **610.57.04**, RTX 4090 24 GB, Ryzen 7 7700, RAM 60 GB, sudo 없음.
- **설치**(`uv`, 시스템 패키지 없음): `uv venv --python 3.11 ~/venvs/es-isaac`(uv가 관리하는
  CPython **3.11.16**), 이어서 `uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url
  https://pypi.nvidia.com --index-strategy unsafe-best-match`, 이어서
  `"isaaclab[isaacsim,all]==2.3.2.post1"`(같은 인덱스 플래그), 이어서 `rsl-rl-lib`(이미 충족).
  모두 종료 코드 0.
- **해석된 버전:** `isaacsim 5.1.0.0`(빌드 `5.1.0-rc.19`, 2025-10-17), PhysX / `omni.physx`
  **107.3.26**, MJCF 임포터 익스텐션 **2.5.13**, `isaaclab 2.3.2.post1`, `rsl-rl-lib 3.0.1`
  (휠 메타데이터는 3.0.1로 고정하고, 번들된 `isaaclab_rl/setup.py`는 3.1.2라고 적는다), `torch
  2.7.0+cu126`, `numpy 1.26.0`, `warp-lang 1.17.0`. `pip-freeze.txt`에 패키지 205개.
- **시간:** venv 4 s, `isaacsim` 154 s, `isaaclab` 20 s(torch는 이미 `uv` 캐시에 있었음).
  헤드리스 앱 시작 ≈ 3.5 s, 100스텝 스모크 한 번 ≈ 4.2 s(전체 벽시계).
- **디스크:** venv **17 GB**, 설치 중 `uv` 캐시가 19 → 35 GB로 증가(≈ 16 GB). §7.2의 호환
  라이브러리: 49 MB.

### 7.2 Ubuntu 26.04가 깨뜨리는 것과 사용자 공간 우회책

| 증상 | 원인 | 우회책(sudo 없음, 시스템 전역 변경 없음) |
|---|---|---|
| `omni.kit.asset_converter`, `omni.kit.tool.asset_importer`, URDF **및 MJCF 임포터** 플러그인에서 `OSError: libxml2.so.2: cannot open shared object file`(어느 것도 로드되지 않음) | 26.04에는 `libxml2.so.16`만 있다 | `archive.ubuntu.com`의 Ubuntu 24.04 `.deb` `libxml2_2.9.14+dfsg-1.3ubuntu3.9`와 `libicu74_74.2-1ubuntu3.1`를 `dpkg-deb -x`로 `~/opt/isaac-compat/root`에 풀고 `LD_LIBRARY_PATH=~/opt/isaac-compat/root/usr/lib/x86_64-linux-gnu` |
| 기본 익스피리언스(`isaacsim.exp.base.python.kit`)로는 헤드리스에서도 `app ready` 직후 `librtx.scenedb.plugin.so`(`carb.scenerenderer-rtx`)에서 세그폴트 | 이 OS / 드라이버에서의 RTX 렌더러 — 더 진단하지 않음 | `SimulationApp`을 Isaac Lab의 물리 전용 익스피리언스 `isaaclab/apps/isaaclab.python.headless.kit`(씬 델리게이트 없음)로 시작한다. `isaaclab.app.AppLauncher(headless=True)`도 같은 파일을 고른다. 따라서 이 호스트에서 렌더링(카메라, `isaaclab.python.headless.rendering.kit`)은 **시험하지 않았다** |

둘 다 적용하면: Isaac Lab의 `AppLauncher`가 헤드리스로 뜨고, `isaaclab_rl.rsl_rl`과
`isaaclab_tasks`가 임포트되며(pip 휠은 이들을 별도 배포판이 아니라 소스 익스텐션으로
번들한다), `*Reach*` gym id 18개가 등록된다(`Isaac-Reach-Franka-v0` 포함, SO-101 태스크는 없음).

### 7.3 MJCF 임포터 충실도 (SO-101과 `mesh_box`, 임포트된 USD 스테이지에서 읽음)

입력: `es_physics_backend::scene_to_mjcf`가 `tests/fixtures/mjcf/so101_pick_place.xml`과
`mesh_box.xml`에 대해 내보내는 MJCF 텍스트(리포 밖의 일회용 바이너리가
`es_assets::parse_mjcf` → `es_assets::mesh::load` → `scene_to_mjcf`를 호출해 덤프), 그리고 원본
픽스처 자체. 임포트 설정: `fix_base`는 표에 적은 대로, `import_inertia_tensor = true`,
`create_physics_scene = false`(씬은 `World`가 소유), 프림 경로 `/World/robot`.
`python/physx_smoke.py --stage-report`로 읽었다.

| MJCF | 임포트 후 USD / PhysX | 판정 |
|---|---|---|
| 힌지 `range`(rad) | `PhysicsRevoluteJoint` `lowerLimit`/`upperLimit`, **도(degree)** 단위로 정확(±1.91986 rad → ±109.9999°) | 유지 |
| 조인트 순서 | SO-101에서 DOF 순서 = MJCF 순서(`shoulder_pan … gripper`). 스모크는 그래도 이름으로 맞춘다 | 유지(이 에셋) |
| 조인트 `armature` 0.028 | `physxJoint:armature` 0.028 | 유지 |
| 조인트 `damping` 0.6 | **드라이브** 감쇠(0.6)와 `physxLimit:X:damping`(0.6)이 된다. 아티큘레이션에는 수동 조인트 감쇠가 없다 | **변경** — 액추에이터 `forcerange` 바깥이 아니라 드라이브 `maxForce` 클램프 안쪽 |
| 조인트 `frictionloss` 0.052 | `physxJoint:jointFriction` 0.0 | **누락** |
| `<position kp=998.22>` | 드라이브 `type = force`, `stiffness = 998.22` 그대로. USD 각 드라이브 게인은 **도당** 단위 | **변경** — rad→deg 변환 없음(USD 단위가 맞다면 N·m/rad로 57.3배 뻣뻣함. 여기서는 어차피 드라이브가 포화한다) |
| `<position kv=2.731>` | 어디에도 없음(드라이브 감쇠는 조인트 감쇠) | **누락** |
| `forcerange ±2.94` | 드라이브 `maxForce` 2.94 | 유지 |
| `ctrlrange` | 쓰지 않음 | 누락(호출자가 목표를 클램프해야 함) |
| 바디 `mass`, `<inertial fullinertia>` | `physics:mass` 정확, `diagonalInertia` = MuJoCo의 주관성 모멘트(순서 바뀜) + `principalAxes`, `centerOfMass` 정확 | 유지 |
| 충돌하지 않는 지오메트리의 `mass=`(`camera_mount_shell mass=0.012`, `contype=0`) | 바디 `physics:mass` 0(PhysX가 기본 밀도로 콜라이더에서 유도) | **누락** |
| 자유 조인트, `fix_base = 0` | 바디가 자기 `ArticulationRootAPI`(DOF 0개인 아티큘레이션)를 받고 올바르게 떨어진다 | 유지 |
| 자유 조인트, `fix_base = 1` | 월드로의 `PhysicsFixedJoint rootJoint_<body>`: `fix_base`가 **모든** 루트에 적용되어 자유 큐브가 용접된다 | **변경** — 스모크가 용접을 지운다 |
| 조인트 없는 루트 바디(`base`) | `fix_base = 1`일 때만 용접, 0이면 팔이 자유 | 호출자의 선택. MJCF 의미론은 1이 필요 |
| `<worldbody>` 지오메트리(테이블, 통) | `/World/robot/worldBody` 아래 지오메트리마다 키네마틱 `RigidBodyAPI` Xform 하나. 그런데 `worldBody` 자체가 강체 **없이** `ArticulationRootAPI`를 가진다 → `World.reset()` 실패(`'NoneType' object has no attribute 'is_homogeneous'`) | **고장** — 스모크가 그 API를 제거 |
| `plane` | `Plane` 콜라이더 | 유지 |
| `box` / `sphere` / `capsule` | `Cube` / `Sphere` / `Capsule` 콜라이더 | 유지 |
| 파일에서 온 메시 지오메트리(STL) | `Mesh` 콜라이더, `physics:approximation = convexHull` | 유지(볼록 껍질로) |
| 인라인 메시 `vertex=`/`face=`(`scene_to_mjcf`가 내보내는 것) | 임포터가 메시 이름의 파일을 찾고, `Unsupported Format (/meshes/box)`를 로그한 뒤 `[Fatal] attempted member lookup on NULL TfRefPtr<UsdStage>`, 그리고 프로세스가 **종료 코드 0**으로 끝난다 | **고장** — 스모크가 각 인라인 메시를 OBJ로 쓰고 다시 쓴 MJCF 옆에 둔다 |
| `contype = conaffinity = 0` | 콜라이더 없음(시각 전용) | 유지 |
| 그 밖의 `contype`/`conaffinity` 비트마스크 | 매핑되지 않음: `PhysicsCollisionGroup` 두 개(`robotCollisionGroup` = `/World/robot`, 그리고 그것과 필터되는 프로토타입 그룹). 로봇 아래 모든 콜라이더가 서로 충돌 | **누락**(비트마스크 의미론) |
| 지오메트리 `friction`(1.0 / 0.005 / 0.0001) | 어디에도 물리 머티리얼을 쓰지 않음 → PhysX 기본 머티리얼 | **누락** |
| `solref`, `solimp`, `condim`, `margin` | 없음 | 누락(대응물 없음) |
| `<option>` `integrator`, `cone`, `solver`, `iterations`, `impratio` | 없음. 씬은 TGS, patch 마찰, PCM, 아티큘레이션 위치/속도 반복 32/1, 바디 16/1 | 누락(대응물 없음) |
| `<option timestep>`, `gravity` | 스모크의 `World(physics_dt=…)`가 MJCF대로 200 Hz / 1000 Hz, 중력 9.81로 돈다(임포터 자체의 `create_physics_scene` 경로는 시험하지 않음) | 호출자가 설정 |
| — | 모든 강체에 `physxRigidBody:angularDamping = 0.05` | **추가**(MuJoCo에는 없음) |
| `<sensor>` | — | 측정 안 함: 두 픽스처 모두 센서를 선언하지 않음 |

추가로 측정: `World.reset()`은 호출자의 첫 `world.step` 전에 물리를 **두** 스텝 진행한다(자유
바디가 z = 0.3 − 3·g·dt²에서 시작). 따라서 PhysX의 k행은 MuJoCo의 k+2행이다.

### 7.4 같은 제어, MuJoCo CPU 대 PhysX — 수치만, 허용 오차는 주장하지 않음

제어: 모든 `<position>` 액추에이터는 스텝 k 전에 자기 `ctrlrange`에서
`mid + ¼·(hi−lo)·sin(2πk/100 + i)`를 받는다. MJCF 자체의 타임스텝으로 100스텝, k행은 스텝 k
후의 상태, `max |Δq|`는 100행 전체에서. MuJoCo 3.13.0(`~/venvs/es`), 따로 적지 않으면 PhysX CPU
파이프라인. 모두 `~/artifacts/plan-x/i0/run_all.sh`에서 나왔다.

| 씬 | 쌍 | max \|Δq\| (rad): pan, lift, elbow, wrist_flex, wrist_roll, gripper | 자유 바디 max \|Δpos\| |
|---|---|---|---|
| SO-101 내보낸 것(dt 5 ms) | MuJoCo 대 PhysX CPU | 0.781, 1.73, 1.05, 1.26, 1.30, 0.851 | 큐브 3.9e-4 m(MuJoCo의 소프트 접촉이 0.11 mm 가라앉힘, PhysX는 z = 0.02 유지) |
| SO-101 원본 픽스처 | MuJoCo 대 PhysX CPU | 0.781, 1.73, 1.05, 1.26, 1.30, 0.851 | 큐브 3.9e-4 m |
| SO-101 내보낸 것 | MuJoCo 대 PhysX GPU 파이프라인(`cuda:0`) | 0.716, 1.71, 1.13, 1.26, 1.31, 0.853 | 큐브 3.9e-4 m |
| SO-101 내보낸 것 | PhysX CPU 대 PhysX GPU | 0.069, 0.261, 0.273, 0.065, 0.0012, 0.0028 | 1.3e-7 m |
| SO-101 내보낸 것 | PhysX CPU 실행 대 재실행 | 0(JSON이 비트 단위로 같음) | 0 |
| SO-101 | MuJoCo 내보낸 것 대 MuJoCo 원본 | ≤ 1.2e-15 | 0 |
| SO-101 | PhysX 내보낸 것 대 PhysX 원본 | 2.1e-4, 3.6e-5, 1.8e-4, 1.9e-5, 1.1e-5, 4.8e-7 | 0 |
| `mesh_box`(dt 1 ms, 100스텝, 자유 낙하) | MuJoCo 대 PhysX, 내보낸 것과 원본 모두 | — | 1.99e-3 m(리셋의 두 스텝: 99행에서 0.25046 대 0.24847 m) |
| `mesh_box`, 500스텝(≈ 0.2 s에 착지) | MuJoCo 대 PhysX | — | 1.55e-2 m(충돌 직후 299행에서 MuJoCo 0.0464 m, PhysX 0.0500 m, 정지 시 MuJoCo 0.04989 m, PhysX 0.05000 m). 메시 상자와 프리미티브 상자는 두 엔진 모두에서 서로 같다 |

SO-101 실행의 99행에서 MuJoCo는 (−0.165, −0.014, 0.142, 0.418, 0.289, 0.550), PhysX는
(−0.946, −1.740, −0.686, 1.644, 1.131, −0.136)에 있다. 실행 대부분 동안 모든 드라이브가 2.94
N·m 한계에 걸려 있어서 둘 다 목표를 뒤따라간다.

원인 배분(진단, 같은 스크립트): MJCF에서 드라이브 게인을 다시 써도(`--drive-gains rad`:
stiffness kp, damping kv + 조인트 감쇠, `deg`: 같은 값 × π/180) 차이는 줄지 않는다(최대
0.99, 1.69, … 및 0.84, 1.67, … rad). MuJoCo의 수동 조인트 감쇠와 frictionloss를 0으로 두면
(PhysX가 버리거나 클램프 안으로 옮기는 두 행), MuJoCo는 임포트된 그대로의 PhysX 실행과 처음
10스텝 동안 다섯 조인트에서 ≤ 1.5e-3 rad(wrist_flex 0.032), 20스텝 동안 wrist_flex(0.074)를
뺀 모든 조인트에서 ≤ 0.02 rad로 맞는다. 두 엔진이 힘 한계에 걸려 있는 동안에는 그 두 행이
발산의 원인이다. 그 뒤의 잔차(최대 0.83 rad)는 원인을 배분하지 않았다.

**I1을 위해:** 이 임포터 위에 만드는 `PhysXBackend`는 (1) 인라인 메시를 파일로 쓰고, (2)
강체 없는 `worldBody` 아티큘레이션 루트를 제거하고, (3) MJCF 자유 바디를 용접하지 않고, (4)
드라이브를 직접 저작하고(kp는 도당 단위로, kv, 그리고 수동 감쇠와 frictionloss에 대한 결정),
(5) 마찰 머티리얼을 저작하고, (6) 리셋/스텝 횟수를 직접 소유하고, (7) 이 호스트에서는 물리
전용 익스피리언스로 시작해야 한다 — 각각이 매핑 리포트의 한 행이다. I1이 각각을 어떻게
했는지는 §8이 적는다.

## 8. `PhysXBackend` (M11 I1, 2026-09-23)

`crates/es-physics-backend/src/physx.rs` + `python/physx_ref.py`, 패킷
`docs/packets/M11/I1-physx-backend.md`. §7과 같은 서버, 같은 venv에서 측정했고 산출물은
`~/artifacts/plan-x/i1/`에 있다. 여기의 모든 수치는 측정값이며 허용 오차는 주장하지 않는다.

### 8.1 어떻게 도는가

- **임포트가 아니라 서브프로세스.** `physx_ref.py`는 `proc.rs`의 JSON-lines 프로토콜(`load`,
  `reset`, `set_ctrl`, `step`, `state`, `set_state`)을 말한다. `set_params`는 없다: env별 모델
  필드를 측정하지 않았으므로 `ModelParams`를 선언하지 않고 `set_params`는 이름을 대며 거부한다.
- **`ES_ISAAC_PYTHON`**이 인터프리터를 가리킨다. 설정되지 않았거나 `import isaacsim`을 못 하면
  `is_available()`의 `Err`, 즉 호출자의 SKIPPED exit 3이다. Rust 쪽이
  `OMNI_KIT_ACCEPT_EULA=YES`를 설정하고 §7.2의 호환 라이브러리를 `LD_LIBRARY_PATH` 앞에 붙인다
  (`ES_ISAAC_COMPAT_LIBS`, 없으면 존재할 때 `$HOME/opt/isaac-compat/root/usr/lib/x86_64-linux-gnu`).
  스크립트가 `isaaclab.python.headless.kit`을 스스로 고른다(`ES_ISAAC_EXPERIENCE`로 덮어씀).
  호출자는 `ES_ISAAC_PYTHON` 말고는 아무것도 설정하지 않는다.
- **스크립트는 파일로 실행한다.** 새 발견: 프로세스를 `python -c <script>`로
  시작하면(`sys.argv == ["-c"]`) `SimulationApp`이 로그 없이 breakpad 핸들러에서 죽는다. 내장
  스크립트를 blake3 이름으로 임시 디렉터리에 한 번 쓰고 파일로 실행한다(`Process::spawn_command`).
- **stdout은 프로토콜이다.** Kit은 stdout에 로그를 쓴다. 스크립트는 무엇이든 임포트하기 전에
  fd 1을 프로토콜용으로 복제하고 fd 1을 stderr로 돌린다. `ES_PHYSX_STDERR=<file>`이면 Kit 로그를
  남긴다.
- **파이프라인:** `ES_PHYSX_DEVICE=cpu`(기본) 또는 `cuda:0`. 패킷은 `LoadConfig`로 고르라 했지만
  `LoadConfig`는 I1의 컨텍스트 밖인 `es-physics-core`에 있어서 벗어났다. 파이프라인은 엔진
  버전(`isaacsim 5.1.0 physx 107.3.26 cpu|gpu`)에 기록되므로 `backend_identity`와
  `evaluation.lock`이 둘을 구별하고, `gpu_resident`도 이를 따른다.
- **Env:** `n_envs > 1`이면 `/World/envs/env_0`을 `GridCloner(spacing = 0)`로 복제하고
  `filter_collisions`로 env를 서로 격리한다. 모든 env는 MJCF 좌표 그대로다.
- **상태**는 `omni.physics.tensors`(힌지 dof와 모든 링크 포즈는 아티큘레이션 뷰, 자유 바디는
  강체 뷰)로 읽고 쓰며, MJCF에서 다시 세운 MuJoCo 배치를 따른다: 바디는 `world`를 먼저 두고
  깊이 우선, 조인트는 바디 순서, 자유 조인트는 `pos ‖ quat(w,x,y,z)`이고 선속도는 월드 프레임에서
  본 바디 원점의 속도, 각속도는 바디 프레임(PhysX는 질량 중심 속도와 월드 프레임 각속도를 준다.
  바디의 COM 오프셋으로 양방향 변환). `xquat`은 `x,y,z,w`. 프로세스당 스테이지 하나: 다시
  로드하면 새 프로세스다(로드된 스테이지까지 ≈ 4 s).

### 8.2 임포터 항목: 고친 것과 선언한 것

"고침"은 `physx_ref.py`가 수리하고 능력 선언의 `BackendQuirk`(그래서 모든 `evaluation.lock`)에
들어간다는 뜻이고, "행"은 매핑 리포트의 한 행(`es backend compare`, spec 14.4 관문)이며 모두
경고다. 조용히 버리는 것은 없다.

| 항목(§7.3과 새로 찾은 셋) | 처리 |
|---|---|
| 인라인 `<mesh vertex/face>`가 치명적, exit 0 | **고침**: 인라인 메시마다 임시 디렉터리에 `<메시 이름>.obj`로 쓰고 MJCF 사본은 `file=`을 쓴다. 새 발견: 임포터는 메시를 **이름으로** 찾는다. 다른 이름의 파일이면 충돌체가 아예 임포트되지 않고, 충돌체 prim은 지오메트리가 아니라 메시 이름을 딴다. 행 `ContactMesh` approximated: 볼록 껍질 |
| 강체 없는 `worldBody` 아티큘레이션 루트 | **고침**: `ArticulationRootAPI` 제거 |
| `fix_base`가 MJCF 자유 바디를 용접 | **고침**: `rootJoint_<body>`와 자유 바디의 `ArticulationRootAPI`를 지워 평범한 강체로 만든다. 자식이 있는 바디의 자유 조인트(부유 베이스)는 로드 때 이름을 대며 거부한다 |
| `World.reset()`이 숨은 2스텝을 돈다 | **고침**: `reset()` 뒤에 MuJoCo의 qpos0(힌지 0, 자유 바디는 임포트된 포즈)과 0 속도를 써서 tick 0이 MuJoCo의 tick 0이다. `mesh_box`의 자유 낙하가 이제 충돌 직후인 tick 225까지 MuJoCo와 1e-6 m 안에서 맞는다(I0: 처음부터 1.99e-3 m 차이) |
| kp가 USD의 도당 게인에 변환 없이 기록, kv 누락 | **고침**: 모든 드라이브를 텐서 API로 쓴다 — stiffness = kp, damping = kv, max force = forcerange — 텐서 API는 SI 단위(N·m/rad)를 받는다. 힌지 하나짜리 탐침(kp 1, kv 0.05, 중력 없음, 무작위 목표)에서 확인: 500틱 동안 MuJoCo와 max \|Δq\| 1.1e-3 rad, 57.3배 단위 오류라면 이렇게 가까이 머물 수 없다. position 액추에이터가 없는 dof에는 드라이브가 없다. 행 `actuator.pd` / `ActuatorPosition` approximated(PhysX 드라이브는 암시적) |
| `ctrlrange` 누락 | **고침**: MuJoCo처럼 스크립트가 ctrl을 그 범위로 자른다 |
| 조인트 `damping`이 힘 클램프 안쪽의 드라이브 감쇠가 됨 | **행** `joint.damping` approximated: `-d·q̇`(와 조인트 스프링 `-k(q − springref)`)를 매 물리 스텝 전에 명시적 조인트 힘으로 가한다. MuJoCo처럼 클램프 바깥이지만 MuJoCo에서는 암시적이다 |
| `frictionloss` 누락 | **행** `JointFrictionLoss` unsupported: PhysX 조인트 마찰은 계수이지 건마찰 토크가 아니다 |
| 지오메트리 마찰: 머티리얼 없음 | **행** `geom.friction` approximated: 미끄럼 계수마다 머티리얼 하나, static = dynamic = μ, 반발 0, 결합 `max`(MuJoCo의 규칙, `priority`는 무시) |
| `contype`/`conaffinity` 비트마스크 | **행** `geom.contype_conaffinity` unsupported, 충돌하는 지오메트리가 1/1도 0/0도 아닐 때 묻는다. 임포터의 충돌 그룹은 아무것도 거르지 않았으므로 지운다 |
| `solref`, `solimp`, `margin` | **행** `ContactSoftParams` unsupported. spec 17.2의 `contact.soft_params`는 상태를 유지하고, 버려진다고 적은 노트를 단다 |
| `condim` 4/6(비틀림, 구름) | **행** `ContactCondim6` unsupported |
| `cone = elliptic` | **행** `ContactElliptic` unsupported: 피라미드로 돈다 |
| `<option>` integrator / solver / iterations / impratio | **행** `option.solver` unsupported, 모든 씬에서: PhysX TGS, 임포터의 32 / 1 아티큘레이션 반복 |
| 충돌하지 않는 지오메트리의 `mass=` | **행** `body.mass_from_geoms` unsupported, `<inertial>` 없이 지오메트리를 가진 바디가 있을 때 묻는다 |
| 모든 바디에 +0.05 각 감쇠 | **고침**: 모든 바디와 아티큘레이션에서 각·선 감쇠 0, 수면 끔(임계값 0) |
| 물리 전용 익스피리언스 | **고침**: 스크립트가 고른다 |
| armature | 임포터가 유지한다(§7.3): `JointArmature` native. spec 17.2의 `joint.armature` 행은 "unsupported" 상태를 유지하고, 임포터가 유지한다는 노트를 단다 |
| **새 발견:** 임포터의 `/collisions`(와 `/meshes`, `/visuals`) 프로토타입 | **고침**: 정의되고 활성이며 충돌이 켜진 prim이라서 PhysX가 모든 바디의 충돌체를 **월드 원점의 정적 충돌체로** 한 번 더 시뮬레이션한다(원점에서의 겹침 쿼리가 `/collisions/wrist/…`, `/collisions/cube/…`에 맞는다). 루트에서 비활성화한다. 바디로 들어가는 인스턴스 참조는 그대로 합성된다(SO-101에서 34개 충돌체가 남고 모두 `/World` 아래). 충돌체가 자기 머티리얼을 가질 수 있도록 인스턴스를 인스턴스 불가로 바꾼다 |
| **새 발견:** 0인 자유 조인트 쿼터니언 | **고침**: env의 리셋은 무작위화하는 자유 조인트의 위치만 쓰고 쿼터니언은 0으로 둔다. MuJoCo는 이를 항등으로 읽지만 PhysX는 **변환 전체를 조용히 버린다**(위치 포함). 스크립트가 `mju_normalize4`처럼 정규화한다(노름 < 1e-15 → 항등). 고치기 전에는 reach A0의 큐브가 움직이지 않았고 nominal이 0.0이었다 |

아예 매핑하지 않는 것 — 프로세스가 뜨기 전에 매핑 리포트가 막는다(spec 14.4): 볼 조인트, 센서
(어댑터가 읽지 않음), `velocity` / `general` / 사이트 액추에이터, 텐던, 높이 필드. `JointSlide`는
구현했지만 어떤 픽스처도 쓰지 않아서 선언하지 않는다.

### 8.3 오라클 2: `es backend compare`, 500틱

`es backend compare --scene <scene> --backends mujoco-cpu,physx --ctrl-random --ticks 500`
([−1, 1]의 splitmix64 목표, 두 엔진 모두 `ctrlrange`로 자름), 그리고 Isaac이 있을 때만 도는
테스트 `physx_against_mujoco_cpu_is_measured`(모든 액추에이터가 `mid + ¼(hi−lo)·sin(2πk/100 + i)`,
§7.4의 제어, 500틱, 이어서 PhysX 실행 대 두 번째 PhysX 실행).

| 씬 | 파이프라인 | 제어 | max \|Δqpos\| | max \|Δqvel\| | 에너지 프록시 MuJoCo / PhysX(마지막 틱) | max 에너지 프록시 Δ | 발산 틱(1e-6) |
|---|---|---|---|---|---|---|---|
| SO-101 | CPU | 무작위 | 7.023e-2 | 3.20 | 6.564 / 7.440 | 5.04 | 0 |
| SO-101 | CPU | 사인 | 3.03e-1 | 5.90 | 31.41 / 35.47 | 30.0 | 0 |
| `mesh_box` | CPU | 없음(자유 낙하, ≈ 0.2 s에 착지) | 1.55e-2 | 2.22 | 2.7e-9 / 1.3e-8 | 4.91 | 225 |
| SO-101 | GPU(`cuda:0`) | 무작위 | 7.025e-2 | 3.20 | 6.564 / 7.440 | 5.04 | 0 |
| SO-101 | GPU | 사인 | 3.54e-1 | 5.60 | 31.41 / 35.51 | 31.1 | 0 |
| `mesh_box` | GPU | 없음 | 1.55e-2 | 2.21 | 2.7e-9 / 8.3e-8 | 4.91 | 225 |

- **재실행:** CPU `es backend compare` 두 번이 두 씬 모두 같은 출력을 냈고, 테스트의 PhysX 대
  PhysX 재실행은 두 씬 모두 **비트 단위로 같다**(max \|Δqpos\| = max \|Δqvel\| = 0, 발산 틱
  없음). GPU 파이프라인의 재실행도 두 곳, 두 씬 모두에서 비트 단위로 같다. 사인 제어에서 CPU
  파이프라인 대 GPU 파이프라인: SO-101은 max \|Δqpos\| 0.124 rad로 tick 0에서 발산하고,
  `mesh_box`는 6.9e-6 m로 tick 225(충돌)에서 발산한다. 두 파이프라인은 MuJoCo에서 똑같이 멀다.
- **§7.4와 비교:** 같은 사인 제어에서 I0의 임포트된 그대로의 PhysX는 100스텝 안에 MuJoCo와
  최대 1.73 rad 벌어졌다. §8.2의 수리를 거치면 500스텝 동안 0.30 rad다. SO-101은 f32만으로도
  tick 0에서 발산하고, `mesh_box`는 충돌 전까지 1e-6 안에 머문다.

### 8.4 오라클 3: `es eval run --backend physx`로 reach A0

W0b의 reach A0(`~/artifacts/plan-w/w0b/reach/seed0/checkpoints/4000.esb`, mujoco-cpu에서 학습),
`tests/fixtures/rl/evaluation-reach.toml`(16 에피소드 × 4 스위트), 씬 `so101_pick_place.xml`,
같은 바이너리:

| 백엔드 | nominal | observation_delay | torque_noise | backlash | 평균 에피소드 길이(nominal) | 벽시계 |
|---|---|---|---|---|---|---|
| mujoco-cpu | **0.5625** | 0.3125 | 0.5 | 0.5 | 129.4 | 13 s |
| physx, CPU 파이프라인 | **0.125** | 0.0 | 0.3125 | 0.0625 | 184.9 | 5분 43초 |
| physx, GPU 파이프라인(`cuda:0`) | **0.125** | 0.0625 | 0.0 | 0.0625 | 183.4 | 8분 1초 |
| mjwarp | 무엇이든 뜨기 전에 매핑 리포트가 거부(`ContactElliptic`, M11 X1) | | | | | |

실행은 끝까지 돌고, `evaluation.lock`은 위의 모든 quirk와 함께 `backend = physx`를 담으며,
`execution_hash`는 mujoco-cpu의 것과 다르다(`2fdd30a0…` 대 `08851281…`, `hardware_capability`
슬롯). GPU 파이프라인 실행은 또 다른 해시를 갖는다(`e16de409…`, 엔진 버전 `… gpu`). MuJoCo에서 학습한 정책은 PhysX에서 nominal 성공의 4분의 1만 지킨다: 측정했지만 여기서
원인을 배분하지는 않은 sim-to-sim 차이다(남은 §8.2의 행들 — frictionloss, 소프트 접촉, 솔버,
명시적 감쇠 — 이 후보다).

실행 해시는 `physx_ref.py` 자체를 덮지 않는다: 영 쿼터니언 수정 전의 첫 A0 실행은 같은 해시
아래에서 0.0을 냈다. 지금은 모든 프로세스 밖 백엔드의 스크립트(`mjwarp_ref.py`도)가 그렇고,
해시 규칙의 후속 과제다.
