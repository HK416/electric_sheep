# Electric Sheep — 로봇 학습 컴파일러·런타임 기술 사양서 (v1.0)

> **Electric Sheep** — 비전 기반 로봇 학습의 태스크·관측·학습·배포 전 구간을 컴파일하고 추적하는 오픈 플랫폼
>
> Electric Sheep는 제품을 **Robot Learning Compiler & Runtime**으로 정의한다. 물리 시뮬레이션은 이 파이프라인의 한 백엔드이고, 제품의 중심은 **Policy 계층**이다.
>
> **이 문서는 Electric Sheep v1.0 기술 사양이다.** 5종 IR(Task / Observation / Learning / Deployment / Evaluation), 정책과 독립적인 Safety Plane, 해시 체인 기반 재현성, 그리고 수집 → 학습 → 평가 → 배포 → 개입 → 재학습으로 닫히는 Learning Loop을 규정한다. 설계의 근거와 2026년 생태계 안에서의 포지셔닝은 §0에서 다룬다.

---

## 0. 포지셔닝

### 0.1 한 문장 정의

> **Electric Sheep는 로봇 학습 파이프라인의 컴파일러다.** 태스크·관측·학습·배포를 타입이 있는 중간 표현으로 기술하면, 이를 검증하고 GPU 커널과 실행 계획으로 컴파일하며, 시뮬레이션과 실기에서 동일한 의미로 실행하고, 전 과정을 해시 체인으로 추적한다.

물리 시뮬레이션은 이 파이프라인의 **한 백엔드**다. 제품이 아니다.

### 0.2 다섯 핵심 목표

1. **전 구간이 하나의 의미 표현으로 기술된다.** 씬에서 액추에이터까지, 그 사이의 어떤 단계도 "플랫폼 바깥의 Python 코드"로 남지 않는다
2. **재현된다.** 동일 조건에서 비트 단위로 같은 결과가 나오고, 태스크 정의·관측 파이프라인·학습 설정·데이터셋 분할·정책 가중치까지 해시 체인으로 추적된다
3. **비전 기반 정책이 1급이다.** ACT·Diffusion Policy·π₀ 계열·SmolVLA·GR00T 계열을 같은 IR 위에서 정의·학습·평가·배포한다
4. **실기까지 같은 의미로 간다.** 시뮬에서 쓴 관측 전처리와 안전 제약이 실기에서 재구현되지 않고 그대로 실행된다
5. **학습 루프가 닫힌다.** 수집 → 학습 → 평가 → 배포 → 개입 → 실패 데이터 → 재학습이 제품의 워크플로다

### 0.3 2026년 생태계

**물리 엔진 계층은 이미 포화되었다.**

| 프로젝트 | 상태 |
|---|---|
| **Newton 1.0** (2026-03, Linux Foundation, Apache-2.0) | NVIDIA·DeepMind·Disney. Warp + OpenUSD. MuJoCo Warp 주 백엔드, Kamino(maximal coordinate, 폐루프), VBD 변형체, MPM, SDF 충돌, hydroelastic. 미분 가능. **NVIDIA 전용** |
| **MuJoCo Warp** | RTX PRO 6000 Blackwell에서 MJX 대비 locomotion 252×, manipulation 475× 보고 |
| **Genesis World 1.0 + Quadrants** (2026-05) | Python 커널을 CUDA·ROCm·Metal·**Vulkan**·CPU로 JIT. 역방향 autodiff 전 백엔드 1급. 자체 렌더러 Nyx |
| **Isaac Lab** | PhysX + Newton 이중 백엔드, RTX 타일 렌더링 |
| **ManiSkill3** | Vulkan 기반 GPU 병렬 시뮬·렌더 |
| **RoboVerse / MetaSim** | simulator-agnostic 설정, 276 태스크, 500K+ 궤적 |

**반면 학습 루프 계층은 빠르게 표준화되고 있고, 그 중심은 LeRobot이다.**

LeRobot은 데이터셋·실기 제어·정책·평가·배포를 하나의 스택으로 묶었고, 지원 정책 계열이 ACT, Diffusion Policy, VQ-BET, HIL-SERL, TD-MPC부터 π₀, π₀-FAST, π₀.₅, GR00T N1.7, SmolVLA, XVLA, EO-1, MolmoAct2, WALL-OSS, EVO1까지 확장되었다. World Model(VLA-JEPA, LingBot-VA, FastWAM)과 Reward Model(SARM, TOPReward, Robometer) 범주까지 들어왔고, LIBERO·MetaWorld 등 표준 벤치마크에서 통합 평가 스크립트를 제공한다.

**정책 아키텍처의 실제 형태** (`policy(obs) → action` 모델이 왜 부족한지의 근거)

| 정책 | 파라미터 | 입력 | 시간 모델 | 액션 |
|---|---|---|---|---|
| ACT | 52–80M | RGB 다중 뷰 + state | `n_obs_steps = 1` | CVAE + Transformer, horizon 50, 실행 chunk 20 |
| Diffusion Policy | 263M | 시계열 visual conditioning | 관측 윈도 | 조건부 denoising, receding horizon |
| SmolVLA | 450M | RGB + state + **언어** | 관측 윈도 | flow matching, chunk 16, 실행 horizon 4–8. 프레임당 visual token 64개로 압축 |
| π₀ | 3.3–3.5B | RGB 3뷰 + sensorimotor + 언어 | 관측 윈도 | VLM + flow matching action chunk |
| π₀.₅ | — | RGB 2뷰 [3,256,256] + state[15] | — | chunk 16 |

**RTX 4090 추론 지연 실측 (LeRobot 보고)**

| 정책 | 지연 |
|---|---|
| ACT (52M) | **5.0 ms** |
| SmolVLA (450M) | **99.2 ms** |
| π₀ (3.5B) | **209.4 ms** |
| Diffusion Policy (263M) | **369.8 ms** |

**이 표가 성능 설계 전체를 규정한다.** 물리 스텝이 1 ms인데 정책 추론이 5–370 ms다. 비전 기반 학습에서 병목은 물리 처리량이 아니라 **정책 추론 지연과 비전 대역폭**이다. GPU 접촉 솔버 최적화는 잘못된 대상이다.

### 0.4 빈틈 분석

| 축 | LeRobot | Newton/MJWarp | Genesis | Isaac Lab | RoboVerse | **Electric Sheep** |
|---|---|---|---|---|---|---|
| 정책 계열 지원 | 최상 | 없음 | 없음 | RL 중심 | 중간 | **IR로 기술·컴파일** |
| 시뮬레이션 | 없음 (외부 의존) | 최상 | 최상 | 최상 | 통합 | **백엔드로 채택** |
| **관측 파이프라인 명세** | 코드 | 코드 | 코드 | config | config | **타입 있는 IR** |
| **학습 그래프 명세** | Python 클래스 | — | — | — | — | **타입 있는 IR** |
| **재현성 보증** | config 기록 | 미보증 | 미보증 | 미보증 | 미보증 | **비트 단위 + 해시 체인** |
| **안전 런타임** | 없음 | 없음 | 없음 | 없음 | 없음 | **정책 독립 Safety Plane** |
| **평가 정규화** | 벤치마크 스크립트 | — | — | — | 4단계 일반화 | **Evaluation IR** |
| 실기 배포 | 있음 | 없음 | 없음 | 제한적 | 없음 | **동일 IR 실행** |
| 배포 형태 | Python | Python | Python | Omniverse | Python | **단일 정적 바이너리** |

**굵게 표시한 다섯 개가 존재 이유다.** LeRobot이 "무엇을 쓸 수 있는가"를 풀었다면, Electric Sheep는 **"그것이 정확히 무엇이었고, 왜 그렇게 동작했으며, 다시 그렇게 만들 수 있는가"**를 푼다.

**왜 방어 가능한가**
- LeRobot은 Python 라이브러리다. 관측 전처리가 Python 함수라 실기 배포에서 재구현되고, 그 순간 sim과 real이 갈라진다. IR로 기술하면 같은 것이 양쪽에서 실행된다
- 재현성은 소급 확보가 불가능하다. 해시 체인은 설계 시점의 선택이다
- 안전 런타임은 정책 라이브러리의 관심사가 아니다. 그런데 EU 기계규정(2027-01-20 적용)에서는 이것이 시장 접근 조건이 된다(§27)
- 평가 정규화는 논문 재현성과 산업 수용 양쪽에서 필요한데, 아무도 표준을 소유하지 않았다

### 0.5 비목표

- **물리 처리량 경쟁을 하지 않는다.** Newton/MJWarp가 더 빠르면 그것을 백엔드로 쓴다
- **자체 물리 엔진이 필수 산출물이 아니다.** M4 이후 선택 항목이며, 축소 시 1순위로 자른다(§1.9)
- **자체 정책 아키텍처를 발명하지 않는다.** ACT·DP·VLA 계열을 표현·실행·평가할 뿐이다
- **자체 학습 프레임워크를 만들지 않는다.** PyTorch/JAX에 위임한다
- **범용 비주얼 스크립팅 언어가 아니다.** 노드 추가는 RFC 대상이다
- **인증 기관이 아니다.** 증거 수집 도구다(§27.1)
- **클라우드 서비스·관리형 학습 플랫폼을 만들지 않는다**
- **미분 가능 시뮬을 약속하지 않는다.** Newton·Genesis가 이미 한다. M5 실험 경로

### 0.6 타깃 태스크

**1차 (M1–M2): 비전 기반 매니퓰레이션.** Franka/UR 급 7 DoF + 평행 그리퍼, RGB 1–2뷰(224×224 또는 256×256) + joint state, ACT/Diffusion Policy/SmolVLA.

**2차 (M3): 비전 기반 이동.** 사족보행 시각 내비게이션, 모바일 로봇.

**3차 (M4+): 양팔·휴머노이드.** 언어 조건부, π₀ 계열.

로봇군마다 **벤치마크 씬 1개 + 표준 태스크 3개 + Evaluation Suite 1세트**를 정의한다(§10).

---

## 1. 개발 모델: 에이전트 주도 개발

구현은 전량 AI 코딩 에이전트(Claude, Codex 등)가 수행한다. 사람은 코드를 쓰지 않고 사양을 쓰고 오라클을 설계하고 판정한다.

### 1.1 전제

| | 사람 팀 | **에이전트 주도** |
|---|---|---|
| 희소 자원 | 구현 시간 | **명세 정밀도, 오라클 강도, 리뷰 주의력** |
| 실패 양상 | 못 끝냄 | **끝났는데 미묘하게 틀림** |
| 규모 한계 | 인력 | **컨텍스트 윈도, 통합 일관성** |
| 되돌리기 비용 | 높음 | 낮음 (재생성이 싸다) |

**명제 1: 에이전트 산출물의 품질 상한은 그것을 판정하는 오라클의 강도다.** 약한 테스트는 약한 코드를 *승인*하므로 사람 팀보다 훨씬 비싸다.

**명제 2: 되돌리기가 싸므로 탐색이 싸다.** 구현 후보 3개를 만들어 벤치마크로 고르는 것이 정상 전술이다.

### 1.2 작업 패킷

계획의 최소 단위는 패킷이다. 다섯 요소를 갖지 않으면 패킷이 아니다.

```
id / type / crate / depends
context      건드릴 파일 범위. CI가 diff를 이 범위로 제한한다
spec         구현 내용. 모호함이 없어야 한다
oracle       사람 없이 통과/실패를 판정하는 실행 가능한 명령
acceptance   수용 기준 체크리스트
forbidden    하지 말아야 할 것. 인접 패킷의 소관을 명시
```

**오라클을 쓸 수 없는 작업은 설계가 덜 된 작업이다.** 그런 항목은 구현 패킷이 아니라 설계 문서 패킷으로 바뀐다.

### 1.3 작업 유형

| 유형 | 정의 | 사람의 역할 | 이 프로젝트의 예 |
|---|---|---|---|
| **A. 자율** | 명세가 외부에 있고 판정이 기계적 | 패킷 작성, 주 1회 표본 검토 | 임포터, 직렬화, Python 바인딩, IR 왕복, 태스크 변환기, LeRobot 데이터셋, CLI, 테스트 하네스, 센서 노이즈 모델 |
| **B. 검토** | 경계가 명확하나 설계 취향 개입 | 패킷 작성, **diff 전체 검토 (≤800줄)** | ECS, 자원 관리, 에디터 UI, 텔레메트리, IR 코어 타입, Safety Plane 구현 |
| **C. 설계 위임** | 알고리즘·타입 설계를 사람이 하고 구현만 위임 | **설계 문서 작성**, 결과 정밀 검토 | IR 타입 시스템, 정규화·해시, Learning IR lowering, 초월함수, 결정적 누산기, 배치 도메인 스케줄러 |
| **D. 사람 주도** | 실패가 조용하고 진단이 탐색적 | **가설 수립**, 에이전트는 실험 실행 | 수치 정합성, 정책 지연 최적화, 비전 대역폭 튜닝, 벤더 드라이버, sim2real 갭 진단 |

- **A**: 초록불 자동 머지
- **B**: diff 검토 후 머지. 800줄 초과 시 분할 요구
- **C**: `docs/design/<주제>.md`가 선행 산출물. 에이전트는 설계를 만들지 않는다
- **D**: 패킷이 "구현"이 아니라 "실험"이다

**이 프로젝트의 유형 구성은 에이전트 주도에 유리하다.** 자체 GPU 물리 솔버(유형 D 집중)를 후순위로 두고 IR·데이터·배포(유형 A/B/C 집중)를 앞에 두기 때문이다.

### 1.4 오라클 우선 원칙

**검증 하네스가 구현보다 먼저 온다. 예외 없다.**

이 프로젝트가 가진 참조 오라클:

| 대상 | 오라클 |
|---|---|
| 물리 백엔드 | MuJoCo (Python 서브프로세스), MJWarp, Newton |
| **Learning IR lowering** | **PyTorch 참조 구현 — 동일 IR을 PyTorch로 실행한 결과가 정답** |
| **Observation IR** | **LeRobot 전처리와의 수치 일치, 실기 카메라 캡처와의 분포 비교** |
| **정책 동등성** | **LeRobot 체크포인트를 불러와 동일 입력에 동일 액션이 나오는가** |
| 초월함수 | MPFR / `rug` |
| IR 정규화 | 셔플 후 해시 일치 property test |
| 결정적 누산기 | 임의 순서 셔플 후 비트 비교 |
| 렌더러 | 골든 이미지, PT/RS 채널 일치 |
| Safety Plane | 위반 시나리오 픽스처 전수 |

**Learning IR의 오라클이 PyTorch라는 점이 중요하다.** 자체 추론 런타임이 LeRobot의 ACT/SmolVLA 체크포인트를 로드해 동일 출력을 내는지가 M1의 게이트다. 이것이 통과하면 "IR이 실제 정책을 표현한다"가 증명된다.

### 1.5 컨텍스트 예산

```
코어 크레이트 1개는 단일 컨텍스트 윈도에 들어가야 한다.
  목표 ≤ 6,000 lines / 상한 ≤ 10,000 lines (소스, 테스트 제외)
  초과 시 분할 패킷이 필수다. 예외 없다.
```

`cargo xtask context-budget`이 CI에서 차단한다. 크레이트 경계를 넘는 작업은 자동으로 유형 B 이상이며, 사람이 인터페이스를 먼저 확정한다.

### 1.6 저장소 구조

```
AGENTS.md / CLAUDE.md        루트 규칙
docs/
  conventions.md             §3 규약
  invariants.md              절대 불변식 + 기계/사람 검사 구분
  design/                    유형 C 선행 설계 문서
  api-notes/                 ash·Slang·torch·LeRobot 실제 시그니처
  packets/<M>/               패킷 정의 + BACKLOG.md
crates/<name>/AGENTS.md      크레이트별 지침
xtask/                       모든 검증의 단일 진입점
tests/golden/                CI에서 읽기 전용
```

전문은 부록 C.

### 1.7 에이전트 실패 모드와 방어

| 실패 모드 | v1.0에서의 구체적 위험 | 방어 |
|---|---|---|
| 그럴듯한 오답 | **Learning IR lowering이 돌고 형상도 맞는데 수치가 미묘하게 다름 → 정책이 조용히 열화** | **PyTorch 참조 대조를 M1 게이트로**. 체크포인트 로드 후 출력 비트 근접 비교 |
| 환각 API | `ash`, Slang, `torch` C++ API, LeRobot 스키마는 학습 데이터가 오래됨 | `docs/api-notes/` 자동 생성, 버전 고정, `cargo check` 전 완료 보고 금지 |
| 테스트 고치기 | 골든 이미지·골든 텐서를 출력에 맞춤 | 골든은 CI 읽기 전용. `xtask verify-goldens`가 git 이력 확인 |
| 조용한 범위 확장 | 겸사겸사 리팩터링 | `xtask check-scope`가 diff를 패킷 범위와 대조 |
| 재구현 | `es-math::approx` 대신 자기 다항식 → §3.4 결정성 붕괴 | 크레이트 `AGENTS.md`의 "이미 있는 것" 목록 + clippy 린트 |
| 과잉 추상화 | 구현체 1개 트레이트 번식 | **허용 확장점 7개만**: `PhysicsBackend`, `PolicyRuntime`, `TaskNodeFactory`, `LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc` |
| 세션 간 표류 | 명명·오류 타입·할당 패턴 불일치 | 루트 `AGENTS.md` 규약 + 주간 코히런스 감사 |
| 결정성 침식 | `HashMap` 순회, `f64` 시간 누적, FP 원자 합산 | 타입으로 차단 + clippy + `invariants.md` |
| **안전 우회** | **Safety Plane을 "테스트 편의로" 비활성화하는 코드 경로 추가** | **Safety Plane 비활성 경로가 존재하지 않게 설계. 컴파일 타임에 필수 통과(§9.4)** |
| 문서·코드 불일치 | 사양 절 번호 참조가 깨짐 | `xtask check-spec-refs` |

### 1.8 사람의 역할

**위임하지 않는 것:** §0.5 비목표와 부록 A.1 확정 사항 변경 / 유형 C 설계 문서 / 유형 D 가설·결론 / 오라클 승인 / 게이트 판정 / 해시 정의·공개 스키마 변경 / **Safety Plane 요구사항 정의** / 주간 코히런스 감사.

**리뷰 예산** (사람 2–3명, 주 40시간 기준): 패킷 작성 8–12h / 설계 문서 6–10h / diff 검토 6–10h / 유형 D 8–14h / 감사 2–3h / 관리 2–3h.

유형 A는 이 표에 없다. **A가 전체의 절반을 넘어야 이 배분이 성립한다.**

### 1.9 범위 축소 순서

1. **자체 물리 솔버 전체** — MJWarp/Newton 백엔드로 충분하다
2. 경로추적기 (M4)
3. 3DGS real-to-sim (M3)
4. IR-C (Control Graph)
5. 편집 가능 Visual Graph — 읽기 전용만 유지
6. USD 네이티브 리더 — Bake로 대체
7. 멀티 GPU 이종 구성
8. NPU 추론
9. 텔레오퍼레이션 OpenXR

**절대 자르지 않는 것:** 5종 IR과 검증기, **Safety Plane**, Evaluation IR, 해시 체인, 오라클 인프라, LeRobot 호환.

---

## 2. 언어 및 툴체인

### 2.1 순수 Rust 코어

코어에 C++를 두지 않는다. C 라이브러리는 `bindgen`으로 허용. Slang 컴파일러는 빌드 시점에만 실행한다.

**학습 경로에서 완화되는 경계 하나:** 학습은 PyTorch/JAX에 위임하므로 **학습 경로에서 Python은 1급 의존이다.** 이것은 타협이 아니라 설계다. 코어 런타임과 실기 배포 경로가 Python 없이 동작하면 된다.

```
Python 필요       학습(PyTorch), LeRobot 데이터셋, USD Bake, xacro, MuJoCo 참조
Python 불필요     시뮬 실행, 관측 파이프라인, 정책 추론, Safety Plane, 실기 배포, 에디터
```

### 2.2 핵심 크레이트

`ash`(Vulkan), `gpu-allocator`, `wide`/`multiversion`(SIMD), `crossbeam`, `loom`, `PyO3`+`maturin`, `wasmtime`, `egui`+`winit`, `egui-snarl`(노드 그래프), `gltf`, `openusd`, `zenoh`, `quinn`, `serde`, `proptest`, `blake3`, `ort`(ONNX Runtime, 선택), `safetensors`.

### 2.3 셰이더: Slang

단일 셰이더 언어, SPIR-V 출력, 오프라인 컴파일 + 콘텐츠 해시 캐시. 제네릭의 용도: 정밀도 파라미터화(§3.3), **Observation IR 노드 특수화**(§7.6), **Learning IR 전처리 커널 특수화**(§8.7).

### 2.4 정책 추론 백엔드

`PolicyRuntime` 트레이트 뒤에 셋을 둔다.

| 백엔드 | 용도 | 시점 |
|---|---|---|
| **`TorchRuntime`** (libtorch FFI 또는 Python 서브프로세스) | 학습 중 인루프 추론, 참조 정답 | M1 |
| **`OnnxRuntime`** (`ort`) | 배포, 양자화 검증, NPU EP | M2 |
| **`VulkanRuntime`** (Slang 커널 + `cooperative_matrix`) | 대량 병렬 소형 정책, 벤더 중립 배포 | M3 |

**Vulkan Compute 추론만으로는 π₀ 3.5B를 돌릴 수 없다.** 대형 VLA는 `TorchRuntime`/`OnnxRuntime`이 담당하고, Vulkan 경로는 ACT급(52M) 소형 정책과 벤더 중립 온로봇 배포를 담당한다.

### 2.5 외부 도구 의존

| 기능 | 의존 | 시점 | 필수 여부 |
|---|---|---|---|
| Slang → SPIR-V | Slang 컴파일러 | 빌드 | 개발자만 |
| 관측·전처리 GPU lowering | 런타임 Slang | 컴파일 시점 | **캐시 히트 시 불필요** |
| 학습 | Python + PyTorch | 학습 시점 | 학습 워크플로만 |
| 대형 VLA 추론 | libtorch 또는 ONNX Runtime | 런타임 | 해당 정책만 |
| LeRobot 데이터셋 | Python (읽기는 Rust 네이티브) | 상호운용 | 내보내기만 |
| USD Bake / xacro | Python | 임포트 | 해당 포맷만 |
| PyTorch 제로카피 | CUDA/HIP 드라이버 | 런타임 | §21 capability |

`es --check-deps`가 환경별 가능 범위를 출력한다.

---

## 3. 규약·정밀도·결정성 계약

### 3.1 좌표계·단위 규약

| 항목 | 규약 |
|---|---|
| 좌표계 | 오른손, **Z-up**, X-forward (ROS/MJCF/URDF) |
| 길이/각도/질량/시간 | m, rad, kg, s |
| 쿼터니언 | **xyzw**, 단위 노름, w ≥ 0 |
| 관성 | 바디 프레임 3×3 대칭, 주축 분해 캐시 |
| **이미지 좌표** | **원점 좌상단, x 우측, y 하단 (OpenCV 규약)** |
| **카메라 좌표** | **+Z forward, +X right, +Y down (OpenCV). ROS `REP-103` 광학 프레임과 동일** |
| **색공간 기본** | **sRGB (비선형). 선형 변환은 명시적 노드로만** |

마지막 세 줄이 비전 파이프라인의 핵심 규약이다. 좌표·색공간 규약이 없으면 전처리 재구현 시 조용히 어긋난다. §7의 `ImageSpec`이 이를 타입에 싣는다.

### 3.2 초월함수

SPIR-V `GLSL.std.450`은 ULP 상한만 규정하고 정확한 결과는 벤더·드라이버별로 다르다. `es-math`가 자체 minimax 다항식을 제공하고 Rust와 Slang이 계수·연산 순서를 공유한다. 정확도 목표 ≤ 2 ULP, 참조는 MPFR.

물리·관측·보상 커널에서 표준 초월함수 호출을 clippy 린트로 차단한다. **신경망 커널은 예외다**(§8.7). 활성화 함수의 비트 재현성은 정책 가중치와 추론 백엔드에 종속되므로 별도 계약을 쓴다.

### 3.3 정밀도

| 하드웨어 | `shaderFloat64` | GPU 물리 | 관측 파이프라인 |
|---|---|---|---|
| NVIDIA Turing+ | 지원 (1/64) | 가능 | 가능 |
| AMD RDNA2+ | 지원 (1/16–1/32) | 가능 | 가능 |
| Intel Arc Alchemist | 에뮬레이션 | DoubleFloat만 | 가능 |
| Intel Xe2 Battlemage | 지원 | 가능 | 가능 |
| Apple / MoltenVK | 미지원 | 불가 (CPU) | 가능 |

- 스칼라 표현은 타입 파라미터: `F64Native` / `DoubleFloat` / `F32`. 기본값은 M2 측정으로 확정
- **관측·학습 파이프라인은 FP32가 기본이고 FP16/BF16을 선택 지원한다.** 정책이 그 정밀도로 학습되기 때문이다
- 렌더링 FP32, 카메라-상대 좌표

### 3.4 결정적 실행 계약

Vulkan float-controls의 `shaderDenormFlushToZeroFloat32` 등은 **지원 여부를 나타내는 device capability 질의값**이지 동작을 강제하는 설정이 아니다. 실제 적용은 SPIR-V execution mode/control로 이루어진다.

**올바른 4단계**

```
1. Capability Query          VkPhysicalDeviceFloatControlsProperties 질의
        ↓
2. Required Capability Check 필요한 항목이 지원되지 않으면 결정적 모드 거부
        ↓
3. SPIR-V Execution Mode     DenormFlushToZero / RoundingModeRTE / SignedZeroInfNanPreserve
                             를 셰이더에 명시. NoContraction 데코레이션
        ↓
4. Deterministic Pipeline    아래 계약 전체를 만족하는 실행 계획
```

**결정적 실행 계약 = 다음 전부**

```
float controls        (capability 확인 + execution mode 명시)
+ no contraction      (NoContraction 데코레이션)
+ deterministic reduction   (RFA/binned summation, §18.4)
+ deterministic scheduling  (정적 파티셔닝, 단일 큐)
+ fixed subgroup behavior   (서브그룹 크기를 논리적으로 고정)
+ fixed algorithm           (디바이스 속성에서 유도한 분기 금지)
+ fixed compiler/SPIR-V     (Slang 버전 + SPIR-V 콘텐츠 해시)
+ fixed driver/device       (계층 1 조건, §3.5)
```

**`NoContraction`만으로 GPU 결정적 실행이 보장되지 않는다.** 여덟 항목이 모두 필요하다.

### 3.5 결정성 계층

| 계층 | 조건 | 보증 |
|---|---|---|
| **0. 의미 동일** | 동일 `*_hash` | 정의가 같다 (수치 보증 아님) |
| **1. Bitwise** | 동일 `execution_hash` + 동일 디바이스·드라이버 | 비트 일치 |
| **2. 교차 백엔드** | CPU ↔ GPU, 또는 백엔드 교체 | 정의된 허용오차 |
| **3. 물리 의미** | vs MuJoCo, vs 실측 | 물리 지표 허용오차 |
| **4. 정책 동등** | **동일 IR을 PyTorch vs 자체 런타임으로 실행** | **액션 텐서 허용오차** |

계층 4는 §8.9에서 정의한다.

**결정적 모드 금지 사항:** 원자 FP 합산, fast-math, 디바이스 속성 유도 워크그룹 수, 부동소수 시간 누적, 물리·관측 커널의 표준 초월함수, 다중 컴퓨트 큐 물리 커널, 전역 RNG, `HashMap` 순회 의존, **Safety Plane 우회**.

---

## 4. 시스템 아키텍처

### 4.1 플레인 구조

```
┌──────────────────────────────────────────────────────────────────────┐
│ AUTHORING PLANE                                                      │
│  Python │ TOML │ Visual Graph │ LLM 생성 │ 외부 변환(LeRobot/IsaacLab)│
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ TASK IR                 §6                                           │
│  씬 참조 / 목표 / 보상 / 종료 / 랜덤화 / 리셋 / ObservationSpec 선언  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ OBSERVATION IR          §7        sensor → tensor                    │
│  ImageSpec / 카메라 동기화 / crop·resize·normalize / 색공간 변환      │
│  TemporalWindow / 마스킹 / state 결합                                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ LEARNING IR             §8        tensor → action chunk              │
│  VisionEncoder / StateEncoder / LanguageEncoder / Fusion             │
│  TemporalEncoder / PolicyHead(Regression·Diffusion·FlowMatching)     │
│  ActionChunker / ActionUnnormalizer                                  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ POLICY RUNTIME          §2.4                                         │
│  Torch │ ONNX │ Vulkan │ NPU     비동기 추론 / 청크 버퍼 / 지연 예산   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ DEPLOYMENT IR + SAFETY PLANE    §9                                   │
│  토크·속도·작업공간 한계 / 충돌 제약 / rate limit / 액션 유효성        │
│  NaN 감시 / 추론 데드라인 / stale observation / E-stop / 폴백 컨트롤러 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
                    CONTROL / ACTUATOR  →  ROBOT (또는 시뮬)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 계층을 관통:  hash · version · provenance · telemetry · replay · evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌──────────────────────────────────────────────────────────────────────┐
│ EVALUATION IR           §10    perturbation suite × metric × 수용기준 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ EXECUTION SUBSTRATE                                                  │
│  PhysicsBackend (MJWarp │ Newton │ PhysX │ 자체)  §17                │
│  Vision Data Plane (렌더 → 관측 캡처)            §15                 │
│  ECS · 잡 · 시간 모델 · 실패 시맨틱              §5, §18             │
│  GPU 자원 · TensorTransport                      §21                 │
│  자산 파이프라인 (USD/glTF/MJCF/URDF, 3DGS)      §16                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 크레이트 레이어링

| 레이어 | 크레이트 |
|---|---|
| 0 | `es-math` |
| 1 | `es-core` (ECS, 잡, 시간, 실패 시맨틱) |
| 2 | `es-gpu`, `es-assets`, `es-usd` |
| 3 | `es-actuator`, `es-sensor`, `es-physics-core` |
| 4 | `es-physics-backend` (MJWarp/Newton/PhysX 어댑터), `es-physics-cpu`, `es-physics-gpu` |
| 5 | `es-render`, `es-splat` (3DGS) |
| 6 | **`es-ir`** (5종 IR 스키마·타입 시스템·정규화·해시·검증) |
| 7 | **`es-compile`** (스케줄링, lowering, 백엔드 코드젠) |
| 8 | **`es-policy`** (PolicyRuntime: Torch/ONNX/Vulkan), **`es-safety`** (Safety Plane) |
| 9 | `es-env` (실행 오케스트레이션, 배치 도메인) |
| 10 | `es-data`, `es-telemetry`, `es-eval` |
| 11 | `es-ros2`, `es-py`, `es-script`, `es-transport` |
| 12 | `es-editor` |

**규칙 (CI 강제)**
1. 상위 → 하위만. 동일 레이어 간 의존 금지
2. `es-physics-cpu` ⇎ `es-physics-gpu`. 공유는 `es-physics-core`
3. 레이어 ≤2 는 Vulkan 심볼을 모른다 (`es-gpu` 예외)
4. **어떤 크레이트도 `es-editor`를 의존하지 않는다**
5. 코어는 CUDA/HIP 심볼을 링크하지 않는다. `es-transport`만 예외
6. **`es-ir`는 컴파일러·백엔드·PyTorch를 모른다.** IR은 실행 수단에 중립이다
7. **`es-ir`는 UI 타입을 모른다.** 레이아웃은 `.eslayout` 사이드카
8. **`es-safety`는 `es-policy`를 의존하지 않는다.** 안전은 정책을 신뢰하지 않는다. 한 방향으로만 흐른다
9. 크레이트 소스 ≤ 10,000 lines (§1.5)

**규칙 8이 v1.0의 핵심 구조 결정 중 하나다.** Safety Plane이 정책 계층에 의존하면 정책의 가정이 안전 검증에 새어 들어간다.

### 4.3 PhysicsBackend

**외부 물리 백엔드가 기본 경로다.**

```
PhysicsBackend
├── MuJoCoWarpBackend    기본. GPU 병렬, MJCF 네이티브          M1
├── NewtonBackend        Kamino·VBD·hydroelastic 필요 시        M2
├── MuJoCoCpuBackend     결정적 참조, CI 오라클                  M1
├── PhysXBackend         Isaac Lab 자산 호환                     M3
└── NativeBackend        자체 CPU/Vulkan 솔버 (선택, §17.4)      M4+
```

백엔드는 capability 집합을 선언하고 컴파일러가 IR 요구사항과 대조한다(§11.6). 백엔드는 **결정성 등급도 선언**하며, 외부 백엔드는 계층 1을 선언하지 않는다.

**이 전환의 효과**
- M1–M3에서 물리 엔진 개발이 임계 경로에서 빠진다. Newton의 발전이 그대로 이득이 된다
- 대신 **백엔드 간 의미 매핑**(§17.2)이 새로운 핵심 작업이 된다. 같은 Task IR이 MJWarp와 Newton에서 다르게 동작하면 안 된다
- §1.9 축소 순서 1번이 "자체 물리 솔버 전체"인 이유

---

## 5. IR 패밀리

### 5.1 경계 정의

다섯 IR의 책임을 명확히 나눈다. **경계가 흐려지면 Task IR이 Blueprint처럼 팽창한다.**

| IR | 질문 | 소유 | 배치 의미론 |
|---|---|---|---|
| **Task IR** | 세계에서 무엇이 일어나고 무엇이 좋은가 | 씬 참조, 보상, 종료, 랜덤화, 리셋, ObservationSpec **선언** | env 배치 강제 |
| **Observation IR** | 센서 출력이 어떻게 텐서가 되는가 | ImageSpec, 동기화, 기하 변환, 정규화, 시간 윈도 | env 배치 기본, 카메라 축 허용 |
| **Learning IR** | 텐서가 어떻게 액션 청크가 되는가 | 인코더, 융합, 시간 모델, 정책 헤드, 액션 디코딩 | **배치 자유. env와 분리** |
| **Deployment IR** | 액션이 어떤 제약 아래 실행되는가 | 안전 한계, 실행 모드, 데드라인, 폴백 | 로봇 단위 |
| **Evaluation IR** | 좋은지 어떻게 판정하는가 | perturbation suite, 지표, 수용 기준 | 에피소드 배치 |

**Task IR은 `ObservationSpec`을 선언만 하고 구현하지 않는다.** "전방 카메라 RGB와 관절 위치가 관측이다"까지가 Task IR이고, "그것을 224×224로 리사이즈하고 ImageNet 통계로 정규화하고 2프레임을 쌓는다"는 Observation IR이다.

같은 Task IR에 다른 Observation IR을 붙일 수 있어야 한다. ACT는 `n_obs_steps=1`이고 Diffusion Policy는 관측 윈도를 쓰는데, 태스크는 동일하다.

### 5.2 배치 의미론의 차등 적용

모든 태스크 노드는 env 배치 의미론을 갖는다. **그러나 이 원칙을 Learning IR에는 적용하지 않는다.**

```
simulation batch      4,096 env          물리 스텝
     ↓ (활성 카메라만)
observation batch     1,024 env × 1 view  렌더 + 전처리
     ↓ (추론 배치)
inference batch       256 samples         정책 추론
     ↓ (학습 미니배치)
training batch        64 sequences        gradient step
```

네 도메인은 서로 다른 크기·주기·장치를 가질 수 있다. §12에서 오케스트레이션을 정의한다.

### 5.3 해시 체계

```
asset_hash ──┐
             ├──► scene_hash ──┐
             │                 │
             │   task_graph_hash (저작 정체성, 노드 ID 포함)
             │           ↓
             └──►    task_hash    (의미 정체성, 정규화 후)
                         │
        observation_hash ┤
           learning_hash ┤
             policy_hash ┤   (가중치 + 아키텍처 + base model)
            dataset_hash ┤   (content + schema + split)
          deployment_hash┤
          evaluation_hash┘
                         ↓
                  compiler_hash · runtime_hash · hardware_capability
                         ↓
                  ┌──────────────────┐
                  │  execution_hash  │
                  └──────────────────┘
```

```
execution_hash = H(
    task_hash, observation_hash, learning_hash, policy_hash,
    dataset_hash, deployment_hash,
    compiler_hash, runtime_hash, hardware_capability
)
```

**`execution_hash`가 §3.5 계층 1의 조건이다.** 그리고 무엇이 바뀌었을 때 어느 해시가 바뀌는지가 명확하므로, "실질적 변경"의 영향 범위를 기계적으로 판정할 수 있다(§27.1).

`dataset_hash`가 셋으로 나뉘는 이유: 같은 데이터셋이라도 **분할이 다르면 학습 샘플 구성이 달라진다.** train/val/test split을 해시하지 않으면 재현이 성립하지 않는다.

### 5.4 공통 타입 시스템

다섯 IR이 공유하는 포트 타입.

```
PortType = (ElemType, Shape, Semantics)

ElemType    f32 | f16 | bf16 | f64 | i32 | u8 | bool
Shape       [D1, D2, ...]           배치 축은 도메인이 정한다 (§5.2)
Semantics   Unit × Frame × TimeRef  (+ 이미지는 ImageSpec, §7.2)
```

**Unit** — 물리 단위 대수

```
Dimensionless | Length(m) | Angle(rad) | Mass(kg) | Time(s)
Velocity | AngularVelocity | Acceleration | Force | Torque
Pressure | Current | Voltage | Quaternion | RotationMatrix
Normalized(lo, hi)            정규화된 값
Pixel | Luminance | Depth(m)  이미지 도메인
Token                          언어·이산 표현
```

곱·나눗셈은 단위 대수를 따르고 덧셈은 단위가 일치해야 한다. **정책 입력 포트는 `Normalized` 또는 `Dimensionless` 또는 `Token`만 받는다.** 정규화되지 않은 raw 값을 신경망에 넣는 흔한 실수를 컴파일 타임에 잡는다.

**Frame**

```
World | LocalOrigin | Body(id) | Sensor(id) | Joint(id)
Camera(id)            카메라 광학 프레임 (§3.1)
Image(id)             픽셀 좌표
Policy                정책 내부 표현 (프레임 검사 없음)
```

**TimeRef**

```
Tick(PhysTick)                        현재 물리 틱
Sensor { id, align }                  align = Hold | Interpolate | Reject
Window { base, n_steps, stride }      TemporalWindow (§7.5)
```

시간 정렬이 명시되지 않은 결합은 컴파일 에러다.

```
ERROR TYPE-014  시간 정렬이 명시되지 않았습니다

  Concat "policy_obs" 의 입력:
    joint_pos     TimeRef = Sensor(encoder, 1000 Hz)
    camera_front  TimeRef = Sensor(cam_front, 30 Hz)

  해결: time_align = "hold" | "interpolate" | "reject"
        (ACT·Diffusion Policy 는 통상 "hold")
```

---

## 6. Task IR

### 6.1 범위

세계와 태스크의 의미만 담는다.

```
허용   씬 참조, 목표, 보상 항, 종료 조건, 도메인 랜덤화, 리셋 분포,
       커리큘럼, 센서 스케줄, ObservationSpec 선언, 데이터셋 기록 대상

금지   ECS 직접 변이, 물리 솔버 내부, 블로킹, GPU 동기화, 스레드,
       raw 디바이스 접근, 파일·네트워크 I/O, 신경망, 전처리 구현
```

금지는 "런타임에 막는다"가 아니라 **"IR에 그런 노드가 없다"**로 구현한다.

### 6.2 IR-D / IR-C 분리

| | **IR-D (Dataflow)** | **IR-C (Control)** |
|---|---|---|
| 의미 | 순수 DAG, 부작용 없음 | 시퀀싱, 분기, 서브태스크 |
| 노드 | 관측·보상·종료·랜덤화·리셋 | `Sequence`, `Branch`, `SubTask`, `Repeat` |
| 일정 | **M0 스키마 / M1 lowering** | **M4** |

표준 비전 매니퓰레이션 태스크(pick&place, reach, push)는 IR-D만으로 표현된다. IR-C가 필요한 것은 멀티스테이지 조립이고 M4다.

### 6.3 노드 집합

**Source** (phase = Observation)
```
GetJointState  GetBodyPose  GetBodyVelocity  GetContact
GetSensor      GetTime      GetRandom(stream 필수)
GetLanguage    태스크 지시문 (VLA용, v1.0 추가)
```

**Transform** (순수 함수)
```
Transform  Normalize  Clamp  Arith  MathFn  Norm  Dot  Cross
Compare    Logic      Select  Concat  Slice  Reduce
```

**Sink**
```
ObservationSpec   관측 선언 (구현은 Observation IR)
ActionSpec        액션 공간 선언 (§9.2)
Reward            보상 항 (name, weight, aggregation)
Terminate         success | failure | timeout
Randomization     대상 파라미터 + 분포
ResetState        초기 상태 분포
Record            데이터셋 기록 대상
```

`Parallel`은 존재하지 않는다. 병렬화는 컴파일러가 결정한다. `Wait`·`Repeat`·`Condition`은 IR-C 또는 `Compare`+`Select`로 대체된다.

### 6.4 실행 의미론

실행 순서는 **phase → 명시적 데이터 의존 → 틱 스케줄 → 노드 ID**로 결정한다. 그래프 배치 위치는 의미에 영향을 주지 않는다.

```
PrePhysics → Physics → PostPhysics → Observation → Reward → Termination → Record
```

phase 간 역방향 의존은 컴파일 에러다. 모든 노드는 활성 `EnvMask`를 암묵 입력으로 받으며, 격리된 env(§18.5)는 계산에서 제외된다.

### 6.5 표현식

문법은 Rhai 유사로 유지하되 **`es-ir`의 파서가 읽어 IR 서브그래프로 확장한다.** Rhai 인터프리터는 호출되지 않는다.

```
사용자 입력:
    norm(pose("gripper").pos - pose("cube").pos) < 0.02

확장 결과:
    GetBodyPose("gripper") ─┐
                            ├─ Arith(-) ─ Norm(L2) ─ Compare(<, 0.02) ─ bool[env]
    GetBodyPose("cube")   ─┘
```

지원 함수는 §6.3 노드에 1:1 대응한다. 루프·변수 할당·함수 정의는 파스 에러다. Rhai는 에디터 자동화와 오프라인 처리 같은 저주기 경로에만 남는다.

### 6.6 결정성 규칙

| 코드 | 규칙 |
|---|---|
| `DET-001` | 전역 RNG 금지. `TaskRng(seed, EnvId, PhysTick, stream)` 필수 |
| `DET-002` | 벽시계 접근 금지 (구조적) |
| `DET-010` | 표준 초월함수 금지. `es-math::approx`만 |
| `DET-020` | 불안정 반복 순서 금지 |
| `DET-021` | 미선언 RNG 스트림 경고 |
| `DET-030` | `Reduce(unordered=true)`는 결정적 모드에서 거부 |
| `DET-040` | 외부 백엔드 사용 시 계층 1 선언 불가 |

---

## 7. Observation IR

### 7.1 왜 독립 계층인가

센서 출력에는 `ElemType, Shape, Unit, Frame, TimeRef`를 부여하고 `GetSensor`가 이를 쓴다. **그러나 이미지에는 이것으로 부족하다.**

시뮬레이터는 이미 렌즈 왜곡, 롤링 셔터, 모션 블러, 노출, 샷 노이즈를 모델링한다(§18.3). 그런데 그 정보가 학습 입력으로 전달되지 않았다. 결과적으로 두 가지 문제가 생긴다.

1. **전처리가 Python으로 재구현된다.** 시뮬에서는 LeRobot 변환을, 실기에서는 별도 코드를 쓴다. 그 순간 sim과 real이 갈라지고 원인 추적이 불가능해진다
2. **카메라 파라미터가 정책과 함께 이동하지 않는다.** 실기 카메라의 intrinsic이 시뮬과 다른데 정책은 그것을 모른다

Observation IR은 **센서에서 정책 입력 텐서까지의 전 구간을 하나의 타입 있는 그래프로 기술**하고, 시뮬·데이터셋·실기에서 동일하게 실행한다.

### 7.2 `ImageSpec`

이미지 포트는 다음 계약을 싣는다.

```rust
pub struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub channels: ChannelFormat,   // Rgb | Rgba | Gray | Depth | Seg | Normal | Flow
    pub dtype: ImageDType,         // U8 | U16 | F16 | F32
    pub color_space: ColorSpace,   // SRgb | Linear | Rec709 | Raw
    pub camera_model: CameraModel, // Pinhole | Fisheye | Equirect | OrthoDepth
    pub intrinsics: Intrinsics,    // fx fy cx cy (+ skew)
    pub extrinsics: Transform,     // T_body_camera, Frame 검사에 사용
    pub distortion: DistortionModel, // None | BrownConrady(k1..k3,p1,p2) | KannalaBrandt
    pub shutter: ShutterModel,     // Global | Rolling { readout: Duration, dir }
    pub exposure: Duration,
    pub rate_hz: f32,
    pub depth_scale: Option<f32>,  // Depth 채널의 단위 (m/LSB)
}
```

**컴파일러가 검사하는 것**

```
ERROR OBS-021  색공간 불일치

  VisionEncoder "dinov2" 는 Linear 입력을 기대합니다.
  camera_front 는 ColorSpace::SRgb 입니다.

  해결: ColorTransform(SRgb → Linear) 노드를 삽입하거나,
        인코더 정규화 스펙을 sRGB 기준으로 바꾸십시오.
        (ImageNet 통계는 sRGB 기준입니다)

ERROR OBS-034  intrinsic 이 리사이즈에 반영되지 않았습니다

  Resize(640×480 → 224×224) 후 CameraProjection 노드가
  원본 intrinsic 을 사용합니다.

  해결: Resize 노드가 intrinsic 을 자동 스케일하도록
        rescale_intrinsics = true 를 설정하십시오.
```

두 번째가 실무에서 특히 자주 나는 버그다. 리사이즈·크롭 후 intrinsic을 갱신하지 않으면 3D 추론이 조용히 틀린다. **`Resize`·`Crop` 노드가 `Intrinsics`를 변환하는 것이 타입 시스템에 들어 있다.**

### 7.3 노드 집합

**Input**
```
ImageInput(sensor_id)      ImageSpec 을 센서 등록부에서 가져온다
StateInput(spec)           관절·IMU·F/T 등 벡터 상태
LanguageInput(source)      태스크 지시문 또는 외부 입력 (VLA)
```

**Geometric**
```
Resize(w, h, filter, rescale_intrinsics)
Crop(rect | center | random, rescale_intrinsics)
Pad / Undistort / Rectify / Warp(homography)
CameraProjection           3D 점 ↔ 픽셀. intrinsic·extrinsic 소비
```

**Photometric**
```
ColorTransform(src → dst)  sRGB ↔ Linear ↔ Rec709
ToGray / ChannelSelect
Normalize(mean, std | range)   Unit → Normalized
QuantizeU8 / Dequantize
```

**Temporal**
```
TemporalWindow(n_steps, stride, align)   학습 입력 의미 (§7.5)
FrameStack(n)                            채널 축 연결 (특수 케이스)
Delta(n)                                 프레임 차분 (이벤트 유사)
```

**Structural**
```
Concat(axis, time_align)   Stack(axis)   Mask(source)
MultiViewPack(cameras)     다중 카메라 → [V, C, H, W]
```

**Augmentation** (학습 시에만 활성, 평가 시 비활성)
```
RandomCrop / ColorJitter / RandomErasing / GaussianNoise
```

증강 노드는 `training_only = true` 플래그를 갖는다. **Evaluation IR로 실행할 때 자동 비활성화되며, 이것이 IR 수준에서 보장된다.** 평가에 증강이 켜진 채로 도는 사고를 구조적으로 막는다.

### 7.4 `ObservationSpec`과의 연결

Task IR이 선언한 `ObservationSpec`이 Observation IR의 입력 계약이 된다.

```
Task IR
  └── ObservationSpec
        ├── "rgb_front"    : Sensor("cam_front"), ChannelFormat::Rgb
        ├── "rgb_wrist"    : Sensor("cam_wrist"), ChannelFormat::Rgb
        └── "joint_state"  : JointState("franka"), 7 DoF + gripper
              │
              ▼
Observation IR
        ├── ImageInput(cam_front) → Resize(224) → ColorTransform → Normalize
        ├── ImageInput(cam_wrist) → Resize(224) → ColorTransform → Normalize
        ├── StateInput(joint)     → Normalize(joint_limits)
        └── TemporalWindow(n=2, align=Hold) → Concat
              │
              ▼
        policy_input : { rgb: f32[2, 2, 3, 224, 224], state: f32[2, 8] }
                              ↑  ↑                         ↑
                           time  view                    time
```

**같은 Task IR에 여러 Observation IR을 붙일 수 있다.** ACT용(`n_steps=1`), Diffusion Policy용(`n_steps=2`), π₀용(3뷰 + 언어)이 같은 태스크를 공유한다. `task_hash`는 같고 `observation_hash`만 다르다.

### 7.5 시간 모델 3계층

`History<T,N>` 하나로 시간을 다루면 신호 처리 버퍼와 학습 입력 의미와 신경망 시간 표현이 구분되지 않는다. **그래서 셋을 분리한다.**

```
History<T, N>          시스템 레벨 버퍼
   = 링버퍼. 센서 지연 모델링, 최근 N 샘플 보관. Observation IR 밖.
   소유: es-core, es-sensor

TemporalWindow(n, stride, align)    학습 입력 의미
   = "정책은 최근 2프레임을 200ms 간격으로 본다"
   소유: Observation IR
   영향: 텐서 형상, 메모리 예산, 데이터셋 샘플링

TemporalEncoder(kind, params)       신경망 시간 표현
   = Transformer / TemporalConv / GRU / 없음(단일 프레임)
   소유: Learning IR
```

```
FrameStack(4)            와   TemporalTransformer(8 frames)
는 완전히 다른 의미다.
전자는 채널 축을 4배로 만드는 전처리,
후자는 8개 토큰 시퀀스에 어텐션을 거는 아키텍처 선택이다.
```

이 구분이 있어야 ACT(`n_obs_steps=1`, 시간 인코더 없음), Diffusion Policy(관측 윈도 + 시계열 conditioning), π₀(VLM 토큰 시퀀스)를 같은 IR로 표현할 수 있다.

### 7.6 실행

- **시뮬:** 렌더 출력(§15)이 GPU에 상주한 채로 전처리 커널이 실행된다. 호스트 왕복 없음
- **데이터셋:** 기록 시점에 `ImageSpec`을 함께 저장하고, 학습 시 같은 IR을 CPU 또는 GPU에서 실행
- **실기:** 카메라 드라이버 출력에 같은 IR을 실행. **`es-runtime-embedded`가 Observation IR 평가기를 포함한다**(§9.6)

세 경로가 동일한 IR을 실행하므로 전처리 불일치가 구조적으로 발생하지 않는다. 검증: 같은 입력에 세 경로의 출력이 비트 일치(정수 연산) 또는 허용오차 내(부동소수).

### 7.7 오라클

| 항목 | 검증 |
|---|---|
| LeRobot 전처리 동등성 | 같은 이미지에 LeRobot 변환 vs Observation IR → 허용오차 내 |
| intrinsic 변환 정확성 | 리사이즈·크롭 후 3D→2D 투영 오차 < 0.1 px |
| 색공간 왕복 | sRGB → Linear → sRGB 무손실 |
| 실기 분포 비교 | 실기 카메라 캡처 vs 시뮬 렌더의 채널 통계·주파수 스펙트럼 |
| 세 경로 일치 | 시뮬·데이터셋·실기 경로의 출력 일치 |

---

## 8. Learning IR

### 8.1 목적과 경계

**신경망을 Task IR에 넣지 않는다.** Task IR이 Blueprint처럼 팽창하면 §0.5가 경계하는 문제가 그대로 발생한다. Learning IR은 별도 IR이다.

```
허용   인코더 선택·구성, 융합 방식, 시간 모델, 정책 헤드, 액션 디코딩,
       정규화 통계, 청크·horizon 설정, 사전학습 백본 참조

금지   레이어 단위 네트워크 저작 (그건 PyTorch가 한다)
       임의 텐서 연산 그래프
       학습 루프 제어 (옵티마이저·스케줄러는 training config)
```

**핵심 원칙: 네트워크 내부는 opaque, 인터페이스 의미는 typed.**

```
                LearningGraph
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
   Preprocessor              PolicyHandle (opaque)
   (IR로 완전 기술)                  │
                        ┌───────────┼───────────┬──────────┐
                        ▼           ▼           ▼          ▼
                       ACT     Diffusion    FlowMatch    VLA
```

`PolicyHandle`은 가중치 내부를 모르지만 **반드시 metadata contract를 갖는다.** 그 계약이 Learning IR의 타입 검사 대상이다.

### 8.2 구조

```rust
pub struct LearningGraph {
    pub schema_version: u32,
    pub inputs: Vec<TensorPort>,     // Observation IR 출력과 계약
    pub nodes: Vec<LearningNode>,
    pub outputs: Vec<TensorPort>,    // ActionSpec 과 계약
    pub policy: PolicyHandle,
}
```

### 8.3 노드 집합

**Encoder**
```
VisionEncoder { backbone, pretrained, frozen, out_dim, token_count }
    backbone = ResNet18 | ResNet34 | ViT{size} | DINOv2 | SigLIP | SmolVLM | Custom(hash)
StateEncoder { kind, out_dim }          MLP | Identity
LanguageEncoder { tokenizer, model, max_len }
```

**Fusion**
```
Concat | CrossAttention | FiLM | AdaLN | TokenConcat
```

**Temporal**
```
TemporalEncoder { kind, n_frames, ... }
    kind = None | TemporalConv | Transformer | GRU | Mamba
```

**Head**
```
RegressionHead   { horizon, action_dim }              ACT, BC
DiffusionHead    { horizon, n_steps, scheduler }      Diffusion Policy
FlowMatchingHead { horizon, n_steps }                 π₀, SmolVLA
DiscreteHead     { vocab, horizon }                   VQ-BET, RT-2 계열
EnergyHead       { ... }                              IBC 계열
```

**Action**
```
ActionChunker    { horizon H, execute_chunk K, replan_hz }
ActionUnnormalizer { stats_source }       데이터셋 통계 또는 명시
```

**참조**
```
PolicyBundle(uri | hash)   전체를 하나의 불투명 번들로 참조 (π₀ 등 대형 VLA)
```

마지막이 중요하다. **π₀ 3.5B를 노드 단위로 분해하지 않는다.** `PolicyBundle`로 통째 참조하되, 입출력 계약(§8.4)은 타입 검사된다. 반대로 ACT 급은 노드로 구성해 백본 교체·동결 같은 실험이 IR 수준에서 가능하다.

### 8.4 Policy 메타데이터 계약

```yaml
policy:
  architecture: act
  base_model: null                       # 또는 "lerobot/pi05_base"
  base_model_hash: "b3:9c4e..."
  input:
    rgb_front:   { shape: [3, 224, 224], dtype: f32, space: normalized_imagenet }
    rgb_wrist:   { shape: [3, 224, 224], dtype: f32, space: normalized_imagenet }
    joint_state: { shape: [8], dtype: f32, space: normalized_joint_limits }
    language:    null
  temporal:
    observation_window: 1                # ACT 기본
  output:
    action_dim: 8
    horizon: 50                          # 예측 길이
  control:
    execute_chunk: 20                    # 실행 길이
    replanning_hz: 10
    execution_mode: receding_horizon
  runtime:
    dtype: f32
    expected_latency_ms: 5.0             # RTX 4090 기준 (§0.3)
    deadline_ms: 50.0                    # 초과 시 §9.4 폴백
```

**컴파일러가 검사하는 것**
- Observation IR 출력 형상·정규화 공간과 `input`의 일치
- `ActionSpec`(Task IR)의 `action_dim`과 `output`의 일치
- `execute_chunk ≤ horizon`
- `replanning_hz`와 `dt_ctrl`의 정수 관계
- **`deadline_ms`가 제어 주기 안에 들어가는가** — 안 들어가면 Safety Plane 폴백 전략이 필수(§9.4)

마지막 검사가 실무적으로 중요하다. Diffusion Policy는 RTX 4090에서 370 ms다. 10 Hz 제어(100 ms)에 넣으면 매 스텝 데드라인을 넘긴다. 컴파일 시점에 이것이 드러나야 한다.

```
ERROR LRN-052  추론 지연이 제어 주기를 초과합니다

  정책:      diffusion_policy (263M)
  측정 지연: 369.8 ms (RTX 4090)
  제어 주기: 100 ms (10 Hz)
  execute_chunk: 20 → 실효 재계획 주기 2000 ms

  이 구성은 다음 중 하나가 필요합니다:
    (a) 비동기 추론 + 청크 버퍼링 (§8.6) — execute_chunk ≥ 4 필요, 현재 20 OK
    (b) Safety Plane 폴백 정책 지정 (§9.4)
    (c) 더 빠른 정책 (ACT 5.0 ms, SmolVLA 99.2 ms)

  현재 (a)는 만족하나 (b) 폴백이 지정되지 않았습니다.
```

### 8.5 액션 모델링

`policy(obs) → action` 모델을 쓰지 않는다.

```
obs_t
  ↓
policy
  ↓
[a_t, a_{t+1}, ..., a_{t+H-1}]     action chunk, H = horizon
  ↓
execute first K                     K = execute_chunk
  ↓
receding horizon 재계획
  ↓
controller (dt_ctrl)
```

```rust
pub struct ActionSpec {
    pub space: ActionSpace,        // JointPosition | JointVelocity | JointTorque
                                   // | EEPose | EEDelta | Gripper | Composite
    pub dim: usize,
    pub horizon: usize,            // 예측 길이 H
    pub execute_chunk: usize,      // 실행 길이 K ≤ H
    pub control_rate_hz: f32,
    pub execution_mode: ActionExecutionMode,
    pub normalization: NormalizationSpec,
    pub limits: ActionLimits,      // Safety Plane 이 소비 (§9)
}

pub enum ActionExecutionMode {
    OpenLoopChunk,        // 청크 전체 실행 후 재계획
    RecedingHorizon,      // K개 실행 후 재계획 (기본)
    TemporalEnsemble,     // ACT 방식: 겹치는 예측을 지수가중 평균
    RealTimeChunking,     // 이전 청크 실행 중 다음 청크 계산 (§8.6)
}
```

`TemporalEnsemble`은 ACT가 실제로 쓰는 방식이므로 1급으로 둔다. 겹치는 시점의 여러 예측을 가중 평균하는 것은 IR 수준에서 기술 가능하고, 실기와 시뮬에서 동일하게 실행되어야 한다.

### 8.6 비동기 추론과 청크 버퍼

추론 지연이 제어 주기보다 크다는 것이 정상 상황이므로, 런타임이 이를 1급으로 다룬다.

```
제어 스레드 (dt_ctrl = 100 ms)
   t=0    청크 A[0..20] 실행 시작, A[0] 출력
   t=100  A[1] 출력
   ...
   t=400  A[4] 출력  ← 이 시점에 추론 스레드가 obs_400 으로 B 계산 시작
   ...
   t=770  B 도착 (370 ms 소요)
   t=800  A[8] 대신 B[0]으로 전환?  ← 전환 정책이 필요하다

추론 스레드 (비동기)
   obs_t 캡처 → 정책 추론 → 청크 도착 → 버퍼 교체
```

**전환 정책** (`ChunkBlendPolicy`)
```
HardSwitch        즉시 교체. 불연속 가능
LinearBlend(n)    n스텝에 걸쳐 선형 혼합
TemporalEnsemble  겹치는 구간 가중 평균 (ACT 방식)
```

**청크 고갈(buffer underrun)은 안전 사건이다.** 청크가 다 떨어졌는데 새 청크가 안 왔으면 Safety Plane이 개입한다(§9.4). 이 사건은 카운터로 기록되고 Evaluation IR의 지표가 된다(§10.3).

### 8.7 Lowering

```
Preprocessor 부분 (Observation IR + Learning IR 의 인코더 전단)
    → Slang 커널로 컴파일. GPU 상주, 호스트 왕복 없음
    → 실기에서는 es-runtime-embedded 의 CPU/NPU 커널

PolicyHandle 부분
    → PolicyRuntime 에 위임 (§2.4)
    → Torch: libtorch 또는 서브프로세스
    → ONNX: ort
    → Vulkan: Slang + cooperative_matrix (ACT 급 소형만)

ActionChunker / Unnormalizer / 전환 정책
    → 결정적 CPU 코드. 실기·시뮬 동일
```

**경계가 명확하다.** 전처리와 후처리는 IR이 완전히 소유해 결정적으로 실행하고, 네트워크 순전파만 런타임에 위임한다. 이것이 §8.1의 "내부는 opaque, 인터페이스는 typed"의 구현이다.

### 8.8 학습과의 연결

Learning IR은 **학습 루프를 소유하지 않는다.** PyTorch가 한다. 대신 다음을 생성한다.

```
es learn export --ir task.toml --obs obs.toml --learning policy.toml
  →  lerobot_config.yaml       LeRobot 학습 설정
  →  dataset_spec.json         데이터셋 스키마 + 분할 정의
  →  preprocess.py             Observation IR 의 PyTorch 미러 (검증용)
  →  policy_stub.py            LearningGraph 의 PyTorch 구성
  →  training.lock             §19.3 학습 아이덴티티
```

**`preprocess.py`는 참조 구현이지 정답이 아니다.** 정답은 IR이고, 이 파일은 학습 프레임워크가 같은 전처리를 하도록 생성된다. 두 경로의 출력 일치가 CI 게이트다(§7.7).

### 8.9 계층 4: 정책 동등성 (§3.5)

**동일 Learning IR을 서로 다른 `PolicyRuntime`으로 실행했을 때 액션이 일치하는가.**

| 비교 | 허용오차 |
|---|---|
| PyTorch(fp32) ↔ ONNX(fp32) | 액션 최대 절대오차 ≤ 1e-5 |
| PyTorch(fp32) ↔ ONNX(int8) | 태스크별 정의. 성공률 저하 ≤ 2%p |
| PyTorch(fp32) ↔ Vulkan(fp32) | ≤ 1e-4 (소형 정책만) |
| 동일 런타임 재실행 | 비트 일치 (결정적 커널 사용 시) |

**M1 게이트: LeRobot의 ACT 체크포인트를 로드해 동일 관측에 동일 액션을 낸다.** 이것이 "IR이 실제 정책을 표현한다"의 증명이다.

---

## 9. Deployment IR과 Safety Plane

### 9.1 원칙

> **정책은 신뢰되지 않는다.** 안전은 정책 바깥에서, 정책과 독립적으로, 결정적으로 강제된다.

```
Policy
  ↓  action chunk
┌─────────────────────────────────────────┐
│ SAFETY PLANE                            │
│  정책 독립 · 결정적 · 실패 시 안전측     │
└─────────────────────────────────────────┘
  ↓  validated action
Controller → Actuator → Robot
```

§4.2 규칙 8이 이것을 구조적으로 보장한다. `es-safety`는 `es-policy`를 의존하지 않는다.

### 9.2 `DeploymentIR`

```rust
pub struct DeploymentIR {
    pub schema_version: u32,
    pub robot: RobotRef,              // 씬 또는 실기 기술
    pub action: ActionSpec,           // §8.5
    pub envelope: SafetyEnvelope,
    pub watchdogs: Vec<Watchdog>,
    pub fallback: FallbackPolicy,
    pub rate: RateSpec,
}
```

### 9.3 Safety Envelope

정책 출력에 적용되는 정적·동적 제약. **전부 결정적이고 전부 필수다.**

| 제약 | 내용 | 위반 시 |
|---|---|---|
| `torque_limit` | 관절별 토크 상한 | 클램프 + 카운터 |
| `velocity_limit` | 관절·EE 속도 상한 | 클램프 + 카운터 |
| `position_limit` | 관절 한계 + 소프트 마진 | 클램프 |
| `workspace` | EE 작업 공간 (박스·실린더·볼록다면체) | 투영 + 카운터 |
| `collision_constraint` | 자기충돌·환경 충돌 최소 거리 | 정지 또는 후퇴 |
| `rate_limit` | 액션 1차·2차 미분 상한 | 필터링 |
| `action_validity` | 정의역·NaN/Inf 검사 | **즉시 폴백** |
| `jerk_limit` | 저크 상한 (선택) | 필터링 |

**클램프와 폴백의 구분이 중요하다.** 토크 상한 초과는 클램프하고 기록한다. NaN은 클램프할 수 없으므로 즉시 폴백이다.

### 9.4 Watchdog과 Fallback

```rust
pub enum Watchdog {
    InferenceDeadline { budget: Duration },     // 추론이 예산 초과
    ChunkUnderrun,                              // 청크 고갈 (§8.6)
    StaleObservation { max_age: Duration },     // 관측이 오래됨
    NanInf,                                     // 액션에 NaN/Inf
    EnvelopeViolationRate { window: usize, max_frac: f32 },
    ControllerHeartbeat { timeout: Duration },
    SensorDropout { sensor: SensorId, max_gap: Duration },
}

pub enum FallbackPolicy {
    HoldPosition,                  // 현재 자세 유지
    ZeroVelocity,                  // 감속 정지
    RetractToHome { traj },        // 미리 계산된 안전 궤적
    HandoffController { id },      // 고전 컨트롤러로 이양
    EmergencyStop,                 // 즉시 정지 + 래치
}
```

**폴백은 정책이 아니다.** 신경망이 아니고, 결정적이며, 정책 런타임이 죽어도 동작해야 한다. `es-safety`는 사전 할당된 워크스페이스에서 no-std로 동작한다.

**시뮬에서도 동일하게 켜진다.** 학습 중 Safety Plane이 꺼져 있다가 배포 때 켜지면, 정책이 안전 제약을 위반하는 방식으로 학습된 것을 배포 직전에 알게 된다. **시뮬 학습 중 envelope 위반율이 Evaluation IR의 1급 지표다**(§10.3).

### 9.5 시뮬과 실기의 동일성

| | 시뮬 | 실기 |
|---|---|---|
| Observation IR | GPU 커널 | CPU/NPU 커널 (동일 IR) |
| Learning IR 전처리 | GPU 커널 | 동일 |
| PolicyHandle | Torch/ONNX/Vulkan | ONNX/Vulkan/NPU |
| ActionChunker | 결정적 CPU | 동일 코드 |
| **Safety Plane** | **동일 코드** | **동일 코드** |
| 컨트롤러 | 시뮬 액추에이터 모델 | 실제 드라이브 |

**`deployment_hash`가 같으면 안전 동작이 같다.** 이것이 §27.1 증거물의 핵심 주장이다.

### 9.6 `es-runtime-embedded`

실기 배포용 최소 런타임. 단일 정적 바이너리, no-std 가능, 힙 할당 0.

```
포함:  Observation IR 평가기 + Learning IR 전·후처리
       + PolicyRuntime (ONNX 또는 Vulkan 또는 NPU)
       + Safety Plane 전체
       + 텔레메트리 링버퍼

제외:  물리 엔진, 렌더러, 에디터, Python, 학습
```

배포 산출물은 `policy.esb` 번들 하나다.

```
policy.esb
├── manifest.json          execution_hash 와 구성 해시 전체
├── observation.ir         §7 (canonical)
├── learning.ir            §8 (전·후처리 부분)
├── deployment.ir          §9 (안전 제약)
├── policy/                weights (onnx | safetensors | spirv)
├── stats/                 정규화 통계
└── signature              선택적 서명
```

**정책 가중치만 배포하지 않는다.** 전처리와 안전 제약을 함께 배포한다.

---

## 10. Evaluation IR

### 10.1 왜 필요한가

현재 로봇 학습에서 정책 비교는 대개 다음 형태다.

```
Policy A = 82%
Policy B = 81%
```

이것으로는 아무것도 알 수 없다. 산업 도입에 필요한 것은 다음이다.

```
                    A       B
nominal            92%     94%
lighting_shift     87%     71%     ← B는 조명에 취약
camera_shift       84%     88%
object_pose_shift  79%     81%
occlusion          61%     63%
latency +20ms      84%     52%     ← B는 지연에 매우 취약
sensor_dropout     73%     70%
actuator_noise     88%     85%
─────────────────────────────────
intervention rate  3.1%    7.8%
envelope violation 0.4%    2.1%    ← B는 안전 제약을 자주 건드림
action smoothness  0.82    0.61
p95 latency        7ms     380ms
```

**Evaluation IR은 이 표를 생성하는 것을 태스크 정의처럼 명세화한다.**

### 10.2 구조

```yaml
evaluation:
  schema_version: 1
  task: tasks/pick_cube.toml
  observation: obs/two_view_224.toml
  episodes_per_cell: 100
  seed_base: 20260912

  suites:
    - name: nominal
      perturbations: []

    - name: lighting_shift
      perturbations:
        - { kind: light_intensity, range: [0.3, 2.5], dist: loguniform }
        - { kind: light_direction, range_deg: 45 }
        - { kind: color_temperature, range_k: [2700, 7500] }

    - name: camera_shift
      perturbations:
        - { kind: camera_extrinsic, pos_sigma_m: 0.02, rot_sigma_deg: 3 }
        - { kind: camera_intrinsic, focal_rel_sigma: 0.02 }

    - name: object_pose_shift
      perturbations:
        - { kind: object_pose, target: "cube", pos_sigma_m: 0.05, yaw_deg: 180 }

    - name: occlusion
      perturbations:
        - { kind: occluder, count: [1, 3], size_m: [0.03, 0.10] }

    - name: latency_injection
      perturbations:
        - { kind: observation_delay, ms: [0, 20, 50] }
        - { kind: action_delay, ms: [0, 20] }

    - name: sensor_dropout
      perturbations:
        - { kind: frame_drop, prob: 0.05, burst: [1, 3] }

    - name: actuator_noise
      perturbations:
        - { kind: torque_noise, rel_sigma: 0.05 }
        - { kind: backlash, rad: [0.0, 0.01] }

  metrics: [success_rate, intervention_rate, collision_rate,
            envelope_violation_rate, action_smoothness, chunk_underrun_rate,
            p50_latency, p95_latency, episode_length, failure_mode_histogram]

  acceptance:
    nominal.success_rate:              ">= 0.85"
    lighting_shift.success_rate:       ">= 0.75"
    latency_injection.success_rate:    ">= 0.70"
    envelope_violation_rate:           "<= 0.01"
    p95_latency_ms:                    "<= 100"
```

### 10.3 지표 정의

| 지표 | 정의 |
|---|---|
| `success_rate` | Task IR의 `Terminate(success)` 도달 비율 |
| `intervention_rate` | 실기·HIL에서 사람 개입이 발생한 에피소드 비율 |
| `collision_rate` | 원치 않는 접촉 발생률 |
| **`envelope_violation_rate`** | **Safety Plane이 클램프·투영한 스텝 비율 (§9.3)** |
| `action_smoothness` | 액션 1차·2차 미분의 정규화 역수 |
| **`chunk_underrun_rate`** | **청크 고갈 발생률 (§8.6)** |
| `p50/p95_latency` | 관측 캡처 → 액션 출력 end-to-end |
| `failure_mode_histogram` | 실패 원인 분류 (타임아웃·충돌·놓침·발산·폴백) |
| **`domain_gap`** | **실기 로그 재생 시 관측 분포 거리 (§24.3)** |

굵게 표시한 셋은 다른 플랫폼에 없다. 각각 Safety Plane·비동기 추론·sim2real 계층이 있어야 측정 가능하다.

### 10.4 결정성과 공정성

- 모든 perturbation은 `TaskRng(seed_base, suite_id, episode_idx, stream)`에서 샘플링된다. **정책을 바꿔도 같은 perturbation 시퀀스가 나온다**
- 증강 노드(§7.3)는 평가에서 자동 비활성화된다
- `evaluation_hash`가 같으면 같은 조건이다. 리포트에 포함된다
- 평가 실행은 리플레이 가능하다. 실패 에피소드를 되감아 §23의 통합 디버거로 분석한다

### 10.5 산출물

```
es eval run --config eval/pick_cube.yaml --policy policy.esb
  →  report.json        지표 × 스위트 전체
  →  report.html        표 + 실패 에피소드 링크
  →  episodes/          리플레이 (실패분 우선)
  →  evaluation.lock    evaluation_hash + 환경 조건
```

`es eval compare A.json B.json`이 두 정책의 스위트별 차이와 통계적 유의성을 출력한다.

---

## 11. 통합 컴파일러

### 11.1 파이프라인

다섯 IR을 하나의 컴파일러가 처리한다.

```
  Frontend (§14)
       ▼
   Parse          표현식 → IR 서브그래프
       ▼
   Normalize      상수 접기, 데드 노드 제거, SubGraph 인라인, 단위 정규화
       ▼
   Type Check     ElemType · Shape · Unit · Frame · TimeRef · ImageSpec
       ▼
   Cross-IR Check ┌ Task.ObservationSpec  ↔ Observation.inputs
                  ├ Observation.outputs   ↔ Learning.inputs
                  ├ Learning.outputs      ↔ Task.ActionSpec
                  ├ Learning.runtime      ↔ Deployment.rate  (§8.4 LRN-052)
                  └ Deployment.envelope   ↔ Robot capability
       ▼
   Semantic       순환, 미연결, phase 역방향, 리소스 참조
       ▼
   Determinism    §6.6 규칙 (결정적 모드)
       ▼
   Capability     PhysicsBackend · PolicyRuntime · GPU (§11.6)
       ▼
   Canonicalize   위상 정렬 + ID 재번호 → *_hash (§5.3)
       ▼
   Schedule       phase 그룹, 배치 도메인 배정 (§12), 퓨전 경계
       ▼
   Memory Plan    중간 텐서 라이브니스 → 버퍼 재사용 (§20)
       ▼
   Lower ──┬──► CPU 참조 실행 계획      (정답 기준)
           ├──► Slang 커널 + SPIR-V     (시뮬 GPU)
           ├──► PolicyRuntime 호출 계획 (Torch/ONNX/Vulkan)
           ├──► Safety Plane 정적 코드  (결정적, no-std 가능)
           └──► 런타임 명령             (랜덤화·리셋·기록)
       ▼
   compile_hash / execution_hash
```

**`Cross-IR Check`가 v1.0에서 가장 가치 있는 검사다.** 다섯 IR의 경계가 전부 여기서 검증된다. 형상 불일치, 정규화 공간 불일치, 액션 차원 불일치, 추론 지연 초과가 전부 컴파일 시점에 잡힌다.

### 11.2 해시 분리

| 해시 | 입력 | 용도 |
|---|---|---|
| `*_graph_hash` | 노드 ID(사용자 부여) 포함 | 저작 정체성. 디버거 추적, 핫패치 대상 |
| `*_hash` | 정규화 후 재번호 | 의미 정체성. 캐시, 중복 제거, 레지스트리 |
| `compile_hash` | 위 + 컴파일러 버전 + 백엔드 + 정밀도 + SPIR-V 해시 | 실행 아티팩트 정체성 |
| `execution_hash` | §5.3 | 계층 1 재현 조건 |

**정규화 불변식** (property test, 부록 B)
```
hash(canon(g)) == hash(canon(shuffle_ids(g)))
hash(canon(g)) == hash(canon(move_ui(g)))
hash(canon(g)) == hash(canon(deser(ser(g))))
hash(canon(g)) != hash(canon(change_any_param(g)))
```

### 11.3 CPU 참조 실행 계획

**CPU 경로는 성능 경로가 아니라 정답 기준이다.**

- Task IR·Observation IR·Learning IR 전처리 전부를 CPU에서 실행
- GPU lowering의 오라클. 계층 2 검증(§3.5)이 물리뿐 아니라 관측·보상·전처리에도 적용된다
- Slang 없이 동작 (macOS, CI 빠른 게이트)
- 노드별 중간값을 항상 볼 수 있다

### 11.4 GPU lowering

연속된 element-wise 노드를 하나의 커널로 퓨전한다. 퓨전 경계는 리덕션·형상 변경·센서 읽기·디버그 모드 노드 경계.

SPIR-V 캐시 키는 `(ir_hash, 퓨전 그룹, 백엔드, 정밀도)`. 형상은 특수화 상수로 넣어 형상만 다른 태스크가 캐시를 공유한다. `es task compile`이 캐시를 번들에 패키징하면 배포 환경에 Slang이 불필요하다.

### 11.5 릴리스 계획 vs 디버그 계획

커널 퓨전 후에는 노드 경계가 없으므로 노드별 타이밍을 얻을 수 없다. 두 계획을 만든다.

| | 릴리스 | 디버그 |
|---|---|---|
| 퓨전 | 최대 | 노드 경계 유지 |
| 노드별 타이밍 | 없음 | 있음 |
| 노드별 값 통계 | 샘플링 | 전부 |
| 전체 텐서 | 요청 시 선택 노드 | 선택 노드 |
| 성능 | 기준 | 2–5× 느림 |

학습 중에는 릴리스 계획이 돌고 값 통계만 샘플링한다. 특정 환경을 지목해 디버그하면 그 환경만 디버그 계획으로 되감기 재생한다(§23.3).

**보상 항별 분해와 Learning IR 노드별 통계는 예외다.** `Reward` 노드와 인코더 출력은 어차피 값을 내야 하므로 릴리스 계획에서도 나온다.

### 11.6 Capability 검증

```
WARN DEP-114  선택된 런타임에서 지원되지 않습니다

  노드:  VisionEncoder "dinov2_vitb14"
  요구:  attention_flash, dtype=bf16

  PolicyRuntime 별 상태:
    Torch (CUDA)    지원
    Torch (ROCm)    지원 (flash attention 미지원 → 표준 어텐션 폴백)
    ONNX (CPU)      지원 (느림: 예상 지연 2.1 s)
    ONNX (TensorRT) 지원
    Vulkan          미지원 (모델 규모 초과)

  현재 선택: Vulkan → 컴파일 실패
  권장: --policy-runtime onnx 또는 더 작은 백본 (ResNet18, SmolVLM)
```

컴파일 시점에 나와야 한다. 4,096 env 배치가 한참 돌다 죽으면 안 된다.

---

## 12. 배치 도메인과 실행 오케스트레이션

### 12.1 네 도메인

환경 하나를 모든 파이프라인의 기본 단위로 삼는 방식은 상태 기반 RL에는 맞지만 **비전 학습에는 맞지 않는다.**

```
┌─────────────────────────────────────────────────────────────┐
│ SIMULATION BATCH         N_sim = 4,096                      │
│  물리 스텝. dt_phys = 1 ms                                   │
│  백엔드: MJWarp / Newton                                     │
└──────────────────────────┬──────────────────────────────────┘
                           │  활성 카메라 env 선택 (§12.2)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ OBSERVATION BATCH        N_obs = 512 env × 2 view           │
│  렌더 + Observation IR. sensor_dt = 33 ms (30 Hz)            │
│  타일 아틀라스 (§15.2)                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │  추론 배치 구성
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ INFERENCE BATCH          N_inf = 256                        │
│  PolicyRuntime. 비동기. 지연 5–370 ms                        │
│  결과는 청크 버퍼로 (§8.6)                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │  에피소드 기록 → 데이터셋
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ TRAINING BATCH           N_train = 64 sequences             │
│  PyTorch. 별도 프로세스·장치 가능                            │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 도메인 간 정책

| 전이 | 결정 사항 |
|---|---|
| sim → obs | 어느 env에 카메라를 켜는가. `all` / `subset(k)` / `round_robin(k, period)` |
| obs → inf | 추론 배치 크기, 패딩 정책, 대기 시간 상한 |
| inf → sim | 청크 버퍼 관리, 전환 정책(§8.6), 고갈 처리 |
| sim → train | 에피소드 기록 주기, 필터(성공/실패/개입) |

**`round_robin`이 실용적으로 중요하다.** 4,096 env 전부에 카메라를 켜면 VRAM이 터진다(§20). 512개씩 8주기로 순환하면 메모리는 1/8이고 데이터 다양성은 유지된다. 순환 순서는 결정적이다(env ID 기반).

### 12.3 결정성

배치 도메인이 분리되어도 계층 1 재현이 유지되어야 한다.

- 카메라 활성 env 선택은 `(seed, tick, period)`의 결정적 함수
- 추론 배치 구성 순서는 env ID 오름차순
- 비동기 추론의 **도착 시점이 아니라 도착한 청크의 논리적 tick**으로 적용 시점이 결정된다
- 청크 고갈 발생 여부가 리플레이에 기록된다

**비동기 추론에서 결정성을 유지하는 핵심은 "언제 도착했는가"가 아니라 "어느 tick의 관측으로 계산되었는가"로 적용을 결정하는 것이다.** 추론이 예상보다 빨리 끝나도 정해진 tick까지 기다린다(결정적 모드). 실시간 모드에서는 즉시 적용하고 이 차이를 기록한다.

### 12.4 성능 지표

`step/s` 단일 지표를 폐기한다.

| 지표 | 의미 |
|---|---|
| `physics_steps_per_sec` | 물리 백엔드 처리량 |
| `camera_frames_per_sec` | 렌더 프레임 (env × view) |
| `pixels_per_sec` | 해상도 독립 렌더 처리량 |
| `observation_GB_per_sec` | 관측 파이프라인 대역폭 |
| `policy_inferences_per_sec` | 추론 처리량 |
| `actions_per_sec` | 실효 제어 스텝 |
| `p50 / p95 end_to_end_latency` | 관측 캡처 → 액션 출력 |
| `gpu_memory_peak` | VRAM 피크 |
| `chunk_underrun_rate` | 청크 고갈률 |

**`simulation throughput ≠ learning throughput`이다.** 물리가 100k step/s여도 카메라가 30 Hz면 관측은 초당 수천 프레임이고, 정책이 370 ms면 추론은 초당 수 배치다. 병목이 어디인지를 이 표가 드러낸다.

**기준 산정 예** (4,096 env × 2 camera × 224×224 × RGB × 30 Hz)
```
프레임:  4096 × 2 × 30              = 245,760 frames/s
픽셀:    245,760 × 224 × 224        = 12.3 Gpixels/s
대역폭:  12.3G × 3 bytes (RGB8)     = 37 GB/s   ← 렌더 출력만
         + FP32 정규화 후            = 148 GB/s  ← 전처리 중간 텐서
```

**이 계산이 §12.2의 `round_robin`이 필요한 이유다.** RTX 4090의 대역폭이 약 1 TB/s이므로 148 GB/s는 이론상 가능하지만, 물리·렌더·추론이 같은 장치를 공유하면 현실적이지 않다. 활성 카메라를 512 env로 줄이면 18.5 GB/s가 되어 여유가 생긴다.

---

## 13. Learning Loop

### 13.1 1급 워크플로

2026년의 실제 로봇 학습 스택은 반복적이다. LeRobot도 실기에서 정책을 실행하며 사람 개입 데이터를 다시 수집하는 HIL 워크플로를 지원한다.

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
   ┌─────────┐    ┌───────┐    ┌──────────┐    ┌──────────┐
   │ COLLECT │───►│ TRAIN │───►│ EVALUATE │───►│  DEPLOY  │
   └─────────┘    └───────┘    └──────────┘    └────┬─────┘
        ▲                           │                │
        │                           │ 실패 에피소드   │
        │                           ▼                ▼
        │                    ┌─────────────┐   ┌──────────────┐
        └────────────────────│ FAILURE SET │◄──│ INTERVENTION │
                             └─────────────┘   └──────────────┘
```

각 단계는 CLI 명령이자 아티팩트다.

```
es collect   --task T --teleop <device> --episodes N     → dataset/
es train     --task T --obs O --learning L --data D      → checkpoint/ + training.lock
es eval      --config E --policy P                       → report.json + episodes/
es deploy    --policy P --deployment D --target <robot>  → policy.esb
es intervene --session S                                 → dataset/ (개입 라벨 포함)
es distill   --failures F --into D                       → dataset/ (실패 집합 병합)
```

### 13.2 개입 데이터의 지위

개입 에피소드는 일반 데모와 다르게 취급된다.

```
에피소드 메타:
  source:        teleop | policy | policy_with_intervention | scripted
  intervention:  [{ start_tick, end_tick, reason, operator_id }]
  outcome:       success | failure | aborted
  failure_mode:  timeout | collision | miss | diverge | fallback_triggered
```

**개입 구간은 학습 시 가중치를 다르게 줄 수 있고, 그 가중 정책이 `dataset_hash`에 포함된다.** HIL-SERL 계열 워크플로가 이것을 요구한다.

### 13.3 루프의 재현성

각 반복이 `execution_hash` 체인으로 연결된다.

```
iteration 3:
  dataset_hash   = H(iteration 2 데이터 + 새 개입 에피소드)
  policy_hash    = H(base=iteration 2 checkpoint, training.lock)
  evaluation_hash= (고정)
  →  "정책이 나아졌는가"를 같은 평가 조건에서 판정 가능
```

**평가 조건을 고정한 채 데이터·정책만 바꾸는 것이 규율이다.** `evaluation_hash`가 바뀌면 비교가 무효이고, 도구가 이를 경고한다.

---

## 14. 저작 프론트엔드

### 14.1 원칙

다섯 프론트엔드가 동일한 canonical IR을 생성한다. **동등성은 CI가 검증한다.**

```
es run task.toml       ┐
es run task.esgraph    ├──►  동일 *_hash  ──►  동일 실행
python train.py        ┘
```

에디터 없이 모든 것을 실행할 수 있다.

### 14.2 Python 빌더

```python
from electric_sheep import Task, Observation, Learning, Deployment, math as m

# ── Task IR ──────────────────────────────────────────────
task = Task("pick_cube", scene="scenes/table_franka.usda")
arm  = task.robot("franka")

task.observe_spec(
    rgb_front   = task.sensor("cam_front").rgb(),
    rgb_wrist   = task.sensor("cam_wrist").rgb(),
    joint_state = arm.joint_positions(),
)
task.action_spec(space="joint_position", robot=arm, limits="from_scene")

task.reward("reach", weight=1.0,
            value=-m.norm(task.body("cube").pose().pos
                          - arm.link("hand").pose().pos))
task.reward("grasp", weight=5.0,
            value=task.contact(arm.link("finger"), "cube").force.sum() > 1.0)
task.terminate("success",
               when=m.norm(task.body("cube").pose().pos
                           - task.site("target").pos) < 0.02)
task.randomize("cube_pose", stream="init",
               target=task.body("cube").initial_pose,
               dist=("uniform_se2", [-0.1, 0.1], [-0.1, 0.1], [-3.14, 3.14]))

# ── Observation IR ───────────────────────────────────────
obs = Observation("two_view_224", task=task)
obs.image("rgb_front").resize(224, 224).to_linear().normalize("imagenet")
obs.image("rgb_wrist").resize(224, 224).to_linear().normalize("imagenet")
obs.state("joint_state").normalize("joint_limits")
obs.temporal_window(n_steps=1)          # ACT 기본
obs.time_align("hold")

# ── Learning IR ──────────────────────────────────────────
lrn = Learning("act_r18", observation=obs)
lrn.vision_encoder("resnet18", pretrained="imagenet", frozen=False, shared=True)
lrn.state_encoder("mlp", out_dim=512)
lrn.fusion("concat")
lrn.temporal_encoder("none")
lrn.head("regression", horizon=50)
lrn.chunker(execute_chunk=20, replanning_hz=10, mode="temporal_ensemble")

# ── Deployment IR ────────────────────────────────────────
dep = Deployment("franka_safe", robot=arm, action=task.action_spec_ref())
dep.envelope(torque_limit="from_urdf", velocity_limit=1.5,
             workspace=dep.box([0.2, -0.4, 0.0], [0.8, 0.4, 0.6]),
             rate_limit=2.0)
dep.watchdog("inference_deadline", budget_ms=50)
dep.watchdog("chunk_underrun")
dep.watchdog("stale_observation", max_age_ms=100)
dep.fallback("hold_position")

Task.save_bundle("tasks/pick_cube/", task, obs, lrn, dep)
```

**빌더는 즉시 실행하지 않는다.** `save_bundle` 또는 `compile` 시점에 IR을 구성하고 Cross-IR Check(§11.1)를 돌린다. 타입 오류는 Python 예외로 올라오되 메시지는 컴파일러와 동일한 진단 형식이다.

**Python이 IR을 넘어서면(임의 콜백 등) 명시적 에러를 낸다.** "Python이면 뭐든 된다"를 허용하면 IR의 보증이 무너진다.

### 14.3 저장 포맷

| 포맷 | 용도 |
|---|---|
| `task.toml` / `observation.toml` / `learning.toml` / `deployment.toml` / `evaluation.yaml` | 사람이 읽고 git diff |
| `*.esgraph` | 그래프 에디터 저장 (명시적 노드·엣지, 안정 ID) |
| `*.eslayout` | 편집기 메타데이터 사이드카. **IR에 들어가지 않는다** |
| `bundle.eslock` | 다섯 IR의 해시 + 참조 고정 |

### 14.4 외부 변환

```
LeRobot 정책 config    ┐
Isaac Lab task config  │
MJCF 센서·액추에이터    ├──► Converter ──► Semantic Mapping Report ──► IR
RoboVerse / MetaSim    │                        │
Gymnasium spec         ┘                        └─► severity=error → 실행 차단
```

**LeRobot 변환이 v1.0의 최우선 변환 타깃이다.** `lerobot/act_*`, `lerobot/smolvla_base`, `lerobot/pi05_base` 설정을 Learning IR로 읽어들이면, 기존 사용자가 자기 정책을 그대로 가져와 Electric Sheep의 평가·안전·재현 계층을 얹을 수 있다. **이것이 채택의 가장 낮은 문턱이다.**

### 14.5 LLM 생성

```
es generate task --from-description "큐브를 집어 목표 지점에 놓는다" \
                 --scene scenes/table_franka.usda --candidates 16
es generate reward --task T --candidates 16 --optimize success_rate
```

타입 있는 IR은 LLM 생성 타깃으로 Python 코드보다 우월하다. 검증기가 구조화된 진단을 즉시 주고, 비결정적·안전 위반 구성을 표현할 수 없고, GPU로 컴파일되고, `ir_hash`로 중복 후보를 학습 전에 제거한다. Eureka류 방법의 비용이 "후보당 수천 gradient step"이므로 중복 제거가 직접 절감이 된다.

진화 루프 오케스트레이터는 만들지 않는다(§0.5). 기존 도구가 Electric Sheep를 백엔드로 쓰게 한다. MCP 인터페이스로 `validate` / `compile` / `estimate_cost` / `eval`을 노출한다(M4).

---

## 15. Vision Data Plane

### 15.1 렌더 → 관측 경로

렌더링과 학습 입력 사이에 명시적 계층을 둔다.

```
Camera Renderer (§15.3)
      ↓  RGB / Depth / Seg / Normal / Flow / Velocity   (동일 텐서 계약)
Sensor Realism (§18.3)
      ↓  렌즈 왜곡 · 롤링 셔터 · 모션 블러 · 노출 · 샷 노이즈 · 깊이 홀
Observation Capture
      ↓  GPU 상주. ImageSpec 부착
Observation IR (§7)
      ↓  resize · crop · color · normalize · temporal · mask
Learning IR 전처리 (§8.7)
      ↓
Vision Encoder → Policy
```

**핵심: 이 전 구간이 GPU에 상주하고 호스트 왕복이 없다.** 그리고 `ImageSpec`이 각 단계에서 변환되며 전파되므로, 인코더 입력 시점에도 원본 카메라 파라미터를 역추적할 수 있다.

### 15.2 대량 병렬 카메라: 타일 아틀라스

`maxMultiviewViewCount`는 주요 데스크톱 GPU에서 32다. 멀티뷰로는 수백 개 카메라를 처리할 수 없다.

- 모든 카메라를 하나의 프레임버퍼에 타일로 배치하고 단일 렌더 패스
- 각 카메라가 자기 intrinsic·pose 유지. 타일링 레이아웃이 결정적이므로 호스트 전송 없이 환경별 텐서로 재구성
- 뷰 인덱스는 draw payload. 컬링 컴퓨트가 `(view, instance)` 쌍 컴팩션
- 제약: `maxImageDimension2D`(통상 16384). 224×224 타일이면 단일 아틀라스에 **5,184개**

**기준 수치:** 224×224 RGB 타일 512개 = 5,376×5,376 아틀라스. RGB8로 87 MB, FP32 정규화 후 347 MB, 더블버퍼 시 2배. §20에 계상한다.

### 15.3 렌더 경로

| 경로 | 조건 | 용도 |
|---|---|---|
| **RS** 래스터 | 항상 | **학습 관측 기본값**, 뷰어, MoltenVK |
| **PT** 경로추적 | RT 확장 | 사실적 데이터셋, 도메인 갭 실험, 골든 |
| **3DGS** 스플랫 | §16 | real-to-sim 재구성 씬 |

**출력 계약:** 세 경로 모두 동일 채널(RGB, 선형 깊이, 인스턴스/시맨틱 세그, 월드 노멀, 옵티컬 플로우, 속도)을 동일 텐서 레이아웃으로. 깊이·세그·노멀은 RS/PT가 비트 동일, RGB는 SSIM 임계.

PT는 M4이며 §1.9 축소 순서 2번이다. **RS만으로 비전 학습이 성립한다**는 것이 설계 전제다.

### 15.4 가속 구조

전체 씬에 TLAS 1개. 동일 로봇 N env는 BLAS 1개 공유. 환경 간 광선 격리는 인스턴스 mask(8비트)로 불가능하므로 **공간 분리 배치 + 광선 `tMax` 제한**이 1차, `instanceCustomIndex`(24비트) any-hit 필터가 2차(기본 비활성).

---

## 16. Real-to-Sim: 3D Gaussian Splatting

### 16.1 왜 필요한가

**비전 정책의 sim2real 병목은 물리 정확도가 아니라 외관이다.** 비전 기반 매니퓰레이션에서 정책이 실패하는 주된 이유는 시뮬 이미지가 실기 카메라 이미지의 분포 밖에 있기 때문이다.

2025–2026에 3DGS 기반 real-to-sim이 이 문제의 실용적 해법으로 자리 잡았다.

| 연구 | 기여 |
|---|---|
| RL-GSBridge | 3DGS 기반 real-to-sim-to-real RL, 비전 제어 zero-shot 전이 |
| Real-is-Sim (Embodied Gaussians) | 수집·학습·평가·배포 전 구간에 동적 디지털 트윈 |
| RoboGSim | 대화형 real2sim2real 플랫폼, 데모 합성·신규 씬/물체 확장·폐루프 평가 |
| Real-to-Sim Policy Eval (2511.04665) | 스마트폰 스캔 → 3DGS, 로봇/물체/배경 분할, **위치·색 정렬**, PhysTwin 소프트바디. **시뮬 롤아웃이 실기 성능과 r > 0.9 상관** |
| RialTo | 소량 실기 데이터로 디지털 트윈 구성 후 시뮬 RL로 정책 강건화 |
| ReaDy-Go | 동적 인간 GS 아바타를 정적 씬에 합성해 내비게이션 정책 학습 |

마지막에서 두 번째가 특히 중요하다. **색 정렬**(초기 스캔의 색공간을 로봇의 실제 카메라(예: RealSense) 색공간으로 다항식 매핑)이 정책이 in-distribution 이미지를 보게 만드는 결정적 단계였다.

### 16.2 Electric Sheep에서의 위치

3DGS는 별도 기능이 아니라 **렌더 경로 하나이자 `ImageSpec` 생산자**다.

```
스마트폰/카메라 스캔
      ↓
COLMAP 또는 유사 SfM  (외부 도구)
      ↓
3DGS 학습             (외부 도구 또는 es-splat)
      ↓
es asset import-splat --scan scan/ --segment robot,objects,background
      ↓
SceneDesc + SplatAsset
  ├── background  정적 스플랫
  ├── objects     물체별 스플랫 + 물리 프록시 (볼록 분해 또는 SDF)
  └── robot       URDF 링크에 바인딩된 스플랫
      ↓
es-render 스플랫 경로 (Vulkan 컴퓨트 래스터화)
      ↓
ImageSpec (실기 카메라 파라미터로 설정)
      ↓
Observation IR  ← 시뮬과 실기가 동일
```

**핵심 설계 결정 셋**

1. **스플랫은 물리를 대체하지 않는다.** 물체마다 물리 프록시(볼록 분해 메시 또는 SDF)를 별도로 갖고, 스플랫은 그 프록시의 변환을 따라간다(Linear Blend Skinning). 물리는 §17 백엔드가 담당한다
2. **색 정렬이 파이프라인의 일부다.** `es asset align-color --scan S --reference <실기 캡처>`가 다항식 매핑을 추정해 `SplatAsset`에 저장한다. 이 매핑이 `scene_hash`에 들어간다
3. **위치 정렬도 파이프라인의 일부다.** ICP + RANSAC으로 스캔 좌표계를 로봇 베이스 프레임에 맞춘다. 결과 `T_robot_scan`이 자산 메타에 기록된다

### 16.3 산출물

```
es-splat 크레이트 (레이어 5)
  ├── .ply / .splat / .spz 로더
  ├── Vulkan 컴퓨트 래스터화 (타일 정렬 + 알파 블렌딩)
  ├── LBS 바인딩 (링크 트랜스폼 → 스플랫 변환)
  ├── 색·위치 정렬
  └── 세그멘테이션 메타 (robot / object / background)
```

**정확한 정렬은 불필요하고, 분포 일치가 목표다.** 스플랫 렌더가 실기 카메라 이미지와 같은 통계·주파수 특성을 가지면 충분하다. 이것을 §10의 `domain_gap` 지표로 측정한다.

### 16.4 시장 의미

이 경로가 있으면 사용자의 워크플로가 이렇게 된다.

```
1. 실제 작업 공간을 휴대폰으로 스캔               (10분)
2. es asset import-splat + align                 (자동)
3. 실기 데모 20–50개 수집                         (1시간)
4. 시뮬에서 데이터 확장 + 정책 학습               (자동)
5. es eval 로 perturbation suite 평가 (§10)      (자동)
6. Safety Plane 포함 policy.esb 배포 (§9.6)      (자동)
7. 실기 개입 데이터로 재학습 (§13)                (반복)
```

**"CAD 없이, 시뮬 전문가 없이, 자기 작업 공간에서 비전 정책을 만든다."** 이것이 현재 어떤 플랫폼도 end-to-end로 제공하지 않는 워크플로이고, §27의 주된 시장 메시지다.

일정: **M3.** §1.9 축소 순서 3번이며, 잘라도 제품이 성립한다(외부 3DGS 도구 + 수동 임포트로 대체 가능).

---

## 17. 물리 (백엔드 계층)

### 17.1 역할 전환

**v1.0에서 물리는 제품이 아니라 백엔드다.** §4.3 참조.

```
PhysicsBackend
├── MuJoCoWarpBackend    기본. MJCF 네이티브, GPU 병렬          M1
├── MuJoCoCpuBackend     결정적 참조, CI 오라클                  M1
├── NewtonBackend        Kamino·VBD·hydroelastic 필요 시        M2
├── PhysXBackend         Isaac Lab 자산 호환                     M3
└── NativeBackend        자체 CPU/Vulkan 솔버                    M4+ (선택)
```

**M1에서 자체 솔버를 쓰지 않는다.** 이것이 가장 큰 일정 절감이며, 임계 경로에서 유형 D 작업(§1.3)의 대부분을 제거한다.

### 17.2 백엔드 의미 매핑 (새로운 핵심 작업)

같은 Task IR이 백엔드마다 다르게 동작하면 안 된다. 물리 엔진을 만들지 않는 대신 **의미 매핑과 그 검증**이 핵심 작업이 된다.

```
Task IR                         MJWarp          Newton          PhysX
─────────────────────────────────────────────────────────────────────
actuator.pd(kp, kd)             position gain   controller      drive stiffness
contact.friction_cone           pyramidal       선택 가능        pyramidal
contact.soft_params             impedance       solver-dependent contact offset
joint.armature                  armature        armature        미지원 → 경고
sensor.contact_force            sensor          contact          contact report
```

`es backend compare --task T --backends mjwarp,newton,mujoco-cpu`가 동일 태스크를 세 백엔드에서 실행하고 §3.5 계층 3 지표로 비교한 리포트를 낸다. **미매핑 항목은 `severity: error`면 실행 차단**(§14.4).

### 17.3 결정성

- `MuJoCoCpuBackend`만 계층 1(bitwise)을 선언한다
- GPU 백엔드는 계층 2·3만 선언한다. **이것이 정직한 상태다**
- 계층 1이 필요한 워크플로(증거물 생성, 회귀 재현)는 CPU 백엔드로 실행한다. 16 env를 100% 신뢰하는 것이 4,096 env를 대충 재현하는 것보다 가치 있다(§27.1)

### 17.4 자체 솔버 (M4+, 선택)

만든다면 유지되는 설계 결정. **지금 구현하지 않는다.**

- TGS + 서브스텝, 리듀스드 좌표(ABA), 소프트 접촉(MuJoCo impedance 호환)
- 고자유도 GPU는 Delassus 전체 조립을 배제하고 Matrix-Free CG
- 접촉 버킷 스케줄링: 접촉 수를 2의 거듭제곱 버킷으로 분류, 안정 정렬 후 버킷별 indirect dispatch
- 결정적 리덕션: RFA/binned summation (§18.4)
- 성능 목표는 **검증 전까지 목표치로만 기술한다.** "평균 25회 → 6–8회" 같은 수치를 사실로 쓰지 않는다

```
Target:  CG iterations ↓ 3–5×
Success criteria:  median / p95 / worst-case iterations, 4k env 기준
Status:  미검증 (M4 게이트)
```

---

## 18. 시간·센서·액추에이터·실패

### 18.1 시간 모델

- `dt_phys` (기본 1 ms), `dt_ctrl` (기본 50–100 ms, 비전 정책 기준), 센서별 독립 주기
- 모든 스텝은 정수 틱. 부동소수 시간 누적 금지, 타입으로 강제
- 모든 센서 샘플은 물리 틱 번호를 동반한다
- 관측 지연·액션 지연을 링버퍼로 모델링하고 도메인 랜덤화 대상

**`dt_ctrl` 기본값은 50–100 ms다.** 비전 정책의 추론 지연(§0.3)이 현실적 제어 주기를 결정한다.

### 18.2 액추에이터

이상 토크/위치/속도, PD + 피드포워드, DC 모터 전기역학, 기어·백래시, 관절 탄성(Two-mass), 전류 루프 유효 모델, 통신 지연·양자화, 버스 지터/패킷 드롭, actuator net(ONNX), 그리퍼·흡착.

백엔드가 지원하는 범위 내에서 매핑하고, 미지원 항목은 §17.2 리포트에 기록한다.

### 18.3 센서와 현실성

| 센서 | 노이즈 모델 |
|---|---|
| RGB/깊이 카메라 | 가우시안·샷 노이즈, 깊이 홀·플라잉 픽셀, **렌즈 왜곡(OpenCV 규약)**, **롤링 셔터**, 모션 블러, 노출, 화이트밸런스 |
| 스테레오 | 시차 오차 |
| IMU | 바이어스, 랜덤워크, 스케일·정렬 오차 |
| 관절 엔코더 | 양자화, 지연, 오프셋 |
| F/T | 가우시안, 온도 드리프트 |
| 라이다 | 거리 노이즈, 드롭아웃, 다중 반사 |
| 촉각 | 접촉 매니폴드 → 압력 이미지 |
| 이벤트 카메라 | 로그 강도 차분, 임계 노이즈, 리프랙토리 |

**카메라 노이즈 파라미터는 `ImageSpec`(§7.2)에 반영되어 Observation IR로 전달된다.** 이것이 v1.0에서 메운 공백이다. 왜곡 계수를 알면 `Undistort` 노드를 쓸지 말지 IR 수준에서 결정할 수 있다.

전 파라미터가 도메인 랜덤화 대상이고, Evaluation IR의 perturbation과 같은 파라미터 공간을 쓴다(§10.2).

### 18.4 결정적 리덕션

Demmel & Nguyen 계열 RFA / binned summation. K개 지수 빈(기본 3), 1회 읽기, 병렬 리덕션 1회. **핵심 불변식은 `merge`가 결합적·교환적이라는 것**이고, 이것이 성립하면 서브그룹 순서·워크그룹 병합 순서·SM 개수가 달라져도 결과가 같다.

NVIDIA CCCL/CUB가 동일 원리의 `gpu_to_gpu` 결정성 레벨을 제공하며 대형 문제에서 20–30% 증가를 보고했다. **§26의 "결정적 모드 저하 ≤ 30%" 목표가 독립 검증 수치와 일치한다.**

### 18.5 실패 시맨틱

```
EnvHealth = Ok | Diverged | Poisoned | Quarantined
```

격리된 env는 리덕션에 참여하지 않는다(누산기가 순서 독립이라 마스킹이 결정성을 깨지 않는다). 격리는 리플레이에 기록되고, 격리율이 임계(기본 5%)를 넘으면 학습을 중단한다.

**v1.0 추가: `EnvHealth`가 Safety Plane 사건과 연결된다.** 폴백이 발동한 env는 `Quarantined`가 아니라 `Ok`로 남되 사건이 기록된다. 폴백은 정상 동작이지 실패가 아니다.

---

## 19. 데이터셋과 아이덴티티

### 19.1 포맷

**LeRobot 호환이 1차다.** Parquet + 비디오 청크. LeRobotDataset v3의 multi-episode packing과 스트리밍을 지원한다. RLDS/TFDS, HDF5(robomimic) 익스포트가 2차.

에피소드 구성:
```
관측    멀티 카메라 비디오 + 저차원 sensorimotor + 물리 틱 정렬
액션    청크 단위 기록 (예측 horizon 전체 + 실행 구간 표시)
보상    항별 분해
메타    scene_hash, task_hash, observation_hash, randomization sample,
        규약 버전, 백엔드, 격리 로그, ImageSpec, 개입 라벨 (§13.2)
```

**`ImageSpec`을 데이터셋에 저장하는 것이 중요하다.** 나중에 다른 Observation IR로 재처리할 수 있고, 여러 출처 데이터를 합칠 때 카메라 파라미터가 다른 것을 알 수 있다.

### 19.2 Dataset Identity

```
dataset_content_hash   실제 샘플 내용
dataset_schema_hash    필드 구성, ImageSpec, 액션 공간
dataset_split_hash     train / val / test 분할 정의
dataset_hash = H(content, schema, split)
```

**분할을 해시하지 않으면 "같은 데이터셋"이라도 학습 샘플 구성이 달라진다.** 분할은 결정적 함수여야 한다.

```yaml
split:
  method: by_episode_hash          # by_index | by_episode_hash | explicit
  seed: 20260912
  ratios: { train: 0.8, val: 0.1, test: 0.1 }
  stratify_by: [task_variant, outcome]
  holdout:                         # 명시적 일반화 홀드아웃
    objects: ["mug_07", "bowl_03"]
    lighting: ["evening"]
```

`holdout`이 §10의 일반화 스위트와 연결된다. 학습에서 본 적 없는 물체·조명으로 평가하는 것이 IR 수준에서 보장된다.

### 19.3 Training Identity

```
training/
├── config.json         하이퍼파라미터 전체
├── optimizer.json      옵티마이저 + 파라미터 그룹
├── scheduler.json      LR 스케줄
├── seed.json           전역 시드 + 데이터로더 시드 + 증강 시드
├── dataset.lock        dataset_hash (content/schema/split)
├── base_model.lock     사전학습 백본 출처 + 해시 + 라이선스
├── augmentation.json   §7.3 증강 노드 구성
├── precision.json      fp32/bf16/fp16, gradient accumulation
├── topology.json       분산 구성 (world size, 병렬 방식)
├── checkpoint.manifest 체크포인트별 step·지표·해시
├── metrics.parquet     학습 곡선
└── hardware.json       GPU 모델·드라이버·라이브러리 버전

training_hash = H(위 전부)
policy_hash   = H(training_hash, checkpoint_hash)
```

**`base_model.lock`이 특히 중요하다.** π₀는 10,000시간 크로스 엠바디먼트 데이터로 사전학습되었고 SmolVLA는 커뮤니티 데이터 30,000 GPU-시간으로 학습되었다. **사전학습 백본의 출처가 결과에 직접 영향을 미치므로 provenance에 반드시 들어간다.** 라이선스 추적의 근거이기도 하다.

---

## 20. 메모리·대역폭 예산

### 20.1 참고 실측

Isaac Lab (RTX 4090): Cartpole 물리만 4,096 env = 3.3 GB / **Cartpole RGB 카메라 1,024 env = 16.7 GB** / G1 locomotion 4,096 env = 6.1 GB.

**비전 태스크만 env가 1/4인데 VRAM은 5배다.** 이것이 §12.2 `round_robin`의 근거다.

### 20.2 예산 모델

| 항목 | 산정 |
|---|---|
| 물리 상태 (백엔드 소유) | 백엔드가 보고 |
| **렌더 타일 아틀라스** | `N_obs × views × H × W × ch × bytes × 2(더블버퍼)` |
| **Observation IR 중간 텐서** | §11.1 라이브니스 분석 |
| **정책 가중치** | ACT 52M×4B=208MB / SmolVLA 450M=1.8GB / π₀ 3.5B=14GB (fp32) |
| **추론 활성화** | `N_inf × 모델별 피크` |
| **청크 버퍼** | `N_sim × horizon × action_dim × 4B × 2` |
| 스냅샷 링버퍼 (되감기) | 상태 크기 × 스냅샷 수 |
| 3DGS 스플랫 (§16) | 가우시안 수 × 59 floats (SH degree 3) |
| 데이터셋 기록 버퍼 | 압축 전 스테이징 |

**계산 예** (512 obs env × 2 view × 224×224 RGB, ACT, N_inf=256)
```
아틀라스 RGB8 더블버퍼      512×2×224×224×3×2        =  308 MB
정규화 FP32 중간 텐서        512×2×224×224×3×4        =  616 MB
정책 가중치 (ACT 52M fp32)                            =  208 MB
추론 활성화 (배치 256)       측정 필요, 추정           = ~2.0 GB
청크 버퍼 4096 env           4096×50×8×4×2            =   13 MB
──────────────────────────────────────────────────────────────
비전·정책 소계                                        ≈  3.1 GB
+ 물리 백엔드 (MJWarp 4096 env)                       ≈  2–4 GB
+ 학습이 같은 GPU면 PyTorch                            ≈  8–16 GB
```

### 20.3 규칙

- 씬·태스크·정책 로드 시 예산을 계산하고 실제 메모리와 대조. **초과 시 `N_obs`·`N_inf`를 자동 축소하거나 명시적으로 실패.** OOM으로 죽지 않는다
- `es bench --memory-report`가 항목별 실측 출력
- 에디터가 편집 중 실시간 표시. **예산 초과는 컴파일 에러**(§11.1 Memory Plan)
- 성능 리포트에 VRAM 피크 병기 (§12.4)

---

## 21. TensorTransport

### 21.1 Zero-copy는 약속이 아니라 capability

제로카피는 **AMD/Intel/ARM에서 항상 가능하지 않으므로 제품 약속이 아니라 capability 협상 개념으로 다룬다.**

```rust
pub enum TensorTransport {
    VulkanShared,        // 같은 VkDevice 내 공유. 항상 가능
    CudaExternal,        // VK_KHR_external_memory + cuImportExternalMemory
    HipExternal,         // hipImportExternalMemory
    DlPack,              // 프레임워크 간 표준, 스트림 의미 포함
    HostPinnedFallback,  // 핀드 호스트 링버퍼. 항상 가능
}

pub struct TransportNegotiation {
    pub preferred: Vec<TensorTransport>,
    pub selected: TensorTransport,      // 런타임 결정
    pub reason: String,                 // 왜 그것이 선택되었는가
}
```

**선택 결과가 `runtime_hash`와 성능 리포트에 기록된다.** "이 실행은 호스트 복사 폴백을 썼다"가 명시적으로 드러난다.

### 21.2 규약

- 더블버퍼 슬롯 소유권을 타입 상태(`Slot<SimOwned>` / `Slot<LearnerOwned>`)로 강제
- `publish` / `release`는 호스트 대기 없이 타임라인 값만 넘긴다
- DLPack은 `__dlpack__(stream=...)` 프로토콜 구현
- **호스트 동기화 카운터:** 디버그 빌드에서 `vkWaitSemaphores` / `cudaStreamSynchronize` 호출을 세고 핫패스에서 0이 아니면 패닉
- 디바이스 UUID 검증. 열거 순서 매칭 금지
- 플랫폼 핸들: Linux `OPAQUE_FD`, Windows `OPAQUE_WIN32`
- **Intel(SYCL/XPU)은 실험적이므로 `HostPinnedFallback` 명시**

---

## 22. 멀티 GPU

환경 샤딩만 지원. 집합 통신은 프레임워크에 위임. rank = 프로세스 = GPU = 환경 샤드.

**v1.0 추가: 역할 분리가 1급이다.**

```
role = sim | obs | infer | train | all

예) 4 GPU 구성
  GPU0  sim   (MJWarp 4096 env)
  GPU1  obs   (렌더 + Observation IR, 512 env × 2 view)
  GPU2  infer (PolicyRuntime, π₀ 3.5B)
  GPU3  train (PyTorch)
```

§12의 배치 도메인이 장치 경계와 자연스럽게 맞는다. 도메인 간 전송은 §21 `TensorTransport` 협상을 따른다.

모든 rank는 동일 `execution_hash`를 가져야 한다. 시작 시 rank 0이 컴파일하고 브로드캐스트하며 불일치 rank는 즉시 종료한다. SPIR-V 캐시는 콘텐츠 주소라 공유 가능하다.

`seed_shard = hash(seed_global, rank)`. `TaskRng`의 `EnvId`는 전역이다. `WORLD_SIZE` 변경 시 계층 1 재현 불가 — 리플레이에 기록·경고.

---

## 23. 에디터와 통합 디버거

### 23.1 원칙

**에디터는 학습을 호스팅하지 않는다. 실행 중인 프로세스에 접속하는 클라이언트다.** §4.2 규칙 4로 강제.

**백엔드 중립:** 텔레메트리 프로토콜은 Electric Sheep 전용이 아니다. 얇은 Python 어댑터로 LeRobot·Isaac Lab·Newton 학습에 붙는 것을 M1 산출물에 포함한다. 이것이 채택의 쐐기다.

### 23.2 계층 그래프 뷰

**전 계층을 하나의 화면에서 본다.**

```
Scene Graph
   │
Task Graph            보상 항, 종료 조건
   │
Observation Graph     ImageInput → Resize → Normalize → TemporalWindow
   │
Learning Graph        VisionEncoder → Fusion → Head
   │
Action Graph          Chunker → Unnormalizer
   │
Safety Graph          Envelope → Watchdog → Fallback
```

**이것이 진짜 차별화 기능이다.** 사용자가 "왜 grasp 성공률이 떨어졌지?"라고 물으면:

```
reward.grasp (0.12 ↓)
   ↑
action quality — 액션이 목표를 지나침
   ↑
policy output — 청크 후반부 분산 증가
   ↑
vision encoder — 특징 노름이 평소의 0.3배
   ↑
Normalize — 입력 평균이 분포 밖
   ↑
camera_front — 노출 시간이 랜덤화 상한에 붙어 있음
```

한 화면에서 보상부터 카메라 노출까지 역추적할 수 있다. 이 체인은 Observation IR과 Learning IR이 IR로 존재해야만 가능하다.

### 23.3 학습 중 확인

- 상태 스트리밍(프레임 아님). 에디터가 씬을 로컬 복제하고 선택 env의 포즈·관절·접촉만 받아 로컬 렌더
- 예산제 샘플링(스텝당 기본 20 µs). 관측 이미지 요청은 별도 rate limit
- 보이는 것: 3D 뷰, **실제 관측 이미지(전처리 전·후 동시)**, 보상 항별 분해, **Learning IR 노드별 통계**, **Safety Plane 사건 로그**, **청크 버퍼 상태와 추론 지연 히스토그램**, `EnvHealth` 분포, rank별 처리량·VRAM
- 조작: 일시정지·단일 스텝, 리셋, 핫패치(허용 파라미터만), 되감기

**"전처리 전·후 이미지 동시 표시"가 실무적으로 강력하다.** 정책이 실제로 보는 것과 렌더된 것의 차이를 눈으로 확인하는 것이 비전 디버깅의 절반이다.

### 23.4 그래프 기능 3단계

| 단계 | 기능 | 일정 |
|---|---|---|
| 읽기 전용 뷰 | 자동 레이아웃, 노드별 실시간 값, 역추적 | **M1** |
| 편집 | 노드 추가·연결, 파라미터 편집, 실시간 검증 | M3 |
| Control Graph | IR-C 편집 | M4 |

읽기 전용부터 시작한다. 편집기에는 레이아웃 영속화·undo/redo·검색·대형 그래프 성능이 전부 필요하지만 읽기 전용에는 하나도 필요 없고, **디버깅 가치의 대부분은 읽기 전용으로 나온다.**

기술: egui + winit, 노드 그래프는 `egui-snarl`, 원격은 QUIC.

**성능 게이트: 텔레메트리 + 그래프 뷰 활성화로 인한 학습 저하 < 1%.** M1 필수.

---

## 24. Sim-to-Real

### 24.1 ROS 2 경계

`rmw_zenoh`와 `zenoh-plugin-ros2dds`는 key expression 체계가 달라 상호운용되지 않는다. `rmw_zenoh_cpp`는 Kilted(2025-05)부터 Tier 1이고 Lyrical(2026-05)에서 전 플랫폼 Tier 1이다.

| 모드 | 대상 | 구현 |
|---|---|---|
| **A. rmw_zenoh 네이티브** (기본) | ROS 2 Kilted+ | `zenoh-rs`로 key expression·CDR·attachment·liveliness 구현 |
| B. DDS 브리지 | 기존 DDS | `zenoh-bridge-ros2dds` 별도 프로세스 |
| C. 순수 Rust DDS (실험) | DDS 직결 | `RustDDS` + `ros2-client` |

A와 B는 동시에 켤 수 없다. 설정에서 상호 배타로 강제하고 시작 시 liveliness로 검증한다.

### 24.2 Hardware-in-the-Loop

실제 컨트롤러를 시뮬에 연결. ROS 2 또는 저지연 UDP(`es-hil`). 데드라인 미스·지터를 텔레메트리로 노출.

**HIL 모드에서 계층 1 결정성을 보증하지 않는다.** 외부 하드웨어가 비결정적이다. 대신 입력 로그를 기록해 사후 재생을 가능하게 한다.

**v1.0 추가: HIL에서 Safety Plane이 실기와 동일하게 동작하는지 검증하는 것이 M3 게이트다.** 시뮬에서만 켜지는 안전 장치는 의미가 없다.

### 24.3 도메인 갭 진단

```
es domain-gap --real <rosbag|dataset> --sim <scene+task+obs>
```

실기 로그를 시뮬에 재생하고 관측 분포를 비교한다.

| 비교 | 지표 |
|---|---|
| 이미지 | 채널 히스토그램, 주파수 스펙트럼, FID 계열, 인코더 특징 거리 |
| 상태 | 관절 궤적 오차, 속도 분포 |
| 접촉 | 접촉 이벤트 타이밍·힘 분포 |
| 정책 응답 | **같은 관측에 대한 액션 차이** |

마지막이 가장 실용적이다. **정책이 sim과 real 이미지에 다르게 반응하면 그것이 갭의 정의다.** §16의 3DGS 색 정렬은 이 지표를 줄이는 것이 목적이다.

---

## 25. 보안·거버넌스·버저닝

### 25.1 보안

- 텔레메트리: 토큰 인증, QUIC TLS, 기본 로컬호스트 바인딩
- 자산 파서 퍼징: USD/MJCF/URDF/glTF/**3DGS ply**
- **IR 파서·검증기 퍼징.** 그래프 파일이 신뢰 경계를 넘는 입력이다
- **Task/Observation/Deployment IR은 스크립트 주입 표면이 아니다.** 임의 네이티브 호출·파일·네트워크 노드가 존재하지 않는다
- **정책 가중치는 신뢰 경계를 넘는 입력이다.** `safetensors` 우선, pickle 기반 포맷은 경고. 모델 해시 검증
- `CustomKernel`(Slang) 런타임 컴파일은 리소스 화이트리스트 + 시간·자원 상한
- 네이티브 플러그인 서명 검증(선택)

### 25.2 법무 확인 항목

| 항목 | 상태 |
|---|---|
| MJCF 포맷 파싱 | 문제없음 |
| MuJoCo Menagerie 자산 재배포 | **모델별 상이. 개별 확인** |
| **사전학습 백본 라이선스** (π₀, SmolVLA, GR00T, DINOv2, SigLIP) | **모델별 상이. `base_model.lock`에 기록하고 배포 시 검증** |
| LeRobot 데이터셋·체크포인트 재배포 | **확인 필요** |
| Isaac Lab / RoboVerse 변환 산출물의 파생 저작물 여부 | **확인 필요** |
| 3DGS 스캔 데이터의 사생활·초상권 | **사용자 스캔에 사람이 포함될 수 있다. 정책 수립 필요** |
| Newton / MJWarp / USD (Apache-2.0) | 문제없음 |

**사전학습 백본 라이선스가 v1.0에서 새로 생긴 실질 리스크다.** 정책 번들(§9.6)에 백본이 들어가면 배포 시 라이선스가 따라간다. `es deploy`가 이를 검사하고 경고한다.

### 25.3 API 버저닝

| 대상 | 정책 |
|---|---|
| **IR 스키마 5종** | 스키마 버전 정수. 상위 버전 읽기 거부. 마이그레이션 도구 필수 |
| **내장 노드 타입 ID** | M1부터 동결. 필드 추가만 허용 |
| `policy.esb` 번들 포맷 | 포맷 버전. 구 버전 읽기 지원 |
| Python API | SemVer. 1.0 이전 불안정 |
| 플러그인 C ABI | ABI 버전 정수. M3부터 고정 |
| 텔레메트리 프로토콜 | 핸드셰이크 협상, N-1까지 지원 |

**IR 스키마와 노드 타입 ID는 코드 API보다 강한 계약이다.** 사용자의 태스크·정책 파일이 깨지는 것이 코드가 깨지는 것보다 훨씬 나쁘다.

---

## 26. 비기능 요구사항과 CI

### 26.1 비기능

- 플랫폼: Linux x86_64(1차), Windows, Linux aarch64(Jetson), macOS arm64(RS·CPU)
- 배포: 단일 정적 바이너리 + Python wheel. **`es-runtime-embedded`는 no-std 가능, 힙 할당 0**
- 결정성: §3.5 5계층. 결정적 모드 저하 ≤ 30%
- 메모리: §20 예산. OOM으로 종료하지 않는다
- **안전: Safety Plane 비활성 경로가 존재하지 않는다.** 컴파일 타임 보장
- 태스크·정책: **검증되지 않은 것은 실행되지 않는다**
- 관측 가능성: Tracy, GPU 타임스탬프, Prometheus

### 26.2 CI 계층

| 계층 | 시간 | 항목 |
|---|---|---|
| **PR** | < 10분 | 단위, 레이어링, 레이아웃 검증, clippy, fmt, IR 스키마·정규화 해시·순환·타입 검사, 결정성 린트, Cross-IR Check 픽스처, Safety Plane 위반 시나리오 |
| **머지** | < 1시간 | 물리 회귀, 골든 이미지, MJCF 적합성, Python/TOML/esgraph → IR 동등성, **Observation IR ↔ LeRobot 전처리 동등성**, CPU/GPU lowering 동등성(소규모) |
| **야간** | 수 시간 | 성능 회귀, 결정성 교차, 퍼저(IR·자산·가중치), `loom`, TSan, 대형 그래프 컴파일, **정책 동등성(계층 4)**, 리플레이 |
| **주간** | 수 시간 | 벤더별 결정성, 메모리 예산, 백엔드 비교(§17.2), **Evaluation IR 전 스위트** |
| **릴리스** | 수 일 | 표준 태스크 전체 학습 곡선, 실기 검증, IR 스키마 호환 매트릭스, 마이그레이션 |

**러너:** CPU×3, NVIDIA(RTX 4090급)×2, AMD×1, Intel×1, macOS×1, **실기 로봇 셀×1**(M3부터).

실기 로봇 셀 러너가 핵심이다. 실기 검증이 없으면 sim2real 주장을 할 수 없다.

---

## 27. 시장 전략

### 27.1 규제 증거물 (2027-01-20 고정 시점)

EU 기계규정 (EU) 2023/1230이 2027년 1월 20일부터 의무 적용된다. 2006/42/EC를 대체하며 **AI 기반 안전 기능을 처음으로 적합성 평가 대상에 포함**하고, "실질적 변경"에 안전 기능 소프트웨어 업데이트를 포함시켜 새 평가를 요구할 수 있게 했다. 디지털 문서화가 인정된다. AI 안전 기능의 정합 표준은 아직 개발 중이다.

대상: AMR/AGV, 적응형 제어·학습 기반 파지 로봇 팔·코봇, 자율 지상 로봇, 위험 구역 판정 비전 시스템, **토크 한계를 조정하는 학습 정책**.

**Provenance Bundle + Safety Case**

Provenance Bundle은 artifact lineage를 표현하지만, 그것만으로는 **"요구사항 → 검증 증거" 관계가 없다.** 그래서 Safety Case 그래프를 함께 둔다.

```
evidence.esb
├── manifest.json           execution_hash 와 구성 해시 전체 (§5.3)
├── safety_case/
│   ├── hazards.json        위험 분석 (식별된 위험 목록)
│   ├── requirements.json   각 위험에 대응하는 안전 요구사항
│   ├── traceability.json   요구사항 ↔ Deployment IR 제약 ↔ 검증 증거
│   ├── residual.json       잔여 위험과 근거
│   └── change_impact.json  변경 시 영향받는 요구사항·재검증 범위
├── task/ observation/ learning/ deployment/     5종 IR (canonical)
├── scene/                  평탄화 씬 + assets.lock (라이선스 포함)
├── policy/                 가중치 + base_model.lock (§19.3)
├── training/               §19.3 전체
├── validation/
│   ├── determinism.json    §3.5 계층별 결과
│   ├── evaluation.json     §10 전 스위트 결과
│   ├── safety.json         envelope 위반·폴백 발동 통계
│   └── domain_gap.json     §24.3
└── signature
```

**`traceability.json`이 핵심이다.**

```json
{
  "REQ-07": {
    "hazard": "HAZ-03 (인간 작업자와의 충돌)",
    "requirement": "EE 속도는 인간 감지 구역에서 0.25 m/s 를 넘지 않는다",
    "implemented_by": {
      "ir": "deployment.ir",
      "constraint": "envelope.velocity_limit.zone_human",
      "hash": "b3:4a71..."
    },
    "evidence": [
      { "kind": "static",  "check": "compile.cross_ir.DEP-031", "result": "pass" },
      { "kind": "runtime", "metric": "envelope_violation_rate.zone_human",
        "suites": ["nominal", "occlusion", "latency_injection"],
        "value": 0.0, "episodes": 800 },
      { "kind": "hil", "session": "hil_2026_11_03", "violations": 0 }
    ],
    "revalidation_trigger": ["deployment_hash", "policy_hash"]
  }
}
```

`revalidation_trigger`가 §5.3 해시 체인과 연결된다. **정책을 재학습하면 `policy_hash`가 바뀌고, 이 요구사항의 재검증이 필요하다는 것이 자동으로 판정된다.** "실질적 변경" 대응이 기계적으로 된다.

`es evidence verify bundle.esb`가 해시 체인 검증 + 리플레이 재실행 + 지표 대조 + traceability 완전성 검사를 수행한다.

**포지셔닝 주의:** 인증 기관이 아니고 번들이 적합성을 보증하지 않는다. **기술 문서 작성을 위한 증거 수집·추적 도구**다. 표준이 확정되면 매핑 문서를 추가한다. 과장하지 않는다.

### 27.2 채택 경로

**낮은 문턱부터 올라간다.**

| 단계 | 사용자가 얻는 것 | 사용자가 포기하는 것 | 시점 |
|---|---|---|---|
| 1. 텔레메트리 클라이언트 | 돌고 있는 LeRobot/Isaac 학습을 붙어서 본다 | 없음 | M1 |
| 2. Evaluation IR | 정책을 perturbation suite로 평가한다 | 없음 (평가만 교체) | M2 |
| 3. Observation IR | sim·데이터셋·실기 전처리가 하나가 된다 | 전처리 코드 | M2 |
| 4. Safety Plane + 배포 | 안전 런타임과 증거물을 얻는다 | 배포 스택 | M3 |
| 5. Learning IR | 정책 구성이 타입 검사되고 재현된다 | 정책 정의 방식 | M3 |
| 6. 전체 스택 | 해시 체인 완결, real-to-sim | 시뮬레이터 | M4 |

**1단계와 2단계에서 사용자는 아무것도 포기하지 않는다.** 이것이 채택 전략의 핵심이다. "우리 것으로 갈아타라"가 아니라 "지금 쓰는 것 위에 얹어라"다.

### 27.3 경쟁 포지션 요약

```
LeRobot          "무엇을 쓸 수 있는가"     — 정책·데이터·하드웨어의 카탈로그
Newton/Genesis   "얼마나 빠른가"           — 물리 처리량
Isaac Lab        "NVIDIA 스택 안에서"      — 수직 통합

Electric Sheep   "정확히 무엇이었고,
                  왜 그렇게 동작했고,
                  다시 그렇게 만들 수 있는가"
```

**"또 하나의 빠른 로봇 시뮬레이터"가 되면 경쟁력이 없다.** Robot Learning Compiler + Reproducibility/Safety Layer가 유일하게 방어 가능한 자리다.

---

## 28. 실행 계획

### 28.1 전체 일정

| 마일스톤 | 기간 | 누적 | 패킷 | 유형 (A/B/C/D) | 게이트 |
|---|---|---|---|---|---|
| **M0 계약** | 1.5개월 | 1.5 | ~40 | 60/25/12/3 | 5종 IR 스키마·정규화·해시, 검증 인프라 |
| **M1 비전 수직 슬라이스** | 3.5개월 | 5.0 | ~50 | 40/35/18/7 | **Franka + RGB + ACT + pick&place end-to-end** |
| **M2 평가·확장** | 3개월 | 8.0 | ~45 | 45/30/15/10 | Evaluation IR, 배치 도메인, 4k env |
| **M3 실기·안전·real2sim** | 3.5개월 | 11.5 | ~48 | 40/30/15/15 | **실기 로봇 배포 + Safety Plane 검증** |
| **M4 확장** | 3.5개월 | **15.0** | ~40 | 35/25/25/15 | PT, IR-C, 자체 물리(선택), 증거물 검증 |
| M5 지속 | | | | | NPU, 레지스트리, 자체 GPU 솔버 |

**총 15개월.** 자체 물리 솔버를 후순위로 둔 절감(-3개월)이 IR 계층 비용(+2개월)을 상쇄한다.

### 28.2 M0 — 계약 (6주, ~40 패킷)

**Wave 0 — 검증 인프라 (6, 1주)** — 기능 코드 이전에 완성한다
```
P01 xtask 골격                        A
P02 레이어링 검사 (§4.2 규칙 9종)      A
P03 골든 정책 + verify-goldens         A
P04 check-scope (diff ↔ 패킷 범위)     A
P05 API 다이제스트 (ash/Slang/torch/LeRobot 스키마)  A
P06 CI 계층 구성                       A
```

**Wave 1 — 수학·규약 (6, 1주)**
```
P07 Scalar 트레이트 + DoubleF32        B
P08 초월함수 계수 생성 + Rust          C   ← docs/design/transcendental.md 선행
P09 초월함수 Slang 미러 + 비트 일치     C
P10 규약 타입 + 왕복 하네스            A
P11 BinnedAcc RFA + property test      C   ← docs/design/deterministic-reduce.md 선행
P12 SIMD f64xN + 디스패치              B
```

**Wave 2 — 코어 (5, 1주)**
```
P13 ECS (아키타입, SoA)                B
P14 잡 시스템 + 결정적 파티셔닝         C
P15 시간 모델 타입                     A
P16 실패 시맨틱 타입                   A
P17 풀 + 범프 아레나 + 할당 0 어설션    B
```

**Wave 3 — IR 코어 (12, 2주)** ← **M0의 무게 중심**
```
P18 공통 타입 시스템 (Unit/Frame/TimeRef)    C  ← docs/design/ir-types.md 선행
P19 ImageSpec + 변환 규칙 (resize/crop)      C  ← docs/design/image-spec.md 선행
P20 Task IR 코어 + 노드 스키마               B
P21 Observation IR 코어 + 노드 스키마        B
P22 Learning IR 코어 + PolicyHandle 계약     B
P23 Deployment IR + SafetyEnvelope 스키마    B
P24 Evaluation IR 스키마                     B
P25 정규화 + *_hash + property test 5종      C  ← 게이트
P26 Cross-IR Check 규칙                      C
P27 Diagnostic 타입 + 진단 코드 사전         A
P28 TOML/esgraph 직렬화 + 왕복               A
P29 NodeFactory trait (Task/Learning)        B
```

**Wave 4 — 자산·백엔드 어댑터 (8, 1주)**
```
P30 SceneDesc + 안정 ID                B
P31 glTF 임포터                        A
P32 MJCF 파서 (default, compiler)      A
P33 MJCF 액추에이터·센서·텐던           A
P34 URDF + package://                  A
P35 임포터 퍼저                        A
P36 PhysicsBackend trait + MuJoCoCpu    B
P37 openusd 스파이크                    D
```

**Wave 5 — 골격 (3, 1주)**
```
P38 RS 뷰어 최소                       B
P39 에디터 셸 (탭 구조)                B
P40 텔레메트리 프로토콜 스켈레톤        B
```

### 28.3 M1 — 비전 수직 슬라이스 (14주, ~50 패킷)

**게이트: Franka + RGB 2뷰 + ACT + pick&place를 씬 조립 → 태스크·관측·학습·배포 저작 → MJWarp에서 데이터 수집 → PyTorch 학습 → 평가 → 실기 인터페이스 배포까지 end-to-end로 수행한다.**

pick&place를 M1 게이트로 삼은 이유: 사족보행 velocity 같은 순수 이동 벤치마크는 physics-first 플랫폼의 증명이지 비전 학습 플랫폼의 증명이 아니다. pick&place 하나로 카메라·Task IR·시간 관측·정책·데이터셋·액션 청킹·sim2real·텔레메트리·리플레이를 **한 번에** 검증한다.

| Wave | 내용 | 패킷 | 유형 |
|---|---|---|---|
| W1 백엔드 | MJWarp 어댑터, MJCF 씬 로드, 의미 매핑 리포트 | 6 | B/A |
| W2 렌더·비전 | 타일 아틀라스, 채널 계약, 센서 현실성 기본 | 7 | B/C |
| W3 Observation IR 실행 | CPU 참조 + GPU lowering, LeRobot 전처리 동등성 | 8 | C/A |
| W4 Learning IR 실행 | PolicyRuntime(Torch), ACT 구성, 청크·앙상블, **LeRobot 체크포인트 로드** | 9 | C/B |
| W5 Safety Plane | Envelope, Watchdog, Fallback, 위반 시나리오 스위트 | 6 | B |
| W6 Env 런타임 | 배치 도메인 기초, 리셋, 랜덤화, 에피소드 기록 | 6 | B |
| W7 데이터셋 | LeRobot 읽기·쓰기, split·identity 해시 | 4 | A |
| W8 에디터 | 텔레메트리, 3D 뷰, **읽기 전용 계층 그래프**, 전처리 전·후 이미지 | 8 | B |
| W9 배포 | policy.esb, es-runtime-embedded 골격, ONNX 내보내기 | 4 | A/B |

**선행 설계 문서:** `observation-lowering.md`, `learning-lowering.md`, `safety-plane.md`, `telemetry-protocol.md`

### 28.4 M2 — 평가·확장 (12주, ~45 패킷)

| Wave | 내용 | 패킷 |
|---|---|---|
| W1 Evaluation IR 실행 | perturbation 커널, 지표 계산, 리포트, 비교 도구 | 10 |
| W2 배치 도메인 | round_robin, 비동기 추론, 청크 버퍼, 결정성 | 8 |
| W3 정책 확장 | Diffusion Policy, SmolVLA, ONNX 런타임, 계층 4 검증 | 9 |
| W4 Newton 백엔드 | 어댑터 + 백엔드 비교 도구 | 5 |
| W5 성능 | 지표 9종 계측, 메모리 예산, GPU lowering 최적화 | 7 |
| W6 저작 | Python 빌더 완성, LeRobot config 변환 | 6 |

**게이트:** Evaluation IR 전 스위트 동작, 4,096 sim env × 512 obs env 안정, 정책 3종 계층 4 통과, 메모리 예산 정확도 ±10%

### 28.5 M3 — 실기·안전·real2sim (14주, ~48 패킷)

| Wave | 내용 | 패킷 |
|---|---|---|
| W1 실기 인터페이스 | rmw_zenoh 네이티브, HIL, 실기 카메라 드라이버 | 9 |
| W2 embedded 런타임 | no-std Observation IR 평가기, Safety Plane, NPU 백엔드 | 8 |
| W3 **3DGS real2sim** | es-splat, 임포터, 색·위치 정렬, LBS 바인딩 | 10 |
| W4 도메인 갭 | 진단 도구, 지표, 리포트 | 4 |
| W5 Safety Case | traceability, evidence 번들, verify CLI | 6 |
| W6 편집 그래프 | 편집 가능 에디터, Node SDK | 7 |
| W7 Learning Loop | collect/intervene/distill CLI, 개입 라벨 | 4 |

**게이트: 실기 로봇에서 정책 실행 + Safety Plane 동작 검증 + 도메인 갭 리포트 생성.** 실기 셀이 CI에 들어간다.

### 28.6 M4 — 확장 (14주, ~40 패킷)

경로추적기 + ReSTIR + SVGF / USD 네이티브 / IR-C(control) / 자체 물리 솔버 프로토타입(선택) / `es evidence verify` 완성 / MCP 인터페이스 / LLM 태스크 생성 / RoboVerse 변환.

### 28.7 크리티컬 패스 게이트

| # | 게이트 | 시점 | 유형 |
|---|---|---|---|
| 1 | 검증 인프라 6종 (Wave 0) | M0 | A |
| 2 | **IR 정규화 불변식 5종** | M0 | C |
| 3 | 초월함수 ≤2 ULP, CPU/Slang 비트 일치 | M0 | C |
| 4 | 5종 IR Cross-Check 픽스처 전수 | M0 | C |
| 5 | **LeRobot ACT 체크포인트 로드 → 동일 액션** | M1 | C |
| 6 | **Observation IR ↔ LeRobot 전처리 동등성** | M1 | C |
| 7 | **Franka pick&place end-to-end** | M1 | — |
| 8 | Safety Plane 위반 시나리오 전수 통과 | M1 | B |
| 9 | 텔레메트리 + 그래프 뷰 저하 < 1% | M1 | B |
| 10 | 내장 노드 타입 ID 동결 | M1 | — |
| 11 | Evaluation IR 전 스위트 결정적 재현 | M2 | C |
| 12 | 정책 3종 계층 4 통과 | M2 | D |
| 13 | 메모리 예산 정확도 ±10% | M2 | D |
| 14 | **실기 배포 + Safety Plane 검증** | M3 | D |
| 15 | 도메인 갭 지표 산출 | M3 | D |
| 16 | Safety Case traceability 완전성 | M3 | B |
| 17 | 증거물 번들 검증 왕복 | M4 | A |

### 28.8 운영 리듬

```
일간   의존 해소 패킷 3–6개 병렬 착수 (크레이트 비충돌)
       유형 A 자동 머지 / B diff 검토 / C 설계 대조 / D 실험 진행
       실패 시 스펙·오라클·구현 결함을 구분. 스펙 결함이면 패킷을 고친다
주간   코히런스 감사, 백로그 재정렬, 유형 D 리뷰
월간   게이트 점검, 사양 갱신
```

실효 병렬도: 유형 A 4–8 / B 2–3 / C 1 / D 1 per day.

### 28.9 M5 이후 — 현 상태 평가와 개선 계획

M5의 수직 슬라이스(plan V)가 처음으로 `collect → bake → train → eval → video` 전 구간을 한 번에
돌렸다. 이 절은 거기서 드러난 것을 사양으로 되먹인 것이다. 근거는 as-built 기록
`docs/design/visible-learning.md` 7.4–7.14와 그 열린 질문 11–15, `docs/packets/M5/V0…V7a`,
`docs/reviews/M0`–`M4`·`M3-W1`, 그리고 2026-09-15 오라클 서버(RTX 4090) 실측이다. 게이트는
§28.7의 번호로, 리스크는 §29의 행으로 참조하고 여기서 다시 쓰지 않는다. 실측이 아닌 수는 모두
`Target / Status: unverified`로 적고, 성능은 §12.4의 9지표로만 말한다(단일 `step/s` 금지).

한 문단 요약: **배관은 돌아간다. 정책은 아직 태스크를 못 한다. 그리고 그 사실은 하네스를 세 번
고친 뒤에야 믿을 수 있게 됐다.** IR 다섯 종, 해시 체인, Safety Plane, Evaluation IR은 데모 전
구간에서 실제로 작동했고, 수집·학습 쪽 산출물(데모 50편, 손실 곡선, `observation_hash`,
`lowering_hash`, `dataset_schema_hash`, 체크포인트)은 그대로 유효하다. 무효화된 것은 평가 수치뿐이고
(visible-learning.md 7.13), 그것이 아래 1·2·3이 다루는 대상이다.

**1. 부족한 점**

| # | 항목 | 스펙 | 현재 (as-built) |
|---|---|---|---|
| L1 | 실기 로봇 | §28.7 게이트 14·15·16, §24.2 | 로봇 셀이 없다. HIL은 호스트 쪽 절반만 증명하고(`tests/fixtures/hil/v1_small.eshil` 재생), 카메라는 ROS 2 메시지 경계의 픽스처 골든까지다. §29의 "Safety Plane 요구사항이 실기와 안 맞음" 행은 그대로 열려 있다 |
| L2 | 정책 성능 (비전) | 데모 `evaluation.toml`의 `success_rate >= 0.5` | 하네스를 모두 고친 뒤 재측정: nominal **0/16**, 6스위트 **0/96** (50 데모, 96×96, from scratch, 20,000 step; visible-learning.md 7.12 phase 2). **V8(2026-09-15)**: LeRobot 자신의 ACT(사전학습 ResNet18·CVAE·DETR 디코더)를 `lerobot-train`으로 100,000 step 학습시켜 비트 동일 임포트로 평가해도 20k/50k/100k에서 **0/16, 0/16, 1/16** — "모델이 문제"라는 가설은 기각됐다 (7.16, 열린 질문 17) |
| L3 | 정책 성능 (특권 상태) | 같음 | V7a phase 2 실측(2026-09-15): 20,000 step에서 nominal **0/16과 1/16**(독립 두 회), 5,000 step 1/16, 6스위트 **4/96**. 전 에피소드가 900 스텝 타임아웃이고 `violation.position`이 최다다. `envelope_violation_rate`는 1,000 step의 0.985에서 20,000 step의 0.15로 내려간다 — 정책은 학습할수록 매끄러워지지만 큐브를 담지 못한다 (visible-learning.md 7.15). **중단 규칙 발동**: 다음 용의자는 모델 크기·학습 길이가 아니라 물리·접촉 모델·전문가 궤적이다. 그 조사에 앞서 아래 사다리 3(V8)을 대조 실험으로 둔다 — 검증된 설계와 검증된 학습기로도 실패하면 데이터·물리·전문가가 문제이고, 성공하면 우리 변형이 문제였다 |
| L4 | 오라클 우선 | §1.4 | 하네스 결함 세 건이 **학습이 끝난 뒤에** 발견됐다: 수집과 평가의 envelope 기준 불일치(V6), 평가에 chunk 버퍼·temporal ensemble 부재와 시드 off-by-one(V6b). 그 전에는 스크립트 전문가 자신이 하네스를 통과하면 0/16이었다. 빠져 있던 오라클은 "하네스가 전문가를 통과시킨다" 하나였다 |
| L5 | 평가 게이트의 기준 | §9.4, §10.3 | 고친 하네스에서 전문가는 8/8이지만 `envelope_violation_rate`가 0.4815–0.5485다. 테스트 게이트는 `< 0.02`에서 Deployment IR의 watchdog `max_frac = 0.9`로 재고정됐다(visible-learning.md 7.13). §9.4가 실제로 작동시키는 수가 그것이라는 근거는 기록됐지만, **0.9는 정책의 퇴행을 잡지 못할 만큼 넓다.** 이것이 조용한 완화인지 아닌지는 사람의 판단으로 남는다 |
| L6 | 안전 한계 | §9.3 표 | `ee_velocity_max`, `contact_force_max`, `min_self_distance`, `min_env_distance`는 `tests/fixtures/visible-learning/deployment.toml`이 선언하지만 `es-safety`가 강제하지 않는다 — 플레인에 FK도 접촉 질의도 없다. 문서상의 한계이지 살아 있는 한계가 아니다 |
| L7 | 추론 지연 | §8.6, §9.2 | 평가는 지연을 모델링하지 않는다(수집은 1틱을 모델링한다). Deployment IR에 latency 필드가 없고 `Evaluation::run`은 Learning IR을 받지 않는다 |
| L8 | 물리·렌더 백엔드 | §4.3, §17.2, §15.3 | 백엔드는 Python 서브프로세스 line-JSON이고 env 하나를 담는다. `es eval run`은 `MuJoCoCpuBackend`만 쓴다. MJWarp·Newton 어댑터는 M4에서 검증됐지만 데모 경로에 연결돼 있지 않다. 렌더는 §15.3의 기본값인 래스터 경로이고, M4가 검증한 PT·ReSTIR·SVGF는 이 데모에 기여하지 않는다 |
| L9 | Python 없는 추론 | §2.4 | 학습도 평가도 추론을 torch 서브프로세스로 한다. `InferenceBackend`의 ONNX·Vulkan 자리는 이름만 예약돼 있다 |
| L10 | 결정성 | §3.5 | 실물 MuJoCo/torch CPU에서 `--jobs 1`과 `--jobs 6`이 비트 동일하지 않다(24셀 중 6셀에서 히스토그램·에피소드 길이가 갈리고, `success_rate`와 `envelope_violation_rate`는 전 셀 동일) — `MuJoCoCpuBackend`가 선언한 tier 3의 귀결이지 병합 로직의 결함이 아니다. CUDA 학습은 런 간 재현되지 않고, `--resident-gpu`는 CPU에서만 비트 동일하다 |
| L11 | 빌드 재현성 | §5.3의 `compiler_hash` | `Cargo.lock`이 `.gitignore`에 있고 추적되지 않는다. 해시 체인이 컴파일러를 주장하는데 의존성은 고정돼 있지 않다 |
| L12 | 특권 관측 | §5.1, §7.4 | `sim_` 접두사가 관례일 뿐 검증기가 없다. `Real` 실행 모드가 로봇이 공급할 수 없는 채널을 참조해도 막는 것은 사람의 눈뿐이다 (열린 질문 14) |
| L13 | 사전학습 백본 | §8.3 | `pretrained = true`는 lowering이 거부한다. 사전학습 가중치를 싣는 경로 자체가 없다. §29의 라이선스 행에 대한 소유자 결정(2026-09-15): torchvision의 ImageNet ResNet18 가중치(BSD-3)는 데모에 허용한다 — 사다리 3(V8)이 LeRobot ACT 기본값으로 처음 쓴다 |
| L14 | 컨텍스트 예산 | §1.5 | `es-ir` 5,947줄(목표 6,000). IR을 건드리는 개선(delta action space, provenance 검증기, `TemporalEncoder` 확장)은 분할 패킷이 선행돼야 한다 |
| L15 | CI 안정성 | §26.2 | `es-telemetry`의 `transport::a_connection_past_max_clients_is_refused`가 Windows 부하에서 간헐 실패한다. 거부를 ECONNRESET으로도 인정해야 한다 |
| L16 | 기록의 형태 | §1.2 | as-built가 1,300줄짜리 설계 노트 하나에만 있다. 이 절이 그 증류의 첫 단계다 |
| L18 | 제어 주기 | §9.2 `rate.control`, §12.1 배치 도메인 | **V10(2026-09-15)이 찾은 근본 원인**: 데모 전체가 선언된 50 Hz가 아니라 **200 Hz**로 돌았다. `Env::new`는 물리를 MJCF의 `timestep 0.005`로 두고 `BatchDomains::single_env()`의 추론 주기는 1틱이라, 기록된 액션 한 행이 물리 한 스텝이다(측정: mujoco 프로브가 substeps 1에서 0.0002 mm, 4에서 203 mm 벌어짐). 결과: 동적 엔벌로프가 스텝당 4배(가속도 16배) 헐겁고, 틱당 명령 증분 중앙값 0.0032 rad, 16행 청크가 320 ms가 아니라 80 ms, 데이터셋 `fps=50` 메타데이터가 틀림. 데이터(재생 50/50)·파지(50/50 들어 올림)·앙상블(범위 보존)은 원인이 아니다 (visible-learning.md 7.18, 열린 질문 18). 수정 = V11: 물리는 그대로, 추론 주기를 `rate.control`과 물리 속도에서 유도(4 substeps), 나누어떨어지지 않는 장면은 거부, 재수집·재학습 |
| L17 | 게이트 7의 판정 | §28.7 게이트 7 | 데모는 Franka + RGB 2뷰가 아니라 SO-101 큐브-담기다. 대체를 기록하고 게이트 7을 충족으로 볼지, 원문대로 열어 둘지가 미결이다 (visible-learning.md 열린 질문 9) |

**이 절이 확정하는 규칙 세 가지.** 나머지는 사람의 판단으로 남기되, 이 셋은 M5가 값을 치르고
배운 것이므로 사양이 된다.

1. **수직 슬라이스는 학습 전에 "하네스가 전문가를 통과시킨다" 오라클을 갖는다**(§1.4). 정책이 아닌
   스크립트 전문가를 같은 평가 경로·같은 플레인·같은 성공 술어로 통과시키지 못하면, 그 뒤의 어떤
   학습 수치도 정책이 아니라 하네스를 잰 것이다. M5는 이것을 세 번 뒤늦게 배웠다.
2. **무효화된 측정은 지우지 않고 무효라고 적는다.** 7.8–7.11의 표는 남아 있고 7.13이 그 위에
   무효 사유를 적었다. 사라진 수는 재발견되고, 무효 표시가 붙은 수는 재발견되지 않는다.
3. **재현되지 않는 지표로 성능을 주장하지 않는다**(§12.4). 지표가 워커별 시계에 의존하면, 그
   지표는 고쳐지기 전까지 보고서에서 선언조차 하지 않는다.

**2. 최적화 지점**

한 사이클의 벽시계 시간이 어디로 가는지는 측정돼 있다(visible-learning.md 7.11·7.14,
`docs/packets/M5/V5-fast-cycle.md`; 오라클 서버, 2026-09-15).

| 구간 | 실측 | 원인 |
|---|---|---|
| nominal 16 에피소드 | 4:16.49 / 4:16.68 (독립 두 회, `report.json`과 모든 프레임이 서로 바이트 동일) | env 하나짜리 물리 서브프로세스 |
| 6스위트 96 에피소드 | 25:56 (순차) → **5:49** (`--jobs 6`) | 셀 단위 샤딩. 샤드별 BLAS/torch 스레드 상한 이전에는 오히려 4시간+로 역전됐다(16코어에 ~90 스레드, load ~47) |
| 학습 20,000 step (batch 8) | 10:53 기본 / 10:57 `--resident-gpu` / 12:53 bf16 / **10:00** `--compile` | RTX 4090에서 어느 knob도 크게 움직이지 않는다. batch 8에서는 모듈이 샘플을 하나씩 forward 하므로 커널 런치 바운드다 |
| 학습 batch 64 + lr 선형 스케일링 | 1:24:31, `final_loss` **NaN** | 선형 스케일링 관례가 이 모델과 이 50 에피소드 데이터셋에서 성립하지 않는다 |

제거 방법과 그 값어치:

- **배치 lowering의 샘플 루프 제거.** 학습 시간의 근본 원인이고 `--resident-gpu`·bf16·`--compile` 중
  어느 것도 이것을 건드리지 못했다. `Target / Status: unverified`.
- **큰 배치를 warmup과 낮은 lr 스케줄로.** 선형 스케일링은 실측으로 발산했으므로 관례가 아니라
  스케줄이 필요하다. `Target / Status: unverified`.
- **에피소드 단위 샤딩.** 지금은 `es-env`의 에피소드 카운터가 막는다 — 에피소드 5의 초기 상태는
  0..4를 돌지 않고는 재현되지 않는다. seek이 생기면 셀 하나짜리 nominal 런도 병렬화된다.
- **물리 백엔드를 in-process·다중 env로.** 수집과 평가 양쪽에 동시에 듣는 유일한 수단이고,
  어댑터는 이미 있다(L8).
- **프레임 경로는 지금 손대지 않는다.** raw `.bin` 쓰기와 `cv2` 모자이크는 사이클 시간의 지배 항이
  아니다.
- **수집과 bake 구간은 아직 벽시계가 기록돼 있지 않다.** 위 표에 없는 것은 빠른 것이 아니라 재지
  않은 것이다. 다음 서버 런에서 같은 형식으로 함께 찍는다. `Target / Status: unverified`.

**측정의 한계도 최적화 대상이다.** §12.4의 9지표 중 `physics_steps_per_sec`와 `actions_per_sec`는
샤딩에서 워커별 시계를 나눈 값이라 재현되지 않는다. 데모의 `evaluation.toml`은 둘 다 선언하지
않지만, 처리량을 주장하려면 먼저 고쳐야 한다.

**중단 규칙.** 한 사이클(collect·bake·train·eval)이 30분 아래로 내려오면 속도 작업을 멈춘다.
실측상 남은 최대 항은 학습 약 11분이고, 그 아래에서의 병목은 기계가 아니라 판단이다.

**3. 정확도를 높이는 부분**

(a) 학습된 정책이 실제로 성공하게 만드는 것 — 시간당 정보량 순서:

| 순위 | 수단 | 왜 |
|---|---|---|
| 1 | V7a 특권 상태 정책의 결과를 먼저 본다 | 큐브의 정확한 자세를 받은 정책이 실패하면 남은 용의자는 그래프가 아니라 물리와 전문가다. 실험 하나가 두 질문을 분리한다 |
| 2 | 데모 200–500개, 224×224 또는 손목 카메라 | 96×96에서 25 mm 큐브를 from scratch로 찾는 것이 지금 비전 정책이 푸는 문제다 |
| 3 | 사전학습 시각 백본 | lowering의 `pretrained` 지원과 safetensors 가중치가 선행 조건이다(L13); 사다리 3(V8)은 LeRobot 체크포인트 로더로 이를 우회한다 |
| 4 | teacher(특권 상태) → student(비전) 증류 | 1이 통과하고 2가 실패하면 남는 경로다 |
| 5 | chunk 50 + temporal ensemble decay를 envelope에 맞춰 조정 | 전문가조차 blend 때문에 0.48–0.55의 위반율을 만든다. 정책이 clamp를 맞는 것은 정책만의 문제가 아니다 |
| 6 | 학습 시 증강에 기존 perturbation 커널 재사용 | `light_intensity`·`light_direction`은 이미 평가에서 돈다. 학습에 쓰면 평가 스위트가 곧 학습 분포가 된다 |
| 7 | 100,000 step + warmup 스케줄 | 2·3 다음에만 의미가 있다. 20,000 step에서 `final_loss`는 이미 0.0179다 |
| 8 | 전문가 페이싱을 blend가 살아남도록 더 부드럽게 | 지금 페이싱은 한계의 90%를 쓰고, 남은 10%를 blend 지터가 먹는다 |

(b) 측정을 신뢰할 수 있게 만드는 것:

- **평가에 추론 지연을 모델링한다**(L7). 지금 평가는 수집보다 한 틱 유리하다.
- **held-out 시드를 늘리고 지표별 신뢰구간을 낸다.** 16 에피소드에서 0/16과 1/16의 차이는 잡음이다
  — V3·V2b·V1c가 0.0625와 0.1250 사이를 오간 것이 그 예이고, 그 수들은 V6b가 모두 무효화했다.
- **모든 평가 리포트에 전문가를 대조군으로 함께 싣는다.** 같은 시드·같은 스위트에서 전문가의
  성공률과 위반율이 옆에 있으면, 정책의 수치가 하네스 탓인지 정책 탓인지가 표 안에서 판별된다.
- **clamp를 단계·관절별로 분해한다.** `events.json`에 이미 들어 있다. nominal의
  `violation.position` 2,031 대 `acceleration` 843 대 `velocity` 439이고, 그리퍼 관절이 유력한
  용의자라는 가설은 기록만 돼 있다.
- **"하네스가 전문가를 통과시킨다"를 모든 수직 슬라이스의 선행 오라클로 못 박는다**(L4). §1.4가
  이미 그것을 요구하고 있었다.

**중단 규칙(정확도).** 2와 3을 다 쓰고도 — 데모 500편, 224×224, 사전학습 백본 — nominal이 여전히
0/16인데 특권 상태 정책은 통과했다면, 더 넣을 데이터도 더 키울 모델도 아니다. 남은 것은 관측에서
액션으로 가는 사상 자체이고 다음 수는 4(증류)다. 반대로 특권 상태 정책까지 0/16이면 2·3·4를 모두
건너뛴다: L3의 중단 규칙이 이미 그 경우의 다음 용의자를 지정하고 있다.

**4. 향후 계획 (패킷 사다리)**

각 행은 §1.2의 패킷 하나이고 오라클은 실행 가능한 한 줄이다. 순서는 정보량과 차단 관계를 따른다.

| 순위 | 패킷 | 답하는 질문 | 오라클 (한 줄) | 유형 |
|---|---|---|---|---|
| 1 | **V7a phase 2** (완료) | 관측에 답이 들어 있으면 이 그래프는 태스크를 하는가 | `es eval run` nominal 16 시드 → `report.json`의 `success_rate` (중단 규칙 L3) | D |
| 2 | **M5-R1 하네스 선행 규칙** | 같은 결함이 다시 학습 뒤에 발견되지 않게 하려면 | 새 수직 슬라이스 패킷의 acceptance가 전문가-통과 테스트 이름을 담는다 | A |
| 3 | **V8 외부 ACT** (완료: 0/16·0/16·1/16, 등가성 비트 동일 — 모델 가설 기각) | 외부에서 설계·학습된 정책(LeRobot ACT, `lerobot-train`)이 같은 시연·같은 하네스에서 성공하는가, 그리고 우리 런타임이 그것을 동일하게 재현하는가 | v3.0 내보내기(+`meta/stats.json`) → `lerobot-train` → `lerobot.rs` 로더 → 게이트 5식 비트 동등 → `es eval run` nominal `success_rate`(데모 acceptance 0.5 그대로) | D |
| 4 | **V9 쇼케이스 렌더** (완료: 전문가 성공 4편 1280×720 mp4) | 사람이 볼 수 있는 영상인가 | 기록된 상태 궤적의 재생 렌더가 관측 해상도에서 기록 프레임과 비트 동일하고, 1280×720 H.264 mp4가 나온다(전문가 8/8부터) | B |
| 5 | **평가 지연 모델링** | 평가와 수집이 같은 시간 규약을 쓰는가 | `cargo test -p es-eval`: 지연이 선언된 배포에서 tick 0이 chunk underrun | B |
| 6 | **데모 acceptance 재고정** | 0.9가 아닌 무엇이 정책의 퇴행을 잡는가 | `es eval run`이 전문가는 통과시키고, 위반율이 전문가 최악값을 넘는 정책은 실패시킨다 | C |
| 7 | **`Cargo.lock` 고정** | `compiler_hash`가 빌드를 재현하는가 | `cargo xtask ci`가 lock 파일과 `--locked` 빌드를 검사 | A |
| 8 | **telemetry flake** | CI가 부하에서도 초록인가 | `cargo test -p es-telemetry transport` 100회 반복 통과 | A |
| 9 | **배치 lowering 벡터화** | 학습 시간을 무엇이 지배하는가 | 40 step 손실 곡선이 현 경로와 바이트 동일하면서 벽시계가 내려간다 | B |
| 10 | **`Env` 에피소드 seek** | 에피소드 단위 병렬이 가능한가 | seek 뒤의 상태가 0..n 재생과 비트 동일 | B |
| 11 | **MJWarp를 평가 경로에** | 백엔드가 정말 교체 가능한가 | `es backend compare --backends mjwarp,mujoco-cpu`가 max \|dqpos\|를 보고하고 두 리포트의 판정이 같다 | C |
| 12 | **미강제 안전 한계**(L6) | 선언된 한계가 살아 있는가 | `cargo test -p es-safety`: EE 속도 한계를 넘는 명령이 `Clamped`로 세어진다 (INV-12 — 넓히되 끄지 않는다) | B |
| 13 | **Python 없는 추론** | 배포 경로가 Python 없이 도는가 | 같은 체크포인트에서 torch 경로와 §8.9 계층 4 동등 | C |
| 14 | **`es-ir` 분할 → provenance 검증기** | 특권 채널이 실기로 새지 않는가 | `Real` 모드 Deployment IR + `sim_` 채널 → 검증 오류 | C |
| 15 | **M5 리뷰와 설계 노트 증류** | 기록이 사양으로 돌아왔는가 | `docs/reviews/M5.md`가 존재하고 `cargo xtask check-spec-refs`가 통과한다 | A |

**사다리에 없는 것, 그리고 그 이유.** 자체 물리 솔버는 §1.9의 축소 1번으로 그대로 잘려 있고, 위
12·14가 그 자리를 대신한다 — 지금 필요한 것은 새 솔버가 아니라 쓰고 있는 솔버의 접촉 모델을
판정하는 일이다. PT·ReSTIR 렌더 경로(§1.9 축소 2번)는 정책의 관측에는 들어오지 않는다: 96×96
래스터 이미지가 정책을 막고 있다는 증거가 없다. 다만 4(V9)의 쇼케이스 렌더는 기록된 궤적의
오프라인 재생이므로, 서버에서 헤드리스로 돌기만 하면 PT를 거기서 처음 쓸 수 있다. 에디터·저작 쪽은
이번 사다리에 없다 — M5가 드러낸 결함 중 UI가 원인인 것은 하나도 없었다. 1·2·4·6·7·8은
사람 한 명의 판단을 기다리지 않으므로 병렬로 착수할 수 있고(§28.8의 유형 A·B), 3·13·14는 각각
앞 행의 결과나 분할 패킷을 기다린다.

**M6은 두 번째 트랙, 4족 보행이다(소유자 결정, 2026-09-15).** 데모의 목적은 "사람 눈에 학습이
잘 됐다"를 보여주는 것이고, 프로젝트의 주장은 "외부에서 설계·학습된 정책을 우리 런타임이 같은
의미론으로 재현한다"이다(§8, §1.9). 3(V8)이 그것을 모방학습·팔에서 증명하면, M6은 같은 주장을
전혀 다른 정책 계열과 형태로 한 번 더 증명한다: MuJoCo Playground의 Go1/Go2 조이스틱 정책을
그쪽 PPO 학습기로 학습시켜(우리는 RL 학습기를 만들지 않는다) MLP 정책을 Learning IR로 가져오고,
우리 MuJoCo CPU 백엔드·Safety Plane·교란 스위트(밀치기·마찰·적재 하중)에서 돌려 4(V9)의 쇼케이스로
렌더한다. 선행 패킷: 기본 도형만 쓴 4족 모델 파생본(우리 로더는 mesh geom을 거부한다), Observation
IR의 히스토리 창으로 관측 스택 표현, 학습 환경과 우리 물리의 동일성 오라클(같은 XML로 두 백엔드
비교, §17.2), 관측 정규화 파라미터의 내보내기. 조사 문서: `docs/api-notes/mujoco-playground-quadruped.md`.
실기 게이트 14·15·16은 로봇 셀이 생길 때까지 열어 둔다. 3(V8)마저 실패하면 M6 앞에 **물리·전문가·액션
표현** 조사를 끼운다(**발동, 2026-09-15** — 패킷 V10 씬 진단: 전문가가 기록한 액션의 `es eval run` 재생,
그리퍼 접촉력으로 파지인지 밀기인지 판정, temporal ensemble이 파지 구간을 살리는지; 그다음 해상도와
시연 수를 하나씩). **V10의 판정(2026-09-15)**: 세 용의자는 모두 무죄이고 원인은 L18의 제어 주기다. 다음 패킷은 V11이며, 그 결과가 나올 때까지 플랜 V의 다른 어떤 것도 움직이지 않는다: 접촉 모델 검증(§17.2의 백엔드 비교로 판정), delta action space(§8.5, `es-ir`
분할 선행), 전문가 궤적의 재설계. 어느 쪽이든 사다리의 2·5·6이 선행 조건이다 — 하네스를 신뢰할 수
없으면 어느 쪽 결론도 측정이 아니다.

---

## 29. 리스크

| 리스크 | 심각도 | 대응 |
|---|---|---|
| **Learning IR이 실제 정책을 표현 못함** | **높음** | **M1 게이트 5·6이 이것을 조기 검증. LeRobot 체크포인트 로드 실패 시 `PolicyBundle` 통째 참조로 후퇴(§8.3)** |
| **Observation IR ↔ LeRobot 전처리 불일치** | **높음** | M1 게이트 6. CI 머지 게이트로 상시 검증 |
| **Safety Plane 요구사항이 실기와 안 맞음** | **높음** | M3에 실기 셀을 CI에 투입. 로봇 파트너 확보가 M2 중 필수 과업 |
| 오라클이 약해 잘못된 구현 승인 | 높음 | §1.4. PyTorch·MuJoCo·MPFR 참조 확보. 게이트 1로 인프라 선행 |
| **비전 대역폭·VRAM이 규모를 제한** | 중간 | §12.2 round_robin, §20 예산 모델, 사전 계산 후 거부 |
| **정책 추론 지연이 제어 주기를 지배** | 중간 | §8.4 LRN-052 컴파일 검사, §8.6 비동기 청크, ACT 급 우선 |
| 사전학습 백본 라이선스 | 중간 | §25.2. `base_model.lock` + 배포 시 검사. 2026-09-15 결정: torchvision ImageNet ResNet18(BSD-3)은 데모에 허용(§28.9 L13) |
| **3DGS 정렬 품질이 부족** | 중간 | §1.9 축소 3번. 외부 도구 + 수동 임포트로 대체 가능 |
| 백엔드 의미 불일치 | 중간 | §17.2 비교 도구, error 시 실행 차단 |
| 세션 간 통합 일관성 붕괴 | 중간 | `AGENTS.md` 계층, 주간 감사 |
| 유형 D 병목 (사람 2–3명) | 중간 | §28.1이 반영. 자체 물리를 미뤄 D 비중을 크게 낮춤 |
| 환각 API (ash/Slang/torch/LeRobot) | 중간 | `docs/api-notes/` 자동 생성, 버전 고정 |
| 규제 포지셔닝 과장 | 중간 | §27.1 명시: 증거 수집 도구. 표준 확정 전 매핑 주장 금지 |
| 범위 팽창 (IR 다섯 개) | 중간 | §5.1 경계 표를 RFC 기준으로. 노드 추가는 RFC |
| CI 고정비 (실기 셀 포함) | 중간 | §26.2 계층화, 실기는 주간·릴리스만 |
| Rust 인력 | 낮음 | C ABI/WASM + Node SDK |
| 벤더 드라이버 편차 | 낮음 | §3.2·§3.4, 벤더별 CI |

---

## 부록 A. 확정 및 보류 결정 사항

### A.1 확정

**아키텍처**
1. 제품은 **Robot Learning Compiler & Runtime**이다. 물리 엔진이 아니다
2. **IR 5종**: Task / Observation / Learning / Deployment / Evaluation. 경계는 §5.1
3. **Task IR은 `ObservationSpec`을 선언만 하고 구현하지 않는다**
4. **신경망을 Task IR에 넣지 않는다.** Learning IR이 별도다
5. **네트워크 내부는 opaque, 인터페이스 의미는 typed.** `PolicyHandle` 메타데이터 계약이 타입 검사 대상
6. **`ImageSpec`이 이미지 포트의 계약이다.** 해상도·색공간·카메라 모델·intrinsic·extrinsic·셔터·노출·왜곡 포함
7. **`Resize`·`Crop`이 intrinsic을 변환한다.** 타입 시스템에 포함
8. **시간은 3계층**: `History`(시스템) / `TemporalWindow`(학습 입력) / `TemporalEncoder`(신경망)
9. **액션은 chunk·horizon·execution mode로 모델링한다.** `policy(obs) → action` 폐기
10. **Safety Plane은 정책과 독립이다.** `es-safety`는 `es-policy`를 의존하지 않는다(§4.2 규칙 8)
11. **Safety Plane 비활성 경로가 존재하지 않는다**
12. **배치 도메인 4종을 분리한다**: simulation / observation / inference / training
13. **Learning IR에는 env 배치 의미론을 강제하지 않는다**
14. **물리는 백엔드다.** MJWarp가 기본, 자체 솔버는 M4+ 선택
15. `PhysicsBackend` / `PolicyRuntime` 트레이트 경계
16. **Learning Loop이 1급 워크플로다** (§13)
17. **3DGS real-to-sim이 렌더 경로의 하나다** (§16)

**재현성·검증**
18. `execution_hash = H(task, observation, learning, policy, dataset, deployment, compiler, runtime, hardware_capability)`
19. **`dataset_hash`는 content·schema·split 셋으로 구성된다**
20. `base_model.lock`이 provenance의 필수 항목이다
21. **결정성 계층 5종.** 계층 4는 정책 동등성이다
22. **결정적 실행 계약 = 8항목 전부.** float controls 단독으로 보장되지 않는다
23. **Vulkan float-controls는 capability 질의이지 강제 수단이 아니다.** SPIR-V execution mode로 적용한다
24. **GPU 물리 백엔드는 계층 1을 선언하지 않는다.** CPU 백엔드만 선언한다
25. RFA/binned summation을 결정적 리덕션으로 채택
26. **증강 노드는 평가에서 자동 비활성화된다**
27. **`evaluation_hash` 고정이 정책 비교의 전제다**
28. **Safety Case traceability가 증거물의 필수 요소다.** `revalidation_trigger`가 해시 체인과 연결된다

**엔지니어링**
29. **Zero-copy는 capability이지 제품 약속이 아니다.** `TensorTransport` 협상
30. **성능은 지표 9종으로 본다.** `step/s` 단일 지표 폐기
31. `graph_hash`(저작) / `*_hash`(의미) / `compile_hash`(실행) 분리
32. 편집기 레이아웃은 `.eslayout` 사이드카. IR 크레이트에 존재하지 않는다
33. CPU lowering은 성능 경로가 아니라 정답 기준
34. 릴리스 계획 / 디버그 계획 분리
35. 내장 노드 타입 ID는 M1부터 동결. 플러그인 ABI와 독립 버저닝
36. **검증되지 않은 태스크·정책은 실행되지 않는다**
37. 크레이트 소스 ≤ 10,000 lines (CI 강제)
38. **허용 확장점 7개만**: `PhysicsBackend`, `PolicyRuntime`, `TaskNodeFactory`, `LearningNodeFactory`, `InferenceBackend`, `Scalar`, `DeterministicAcc`

**개발 프로세스**
39. 구현 전량을 AI 에이전트가 수행한다. 사람은 사양·오라클·설계·판정
40. **오라클 우선 원칙.** 검증 하네스가 구현보다 먼저 온다
41. 작업 패킷이 계획의 최소 단위 (5요소)
42. 유형 A 자동 머지 / B diff ≤800줄 + 승인 / C 설계 문서 선행 / D 사람 주도
43. 골든 파일은 CI 읽기 전용. 범위 밖 수정 CI 차단
44. 검증 인프라 완성 전 기능 패킷 착수 금지

**채택·시장**
45. **채택은 "얹기"부터 시작한다.** 텔레메트리 → 평가 → 관측 → 안전 → 학습 → 전체(§27.2)
46. LeRobot 호환이 최우선 상호운용 타깃
47. 인증 기관이 아니라 증거 수집·추적 도구다

### A.2 보류·검증 필요

| 항목 | 해소 | 판단 기준 |
|---|---|---|
| **Learning IR이 ACT·DP·SmolVLA를 실제로 표현하는가** | **M1** | **LeRobot 체크포인트 로드 후 액션 일치** |
| **Observation IR이 LeRobot 전처리와 수치 일치하는가** | **M1** | 허용오차 내 일치율 |
| **Safety Envelope 항목이 실기에 충분한가** | **M3** | 로봇 파트너 피드백, HIL 위반 시나리오 |
| π₀ 급 대형 VLA를 `PolicyBundle`로 다루는 것이 충분한가 | M2 | 실사용 피드백 |
| 비전 대역폭 실측 vs §12.4 산정 | M2 | 측정 오차 ±20% 이내 |
| 메모리 예산 모델 정확도 | M2 | 실측 대비 ±10% |
| GPU 스칼라 표현 기본값 | M2 | 커널 시간 + 궤적 오차 |
| **3DGS 색·위치 정렬 후 도메인 갭 감소량** | **M3** | 정렬 전후 정책 응답 차이 |
| **실기 배포에서 `execution_hash` 재현이 성립하는가** | **M3** | 동일 번들 2회 배포 비교 |
| 백엔드 간 의미 매핑의 미매핑 항목 분포 | M2 | §17.2 리포트 |
| 계층 1을 "동일 벤더 계열"로 확장 가능한가 | M4 | 벤더 3사 교차 replay |
| `es generate`의 실제 성공률 | M4 | 검증 통과율·중복 제거율 |
| Safety Case 스키마와 정합 표준의 정합 | M4 | AI 안전 기능 표준 초안 공개 후 |
| 유형 A 패킷이 절반을 넘는가 | M0 종료 | 실측. 미달 시 §1.8 예산 재조정 |
| 자체 물리 솔버를 만들 가치가 있는가 | M4 | MJWarp/Newton 대비 차별 근거 |

---

## 부록 B. 핵심 코드 설계

### B.1 IR 공통 타입

```rust
// es-ir/src/types.rs
pub struct PortType {
    pub elem: ElemType,          // F32 | F16 | Bf16 | F64 | I32 | U8 | Bool
    pub shape: Shape,
    pub unit: Unit,
    pub frame: Frame,
    pub time: TimeRef,
    pub image: Option<ImageSpec>,   // 이미지 포트만
}

pub enum Unit {
    Dimensionless, Length, Angle, Mass, Time,
    Velocity, AngularVelocity, Acceleration, Force, Torque,
    Pressure, Current, Voltage, Quaternion, RotationMatrix,
    Normalized { lo: f64, hi: f64 },
    Pixel, Luminance, Depth, Token,
    Composite(UnitPowers),       // m^a · kg^b · s^c · rad^d
}

pub enum Frame {
    World, LocalOrigin, Body(StableId), Sensor(StableId),
    Joint(StableId), Camera(StableId), Image(StableId), Policy,
}

pub enum TimeRef {
    Tick,
    Sensor { id: StableId, align: Align },          // Hold|Interpolate|Reject
    Window { base: Box<TimeRef>, n: u32, stride: u32 },
}
```

### B.2 `ImageSpec` 변환 규칙

```rust
impl ImageSpec {
    /// Resize 는 intrinsic 을 스케일한다. rescale=false 면 경고 OBS-034.
    pub fn resized(&self, w: u32, h: u32, rescale: bool) -> Self {
        let (sx, sy) = (w as f64 / self.width as f64, h as f64 / self.height as f64);
        let intr = if rescale { self.intrinsics.scaled(sx, sy) } else { self.intrinsics };
        Self { width: w, height: h, intrinsics: intr, ..*self }
    }

    /// Crop 은 principal point 를 이동시킨다.
    pub fn cropped(&self, rect: Rect, rescale: bool) -> Self { /* cx -= x0, cy -= y0 */ }

    /// Undistort 는 distortion 을 None 으로 만들고 intrinsic 을 새로 설정한다.
    pub fn undistorted(&self, new_intr: Intrinsics) -> Self { /* ... */ }
}
```

**이 세 함수가 §7.2 OBS-034 검사의 근거다.** 컴파일러가 `ImageSpec`을 그래프 따라 전파하면서 `CameraProjection` 노드에 도달했을 때 intrinsic이 일관되는지 확인한다.

### B.3 `LearningGraph`와 `PolicyHandle`

```rust
// es-ir/src/learning.rs
pub struct LearningGraph {
    pub schema_version: u32,
    pub inputs: Vec<TensorPort>,
    pub nodes: Vec<LearningNode>,
    pub outputs: Vec<TensorPort>,
    pub policy: PolicyHandle,
}

pub struct PolicyHandle {
    pub architecture: ArchKind,          // Act|Diffusion|FlowMatching|Discrete|Bundle
    pub base_model: Option<BaseModelRef>,// uri + hash + license
    pub weights: WeightsRef,             // safetensors|onnx|spirv + hash
    pub contract: PolicyContract,        // §8.4
}

pub struct PolicyContract {
    pub inputs: BTreeMap<SmolStr, TensorPort>,
    pub observation_window: u32,
    pub action_dim: u32,
    pub horizon: u32,
    pub execute_chunk: u32,
    pub replanning_hz: f32,
    pub execution_mode: ActionExecutionMode,
    pub runtime: RuntimeHints,           // dtype, expected_latency_ms, deadline_ms
}
```

### B.4 Safety Plane

```rust
// es-safety/src/lib.rs   — no_std 가능, 힙 할당 0
pub struct SafetyPlane<const NJ: usize, const H: usize> {
    envelope: SafetyEnvelope<NJ>,
    watchdogs: WatchdogSet,
    fallback: FallbackPolicy,
    state: SafetyState<NJ, H>,          // 사전 할당
    counters: SafetyCounters,           // 위반·폴백 통계 (§10.3)
}

impl<const NJ: usize, const H: usize> SafetyPlane<NJ, H> {
    /// 청크를 검증·보정한다. 실패할 수 없다. 항상 안전한 액션을 반환한다.
    pub fn validate(&mut self, chunk: &ActionChunk<NJ, H>, obs_age: Duration,
                    now: PhysTick) -> SafeAction<NJ> { /* ... */ }
}

// es-policy 를 의존하지 않는다 (§4.2 규칙 8).
// validate 는 Result 를 반환하지 않는다. 안전측 액션은 항상 존재한다.
```

**`validate`가 `Result`를 반환하지 않는 것이 설계 의도다.** 안전 계층이 실패를 위로 전파하면 호출자가 처리를 잊을 수 있다. 항상 실행 가능한 액션을 반환하고 사건은 카운터에 기록한다.

### B.5 배치 도메인 스케줄러

```rust
// es-env/src/batch.rs
pub struct BatchPlan {
    pub sim: SimDomain,            // N_sim, backend, dt_phys
    pub obs: ObsDomain,            // N_obs, selection, views, sensor_dt
    pub inf: InfDomain,            // N_inf, runtime, async, deadline
    pub train: Option<TrainDomain>,
}

pub enum EnvSelection {
    All,
    Subset(Vec<EnvId>),
    RoundRobin { k: u32, period: u32 },   // 결정적: (tick / sensor_period) % ceil(N/k)
}

/// 비동기 추론 결과의 적용 시점은 "도착 시각"이 아니라
/// "어느 tick 의 관측으로 계산되었는가" 로 결정된다 (§12.3).
pub struct ChunkArrival {
    pub computed_from: PhysTick,
    pub apply_at: PhysTick,        // computed_from + 결정적 지연
    pub chunk: ActionChunk,
}
```

### B.6 해시 체인

```rust
// es-ir/src/hash.rs
pub struct HashChain {
    pub asset: Vec<[u8; 32]>,
    pub scene: [u8; 32],
    pub task_graph: [u8; 32],      // 저작 정체성
    pub task: [u8; 32],            // 의미 정체성
    pub observation: [u8; 32],
    pub learning: [u8; 32],
    pub policy: [u8; 32],          // weights + arch + base_model
    pub dataset: DatasetHash,      // content + schema + split
    pub deployment: [u8; 32],
    pub evaluation: Option<[u8; 32]>,
    pub compiler: [u8; 32],
    pub runtime: [u8; 32],         // PolicyRuntime + TensorTransport 선택 포함
    pub hardware: HardwareCapability,
}

impl HashChain {
    pub fn execution_hash(&self) -> [u8; 32] { /* BLAKE3 over canonical encoding */ }
    /// 무엇이 바뀌면 무엇을 재검증해야 하는가 (§27.1 revalidation_trigger)
    pub fn diff(&self, other: &Self) -> Vec<ChangedComponent> { /* ... */ }
}
```

### B.7 정규화 불변식 (property test)

```rust
proptest! {
    #[test] fn hash_independent_of_node_ids(g in arbitrary_ir()) {
        prop_assert_eq!(ir_hash(&canon(g.clone())), ir_hash(&canon(shuffle_ids(g))));
    }
    #[test] fn hash_independent_of_ui(g in arbitrary_ir(), l in arbitrary_layout()) {
        prop_assert_eq!(ir_hash(&parse_esgraph(&write_esgraph(&g, &l))?), ir_hash(&g));
    }
    #[test] fn roundtrip_preserves_hash(g in arbitrary_ir()) {
        prop_assert_eq!(ir_hash(&parse_toml(&write_toml(&g))?), ir_hash(&g));
    }
    #[test] fn param_change_changes_hash(g in arbitrary_ir(), p in arbitrary_edit()) {
        let g2 = apply(g.clone(), p); prop_assume!(g2 != g);
        prop_assert_ne!(ir_hash(&canon(g)), ir_hash(&canon(g2)));
    }
    #[test] fn unit_algebra_laws(a in arbitrary_unit(), b in arbitrary_unit()) {
        prop_assert_eq!(mul(a, b), mul(b, a));
        prop_assert_eq!(div(mul(a, b), b), a);
    }
}
```

**§28.7 게이트 2가 이 다섯 개다.** 여기가 흔들리면 §5.3 해시 체인 전체가 무의미하다.

### B.8 결정적 누산기와 레이어링 검사

```rust
pub trait DeterministicAcc<T>: Default + Clone {
    fn add(&mut self, v: T);
    fn merge(&mut self, other: &Self);   // 결합적·교환적
    fn finish(&self) -> T;
}
pub struct BinnedAcc<const K: usize = 3> { bins: [f64; K], index: Option<i32> }
```

```rust
// xtask: §4.2 규칙 9종을 cargo metadata 로 검사
const LAYERS: &[(&str, u8)] = &[
    ("es-math",0), ("es-core",1),
    ("es-gpu",2), ("es-assets",2), ("es-usd",2),
    ("es-actuator",3), ("es-sensor",3), ("es-physics-core",3),
    ("es-physics-backend",4), ("es-physics-cpu",4), ("es-physics-gpu",4),
    ("es-render",5), ("es-splat",5),
    ("es-ir",6), ("es-compile",7),
    ("es-policy",8), ("es-safety",8),
    ("es-env",9),
    ("es-data",10), ("es-telemetry",10), ("es-eval",10),
    ("es-ros2",11), ("es-py",11), ("es-script",11), ("es-transport",11),
    ("es-editor",12),
];
// 추가 검사:
//   es-safety 가 es-policy 를 의존하지 않는가        (규칙 8)
//   es-ir 이 es-compile / 백엔드 / torch 를 모르는가  (규칙 6)
//   es-ir 이 egui 를 모르는가                        (규칙 7)
//   es-transport 외 CUDA/HIP 심볼 부재               (규칙 5)
```

---

## 부록 C. 에이전트 지침 (요약)

전문은 저장소 `AGENTS.md`. 여기서는 핵심 항목만 기록한다.

### C.1 루트 `AGENTS.md` 추가 규칙

```markdown
## Electric Sheep 고유 규칙

### IR 경계 (사양 §5.1)
Task IR 에 신경망을 넣지 마라. Learning IR 이 별도다.
Task IR 은 ObservationSpec 을 선언만 한다. 전처리는 Observation IR 소관.
IR 에 UI·레이아웃 타입을 넣지 마라. .eslayout 사이드카에 있다.

### 안전 (사양 §9, §4.2 규칙 8)
es-safety 는 es-policy 를 의존하지 않는다. 역방향 의존을 만들지 마라.
Safety Plane 을 비활성화하는 코드 경로를 만들지 마라.
  테스트에서도 만들지 마라. 테스트는 envelope 를 넓히는 방식으로 한다.
SafetyPlane::validate 는 Result 를 반환하지 않는다. 시그니처를 바꾸지 마라.

### 이미지 (사양 §7.2)
Resize / Crop 을 구현할 때 ImageSpec::resized / cropped 를 쓰고
intrinsic 변환을 건너뛰지 마라. rescale_intrinsics=false 는 사용자가
명시적으로 선택했을 때만이다.

### 정책 런타임 (사양 §2.4, §8.7)
전처리·후처리는 IR 이 소유한다. PolicyRuntime 에 넘기지 마라.
정책 가중치는 safetensors 우선. pickle 기반 로딩을 추가하지 마라.

### 결정성 (사양 §3.4)
Vulkan float-controls 는 capability 질의다. 이것을 "설정"으로 다루는
코드나 주석을 쓰지 마라. 적용은 SPIR-V execution mode 로 한다.

### 성능 (사양 §12.4)
step/s 단일 지표로 성능을 보고하지 마라. 지표 9종을 쓴다.
검증되지 않은 성능 수치를 사실로 쓰지 마라. "Target / Status: 미검증" 형식.
```

### C.2 `docs/invariants.md` 추가 항목

| ID | 내용 | 검사 |
|---|---|---|
| INV-11 | `es-safety` ⇏ `es-policy` | xtask layering |
| INV-12 | Safety Plane 비활성 경로 부재 | clippy 커스텀 + 코드 리뷰 |
| INV-13 | `SafetyPlane::validate` 시그니처 고정 | 컴파일 타임 |
| INV-14 | `ImageSpec` 변환 시 intrinsic 갱신 | 단위 테스트 + OBS-034 |
| INV-15 | 평가 실행 시 증강 노드 비활성 | Evaluation IR 실행 테스트 |
| INV-16 | pickle 기반 가중치 로딩 부재 | 의존성 검사 |
| INV-17 | 허용 확장점 7개 외 단일 구현 트레이트 부재 | 주간 감사 |

---

## 부록 D. 참고 자료

**정책·학습 스택**
- LeRobot — https://github.com/huggingface/lerobot
- LeRobot: An Open-Source Library for End-to-End Robot Learning — arXiv:2602.22818 (지원 정책 목록, 추론 지연 실측)
- SmolVLA — arXiv:2506.01844, https://huggingface.co/blog/smolvla
- ACT (Action Chunking with Transformers) — Zhao et al., 2023
- Diffusion Policy — Chi et al., arXiv:2303.04137
- π₀ — Black et al., arXiv:2410.24164
- π₀.₅ — Physical Intelligence, arXiv:2504.16054
- OpenVLA — Kim et al., arXiv:2406.09246
- OpenVLA-OFT (action chunking, parallel decoding) — arXiv:2502.19645
- Real-Time Execution of Action Chunking Flow Policies — arXiv:2506.07339
- GR00T N1 / N1.5 / N1.7 — NVIDIA

**Real-to-Sim / 3DGS**
- Real-to-Sim Robot Policy Evaluation with Gaussian Splatting — arXiv:2511.04665, https://real2sim-eval.github.io/
- RL-GSBridge — Wu et al., 2025
- Real-is-Sim (Embodied Gaussians) — Abou-Chakra et al., 2025
- RoboGSim — Li et al., 2025
- RialTo — Torne et al., 2024
- ReaDy-Go — arXiv:2602.11575
- PhysGaussian — CVPR 2024

**시뮬레이션 백엔드**
- Newton — https://github.com/newton-physics/newton
- Newton 1.0 GA — https://github.com/newton-physics/newton/discussions/2176
- MuJoCo Warp — https://github.com/google-deepmind/mujoco_warp
- Genesis World / Quadrants — https://github.com/Genesis-Embodied-AI/genesis-world
- Isaac Lab 성능 벤치마크 — https://isaac-sim.github.io/IsaacLab/
- Isaac Lab (타일 렌더링, 센서 처리량) — arXiv:2511.04831
- ManiSkill3 — arXiv:2410.00425
- RoboVerse / MetaSim — arXiv:2504.18904

**결정성·수치**
- NVIDIA CCCL 부동소수 결정성 — https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/
- Demmel & Nguyen, Reproducible Floating Point Summation — ACM TOMS 2020, https://dl.acm.org/doi/10.1145/3389360
- ExBLAS / Kulisch accumulator
- Vulkan Float Controls — https://docs.vulkan.org/spec/latest/chapters/shaders.html
- Vulkan Limits — https://docs.vulkan.org/spec/latest/chapters/limits.html
- CUDA–Vulkan external memory/semaphore interop — NVIDIA CUDA Programming Guide

**플랫폼**
- rmw_zenoh — https://github.com/ros2/rmw_zenoh
- ROS 2 Lyrical Luth RMW 등급 — https://docs.ros.org/en/lyrical/Releases/Release-Lyrical-Luth.html
- openusd (Rust) — https://github.com/mxpv/openusd
- Slang — https://shader-slang.org/
- egui-snarl — https://github.com/zakarumych/egui-snarl

**규제**
- Regulation (EU) 2023/1230 (기계규정), 2027-01-20 적용 — EUR-Lex
- Regulation (EU) 2024/1689 (AI Act)
- Machinery Regulation ↔ AI Act 관계 (2026 개정) — https://ai-resources.eu/en/atti-normativi/ue/macchine/
