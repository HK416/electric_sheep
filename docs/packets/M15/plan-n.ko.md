<!-- Korean translation of docs/packets/M15/plan-n.md. The English file is the working copy; regenerate this when it changes. -->
# M15 플랜 N — 다중 카메라 정책 (여러 시점으로 학습하고, 더 적은 시점으로 배치한다)

> `docs/design/multi-camera.md`의 패킷(오너의 결정은 그 문서의 6절). 오너는 2026-09-29에 이 플랜을
> 시작했다("시작"). 각 작업은 §1.2의 패킷이며, **Files** 블록이 `cargo xtask check-scope`용
> `context`이다.

**목표:** 세 카메라 — 손목, 머리 위, 옆면 — 로 보는 SO-101 큐브 작업을 모든 경로(렌더, 수집, 베이크,
학습, 평가)에 통과시킨다. 이어서 `Sum` 융합과 단일 시점 학습을 갖춘 MAD의 공유 인코더를 도입하고,
세 시점으로 학습한 정책을 한 시점에 배치한다. 각 단계는 측정한다.

## 전역 제약

- `docs/packets/M14/plan-q.md`의 전역 제약이 모두 그대로 유효하다(MSRV 1.85, egui 0.32.3, 골든은
  생성기로만, 트레일러 없음, 워크트리 + `--ff-only`, 푸시 금지, 원격 서버 없음, 에이전트 패킷
  안에서 학습 실행 금지, 결정은 `es-editor-model`에).
- **동시에 도는 에이전트마다 `CARGO_TARGET_DIR` 하나**(`target/wt-<packet>`)를 두고 병합 뒤에 지운다.
  오케스트레이터는 어떤 에이전트도 빌드하지 않을 때만 웨이브의 `cargo xtask ci`를 돌린다.
- **커밋된 모든 문서와 숫자는 그대로다.** 여러 카메라는 새 해시를 가진 새 문서에 산다. 카메라가 하나인
  Task IR은 오늘과 바이트 단위로 같게 렌더·수집·베이크된다(평평한 프레임 레이아웃 포함). 기존
  골든과 데이터셋 내용 해시는 움직이면 안 된다.
- **IR 변경은 스펙 먼저**(N5): `docs/ARCHITECTURE.ko.md`(정본)와 `ARCHITECTURE.md`를 같은 커밋에
  담는다. 새 선택적 IR 필드는 설정하지 않으면 정규 인코딩에 나타나지 않으므로, 커밋된 어떤 문서의
  해시도 움직이지 않는다(커밋된 문서들의 해시에 대해 테스트한다).
- **안전 평면은 손대지 않는다.** 런타임에 카메라가 없으면 여전히 `SensorDropout` 워치독이다.
- **오라클(§1.4):** 렌더는 단일 카메라 렌더와 비트 단위로 비교한다. Learning IR 로워링은 손으로 쓴
  PyTorch 모듈과 비교한다. 부분집합 번들은 버려진 항을 제거한 전체 그래프와 비교한다.

## 웨이브

| 웨이브 | 작업 | 선행 |
|---|---|---|
| 1 | N1 세 시점 장면과 문서 · N2 env당 여러 카메라(렌더, 프레임, 평가) | — |
| 2 | N3 카메라별 베이크와 내보내기 | N2 |
| 3 | E1 실험 1: 한 시점 대 세 시점(오케스트레이터) | N1–N3 |
| 4 | N5 IR과 스펙의 `Sum`, `share` · N6 그 로워링 · N7 단일 시점 학습 | — (N6은 N5 뒤) |
| 5 | N8 `es policy subset` | N5, N6 |
| 6 | E2 실험 2: MAD, 더 적은 시점으로 배치(오케스트레이터) | N5–N8 |
| 7 | N9 에디터의 카메라 옵션 | E1 |

---

### 작업 N1: 세 시점 장면과 그 문서

**Files:** `tests/fixtures/mjcf/so101_pick_place_views.xml`(기존 장면에 있다면 그
`.PROVENANCE.json`도), 새 `tests/fixtures/visible-learning/*-views*.toml`과 `*-cam*.toml` 문서,
그것을 쓰는 생성기(`task-pt.toml` 등이 어떻게 만들어졌는지 찾아 그대로 따른다), 그 테스트.

- 장면: 오늘의 `so101_pick_place.xml`에 `camera_mount` 안의 **손목 카메라**(그리퍼를 따라 집게를
  내려다본다)와 **옆면 카메라**(월드에 고정, 작업 공간과 통이 보인다)를 더한다. 머리 위 카메라와 모든
  바디, 지오메트리, 조인트는 그대로다.
- 문서는 각각 새 파일이다. 세 이미지 채널(`rgb_overhead`, `rgb_wrist`, `rgb_side`, 오늘처럼 96×96
  RGB)을 갖고 새 장면을 고정(pin)하는 Task IR. 실험 1을 위해서는 IR 경로 위의 카메라 전용 쌍 —
  한 시점(머리 위)과 세 시점 — 을 두며, 각각 Observation IR(카메라마다 `observation-augmented.toml`
  체인, `joint_state`, `sim_cube_pose`는 **없음**), Learning IR(U3의 `learning-pretrained.toml`
  그래프: 시점마다 ResNet18 하나, `Concat` 융합), Evaluation IR(`evaluation-augmented.toml`의 스위트,
  시드, 메트릭, 합격 기준 그대로이고 참조만 다르다), 배치(오늘의 것), 학습 레시피(U3의 설정), 사이클을
  가진다.
- **오라클:** 문서들이 검증되고 교차 검증된다(XIR). 리셋 자세에서 각 카메라를 렌더하면 큐브가 보이고
  (고정된 몇 개의 큐브 위치에 대해, 시점별로 큐브 색 픽셀 수가 하한을 넘는다), 손목 시점은 그리퍼와
  함께 움직인다(두 팔 자세, 두 개의 서로 다른 이미지). 커밋된 문서의 해시는 그대로다.

### 작업 N2: env당 여러 카메라

**Files:** `crates/es-env/src/render.rs`, `crates/es-tools/src/lib.rs`,
`crates/es/src/cmd/{loop,eval}.rs`, `crates/es-eval/src/runner.rs`,
`crates/es-data/src/collect.rs`(프레임 싱크만), 그 테스트, `tests/golden/**`(생성기로만).

- `EnvRenderer`는 스텝마다 Task IR의 모든 이미지 채널을 렌더한다(카메라마다 아틀라스 타일 하나,
  §15.2). 채널이 하나인 Task IR은 오늘과 똑같이 렌더된다.
- 채널이 여럿일 때 `--frames`는 `<frames>/<channel>/<NNNNNN>.bin`(+ `.json`)을 쓴다. LeRobot v3
  내보내기가 이미 기대하는 레이아웃이다. 채널이 하나면 오늘의 평평한 레이아웃을 바이트 단위로 유지한다.
- 평가 러너와 `es loop collect`는 각 이미지 포트에 자기 카메라의 프레임을 먹인다.
- **오라클:** 각 카메라의 타일은 그 카메라의 단일 렌더와 비트 단위로 같다(M11/X3b가 env에 대해 고정한
  것처럼). 두 카메라 테스트 장면은 두 포트 모두 먹인 채로 수집·평가되며, 프레임은 재사용된 하나가
  아니라 서로 다른 둘이다. 기존의 모든 프레임 골든과 데이터셋 내용은 그대로다.

### 작업 N3: 카메라별 베이크와 내보내기

**Files:** `crates/es/src/cmd/{dataset,train}.rs`, `crates/es-data/src/{training,lerobot/**}.rs`,
그 테스트.

`es dataset bake`는 각 이미지 포트를 자기 카메라 디렉터리에서 읽는다. 하나의 디렉터리를 모든
카메라의 것으로 하드 링크하던 `mirror_frames`는 실제 카메라별 레이아웃으로 대체된다.
**오라클:** 세 카메라 픽스처가 서로 다른 세 이미지 텐서를 베이크한다. 단일 카메라 베이크는
바뀌지 않는다.

### 작업 E1: 실험 1(오케스트레이터, 조용한 PC, 에디터를 열어 둔 채)

세 카메라로 렌더한 전문가 시범 200개(한 번의 수집이 두 갈래를 모두 먹인다). 카메라 전용 IR 경로를
머리 위 시점만으로, 그리고 세 시점 모두로 학습하며 각각 학습 시드 둘, 같은 스위트로 평가한다.
`docs/packets/M15/NV-verification.md`에 기록한다.

### 작업 N5: IR과 스펙의 `Sum`과 `share`

**Files:** `crates/es-ir/src/learning.rs`, `docs/ARCHITECTURE.ko.md`, `docs/ARCHITECTURE.md`(§8.3),
그 테스트.

`FusionKind::Sum`(폭이 같은 입력들, 출력은 그 합). `VisionEncoder.share: Option<NodeId>` — 이
인코더가 가중치를 빌려 쓰는 인코더. 이름을 대어 거부하는 경우: `VisionEncoder`가 아닌 노드를 가리킬
때, 다른 백본·폭·토큰 수·사전학습 출처를 가진 노드를 가리킬 때, 자기 자신일 때, 스스로 공유하는
노드일 때(체인 금지). 없는 필드는 정규 인코딩에 나타나지 않는다: 커밋된 모든 Learning IR 해시는
그대로다.

### 작업 N6: 그 로워링

**Files:** `crates/es-policy/src/lower/torch.rs`, `python/es/builder.py`(`shared` 인자가 이제
`share`를 쓴다), 테스트.

`Sum` → `torch.stack(inputs).sum(0)`. 공유하는 인코더는 시점마다 적용되는 하나의 모듈이 된다.
**오라클:** 로워링된 모듈은 고정 입력에서 손으로 쓴 PyTorch 모듈과 같다(Python 게이트). 가중치
파일에는 공유 인코더의 사본이 하나만 들어 있다.

### 작업 N7: 단일 시점 학습

**Files:** `python/es/train_act.py`, `crates/es-data/src/training.rs`(`[run] single_view`), 테스트.

`[run] single_view = { weight = <alpha> }`: 시점들이 `Sum` 융합에서 만나는 그래프에서는, 배치마다
각 시점의 항만으로 계산한 손실도 `alpha`로 가중해 함께 계산한다(MAD의 "disentangle"). 이미지
특징에 대한 `Sum`이 없는 그래프에는 거부한다.

### 작업 N8: `es policy subset`

**Files:** `crates/es/src/cmd/policy.rs`, `crates/es-policy/**`(번들 재작성), 테스트.

`es policy subset --policy <bundle> --views <channel,...> --out <bundle>`: 같은 가중치에, 버려진
포트·인코더·`Sum` 항이 빠진 Observation IR과 Learning IR(버려진 인코더를 공유하던 남는 인코더는 그
가중치를 자기 이름으로 가져간다). 해시는 새로 만들어진다. **오라클:** 부분집합 번들의 출력이,
PyTorch에서 버려진 항을 제거한 전체 그래프와 같다. 시점들이 `Sum`에서 만나지 않는 그래프에는
거부한다.

### 작업 E2: 실험 2(오케스트레이터)

세 시점 MAD 정책(공유 인코더, `Sum`, 단일 시점 학습)을 세 카메라 모두로, 각각 하나씩으로, 그리고
실제 로봇 사례로서 손목 하나만으로 평가하고, E1의 정책들과 비교한다.

### 작업 N9: 에디터의 카메라 옵션

템플릿 옵션 "카메라: 머리 위 / 머리 위 + 손목 + 옆면"과, 시점들을 나란히 보여 주는 ⑤의 플레이어.
E1이 이 옵션을 제공할 가치가 있는지 알려 준 뒤에 계획한다.
