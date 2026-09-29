<!-- Korean translation of docs/design/multi-camera.md. The English file is the working copy; regenerate this when it changes. -->
# 멀티 카메라 정책 — 여러 시점으로 학습하고, 더 적은 시점으로 배포한다

설계 노트, 2026-09-29. M13/M14의 실제 실행 뒤 소유자가 요청했다("S3 끝나면 멀티 카메라 설계
노트부터 써줘"). 스펙: §6(Task IR), §7.2–§7.4(Observation IR, `MultiViewPack`, 두 시점 예제),
§8.3–§8.4(Learning IR, 퓨전), §14.2(`shared=True`), §15.2(아틀라스 하나, 카메라마다 타일 하나),
§19.1(멀티 카메라 데이터셋), §20.2(아틀라스 예산), §13.3(평가는 고정된다), §1.4(오라클 우선),
§5.3(해시 체인). 여기 있는 것은 아직 하나도 만들어지지 않았다. 6절은 소유자의 결정을
기록한다.

## 1. 왜

힌트 카드의 정책은 머리 위 카메라 하나가 가장 못 보는 두 순간에 실패한다. 큐브를 아예 들어
올리지 못하거나(96회 중 58–65회), 큐브를 통까지 옮기고 놓지 않는다(96회 중 20–22회) —
`docs/packets/M12/YV-verification.md`, `docs/packets/M14/QV-verification.md`. 잡기와 놓기는
가까운 거리에서 깊이에 의존하는 동작이고, 모방 학습의 표준 처방(ALOHA, ACT, 스펙 §28.9 자신의
"손목 카메라" 레버)은 손목에 다는 카메라다. 한 번도 통과하지 못한 카메라 전용 카드는 이것이 더
필요하다.

소유자는 **MAD**를 가리켰다 — *Merging and Disentangling Views in Visual Reinforcement Learning
for Robotic Manipulation* (Almuzairee, Patil, Bhatt, Christensen; CoRL 2025; arXiv 2505.04619;
코드 `github.com/aalmuzairee/mad`, MIT). 아이디어는 둘이다.

- **Merge:** 모든 시점이 공유 인코더 하나를 거치고, 시점별 특징이 (이어 붙이지 않고) **합산**되어
  액터와 크리틱이 읽는 특징이 된다.
- **Disentangle:** 학습 중에는 단일 시점 특징도 증강된 입력으로 `alpha` 가중치를 붙여 함께
  넣는다. 그래서 정책은 테스트 때 시점 하나만으로도 동작한다.

이것은 시각 RL(SAC 위의 DrQ)이고, 카메라 세 대(팔에 하나, 제3자 시점 둘)를 쓰는 Meta-World와
ManiSkill3에서, 실물 로봇 없이 수행됐다. 모든 카메라를 쓰는 멀티 시점 베이스라인보다 약 +30 %,
카메라를 빼도 견고하다고 보고한다. 모방 학습(ACT)에 적용한 사례는 찾지 못했다. 여기서는 가정이
아니라 측정하는 우리의 실험이 된다.

이것이 이 프로젝트에 주는 것: 손목 카메라와 머리 위 카메라로 학습하고 손목 카메라 하나로
배포하거나(값싼 실물 로봇 구성 — SO-101의 `camera_mount`), 카메라 하나가 고장 나도 계속
동작하는 것.

## 2. 지금 있는 것

트리 조사(2026-09-29)에서 **타입**은 준비돼 있고 **경로**는 그렇지 않음을 확인했다.

| 계층 | 카메라 여러 대를 말할 수 있다 | 카메라 여러 대를 돌린다 |
|---|---|---|
| 씬 | 그렇다. 바디 안의 `<camera>`는 바디에 붙고 렌더러가 매 프레임 바디를 따른다(`es-env/src/render.rs`) | SO-101에는 `camera_mount` 바디가 있고 그 안에 카메라가 없다 |
| Task IR | 그렇다. `ObservationSpec.channels`는 맵이고, 각 채널은 `ObservationSpec` 그래프 노드가 묶는 `Sensor`다 | `es-tools`는 이미지 채널이 둘 이상인 `--frames`를 거부한다. 카메라 둘인 픽스처가 없다 |
| 렌더러 | `Renderer::render(&[CameraView])`는 모든 카메라를 아틀라스 타일 하나씩에 넣는다(§15.2) | `EnvRenderer`는 env당 카메라 하나를 렌더한다. `EnvBatchRenderer`는 타일을 시점이 아니라 *env*로 만든다 |
| Observation IR | 그렇다. 독립된 `ImageInput` 체인, 각각 출력 포트 하나(§7.4의 예제). `MultiViewPack`과 `Mask`가 있다 | CPU 플랜은 독립 체인을 낮춘다. `MultiViewPack`과 `Mask`는 COMPILE-002. `FrameSource`, `Rollout`, `es dataset bake`는 이미지 하나를 가정한다 |
| Learning IR | 그렇다. 이미지 포트마다 `VisionEncoder` 하나, `Fusion { Concat \| CrossAttention \| FiLm \| AdaLn \| TokenConcat }` | `Sum`이 없다. 가중치 공유도 없다 — `python/es/builder.py`는 §14.2의 `shared=True`를 버리고("컴파일러 패스의 일") 그것을 구현하는 패스가 없다 |
| 데이터셋 | LeRobot v3 내보내기는 카메라마다 `observation.images.<name>`을 하나씩 쓴다 | `mirror_frames`는 평평한 타일 디렉터리 하나를 모든 카메라의 디렉터리에 하드 링크한다 — 카메라 둘이 같은 프레임을 받는다(잠재 버그) |
| LeRobot 경로 | LeRobot의 ACT 자체는 공유 백본 하나로 카메라 여러 대를 받는다 | 체크포인트를 다시 불러올 때 카메라 둘 이상을 거부한다(`es-policy/src/lerobot.rs`) |
| 없는 카메라 | — | 없다. 모든 입력 포트는 만들어져야 한다(XIR-010, LRN-011). 이미지가 없으면 평가, 베이크, 학습에서 오류다. `SensorDropout`은 안전 평면 워치독이지 정책 마스크가 아니다 |

두 번째 카메라는 `scene_hash`, `task_hash`, `observation_hash`를 옮기고, 이어서 `learning_hash`,
`policy_hash`, 데이터셋 스키마, `evaluation_hash`, `execution_hash`를 옮긴다. 해시 체인이 제 일을
하는 것이다. 문서를 고치는 게 아니라 새로 만든다.

## 3. 설계

### 3.1 새 문서, 옛 문서는 그대로

새 씬 `tests/fixtures/mjcf/so101_pick_place_views.xml` — 지금의 씬에 `camera_mount`의 카메라와
옆 카메라 하나를 더한 것(세 시점, 6절) — 과 그 옆에 새 Task IR,
Observation IR, Learning IR, Evaluation IR을 둔다. 커밋된 모든 문서와 숫자는 그대로 두고, 새
것은 자기 해시를 가진다.

### 3.2 모든 경로에서 카메라 여러 대 (스펙 결정 없음)

- `EnvRenderer`는 스텝마다 Task IR의 모든 이미지 채널을, 카메라마다 아틀라스 타일 하나로
  렌더한다(§15.2가 이미 기술한다). `EnvBatchRenderer`는 env × 시점으로 타일을 나눈다. 비트 단위
  오라클: 각 타일은 X3b가 env에 대해 고정한 것처럼 그 카메라의 단독 렌더와 같다.
- `--frames`는 `<frames>/<channel>/<NNNNNN>.bin`을 쓴다(v3 내보내기가 이미 기대하는 레이아웃).
  `es dataset bake`는 각 이미지 포트의 자기 디렉터리를 읽고, `mirror_frames`는 없어진다.
- `FrameSource`, `Rollout`, 평가 러너는 채널마다 프레임 하나를 받는다.
- 예산(§20.2)은 첫 `ImageInput`이 아니라 Task IR의 `views`를 센다.

### 3.3 IR 안의 MAD (스펙 결정 — 6절)

- **Sum 퓨전.** `FusionKind::Sum`(모든 입력의 너비가 같고, 출력은 그 합). 퓨전을 다시 학습하지
  않고도 시점 하나를 뺄 수 있게 해 준다. 남은 항의 의미가 유지되기 때문이다.
  `torch.stack(...).sum(0)`으로 낮추고, PyTorch 오라클은 손으로 쓴 모듈이다.
- **인코더 하나, 시점 여럿**(결정: 6절). 시점마다 자기 `VisionEncoder` 노드를 두고,
  `share = "<node>"`가 가중치를 쓸 인코더를 가리킨다; 지금의 항수 규칙은 그대로다.
- **단일 시점 학습**은 그래프가 아니라 학습 레시피의 몫이다. `[run] single_view =
  { weight = alpha }`는 `train_act.py`가 배치마다, 각 시점의 특징만으로 같은 퓨전을 통과시킨
  손실을 더하게 한다. IR 필드는 없다. Learning IR의 배치 의미는 자유롭다(§8).

### 3.4 더 적은 카메라로 배포하기: 파생 번들

`Sum` 퓨전과 공유 인코더가 있으면 시점을 빼는 것은 모든 가중치를 유지하는 그래프 편집이다.
시점의 `ImageInput` 체인, 그 인코더 적용, 그 퓨전 항을 제거한다. 그래서 선택적 포트(지금은
어디에도 없는 런타임 의미) 대신 **`es policy subset --views <names>`**가 새 번들을 쓴다. 가중치는
같고, 빠진 포트가 없는 Learning IR과 Observation IR, 새 해시를 가진다(다른 배포물이 *맞다*).
XIR-010과 LRN-011은 계속 성립한다. 오라클: 부분집합 번들의 출력은 빠진 시점의 항을 제거한 채
평가한 전체 그래프와 같고, PyTorch로 계산한다.

### 3.5 평가하기 (§13.3)

같은 Evaluation IR의 스위트, 시드, 지표, 합격 기준을 각 번들에 적용한다. 모든 시점, 그리고
각각의 단일 시점. 부분집합 번들의 Task IR은 카메라가 더 적으므로 그 Evaluation IR은 다른 과제와
관찰을 참조한다 — 문서는 정확히 그 참조에서만 다르고, 보고서가 그렇게 밝힌다. 사람이 비교하는
조건(스위트, 시드, 임계값)은 같다.

## 4. 실험, 순서대로

1. **두 번째 시점이 도움이 되긴 하는가?** 카메라 전용 IR 경로(`sim_cube_pose`가 없는 U3의
   그래프)를 머리 위 카메라에서, 그다음 머리 위 + 손목에서, `Concat` 퓨전, 같은 시연 200개를
   다시 렌더해서. 각각 학습 시드 둘(실행마다 ±0.3, M10 리뷰 S-4). 3.2절만 필요하다.
2. **MAD의 주장.** 공유 인코더, `Sum` 퓨전, 단일 시점 학습을 쓴 머리 위 + 손목. 두 카메라 모두,
   손목만, 머리 위만으로 평가하고 1단계의 단일 카메라 정책과 비교한다. 3.3과 3.4가 필요하다.
3. **에디터.** "카메라: 머리 위 / 머리 위 + 손목" 템플릿 옵션, 그리고 ⑤가 플레이어에서 시점을
   나란히 보여 주는 것.

이 PC에서의 비용: 렌더와 데이터는 카메라 수에 따라 늘고(카메라 카드의 2.9 GB는 둘일 때 약
6 GB가 된다), 학습 메모리는 가중치가 아니라 인코더 적용 횟수에 따라 는다.

## 5. 이 설계가 지키는 것

다섯 IR과 그 검증기. 안전 평면은 손대지 않는다(런타임에 카메라가 없으면 조용히 저하된 정책이
아니라 계속 `SensorDropout` 워치독 → 폴백이다). 해시 체인은 정직하게 유지한다(부분집합 번들은
새 번들이다). 골든은 생성기를 통해서만. 커밋된 모든 문서는 그대로.

## 6. 소유자가 정한 것 (2026-09-29)

1. **`FusionKind::Sum`을 Learning IR에 더한다**(§8.3의 목록에 들어간다; 스펙 글은 이를 구현하는
   패킷과 함께, `docs/ARCHITECTURE.ko.md`부터 고친다).
2. **가중치 공유는 인코더마다의 필드다:** 시점마다 자기 `VisionEncoder` 노드를 두고(오늘의 입력 한
   개 규칙은 그대로), `share = "<node>"`가 가중치를 쓸 인코더를 가리킨다. 낮추기는 모듈 하나를 만들어
   시점마다 적용한다; 다른 백본·폭·사전학습 출처의 노드를 가리키는 `share`는 이름을 대며 거부한다.
3. **배포 시 더 적은 카메라는 파생 번들이다**(`es policy subset --views <names>`), 선택적 입력이
   아니다.
4. **MAD처럼 카메라 셋:** 손목(SO-101의 `camera_mount` 안), 오늘의 머리 위 카메라, 옆 카메라 하나.
   실험 1은 한 시점(머리 위)과 세 시점을 비교하고; 실험 2는 세 시점 MAD 정책을 각 시점 하나만으로,
   그리고 실제 로봇 경우로 손목 하나만으로 배포한다.

플랜 N(`docs/packets/M15/plan-n.md`)은 3.1–3.5절을 이 순서의 패킷으로 만든다: 여러 카메라 경로(3.2)와
새 문서(3.1), 실험 1, 그다음 `Sum`, `share`, 단일 시점 학습과 `es policy subset`, 실험 2, 그다음
에디터.
