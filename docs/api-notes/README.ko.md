<!-- Korean translation of docs/api-notes/README.md. The English file is the working copy; regenerate this when it changes. -->

# API notes

에이전트 학습 데이터가 낡았거나 틀리기 쉬운 외부 API 표면을 고정해 둔 요약본 —
§1.7이 "환각 API"(hallucinated API)라고 부르는 실패 양상이다. 라이브러리당 파일
하나, 그 이름을 따서 명명한다 (`ash.md`, `slang.md`, `torch.md`,
`lerobot-schema.md`, …).

## Convention

- 모든 요약은 파일 맨 위에 그것이 채택한 **정확히 고정된 버전**(`Cargo.lock` /
  `pyproject.lock` / 서브모듈 커밋 해시)을 명시한다.
- 요약은 실제로 그 버전에 대해 검증한 것만 기록한다 — 실제 타입과 함수 시그니처,
  실제 필드 이름 — 그럴듯해 보이는 것이 아니다. 어떤 패킷이 아직 여기 없는
  시그니처를 필요로 한다면, 그것에 기대어 코드를 작성하기 전에 (설치된 버전에 대해
  검증한 뒤) 먼저 추가한다.
- 고정된 버전이 올라갈 때마다 요약도 갱신한다. 낡은 요약은 없는 것보다 나쁘다 —
  조용히 틀린 채로 두지 말고 삭제하거나 갱신한다.
- 이 파일은 규약만 정리해 둔 색인이다. 아직 어떤 라이브러리 요약도 존재하지
  않는다 — `ash`, Slang, `torch`, LeRobot 데이터셋 스키마 중 어느 것도 아직
  워크스페이스에 들어오지 않았다 (`docs/ARCHITECTURE.ko.md` §28.2에 따르면 Wave 0는
  검증 인프라만 다룬다).

## Planned files

- `ash.md` — `es-gpu`/`es-render`가 실제로 사용하는 Vulkan 바인딩 표면.
- `slang.md` — Slang → SPIR-V 컴파일러 호출과 reflection API.
- `torch.md` — `es-policy`와 Learning IR의 PyTorch 참조 오라클(§1.4)이 사용하는
  `tch`/libtorch C++ 표면.
- `lerobot-schema.md` — Observation IR과 policy-equivalence 오라클(§1.4)이 사용하는
  LeRobot 데이터셋·체크포인트 스키마.
