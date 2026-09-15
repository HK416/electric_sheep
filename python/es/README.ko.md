<!-- Korean translation of python/es/README.md. The English file is the working copy; regenerate this when it changes. -->

# `es` — Python 저작(authoring) 빌더 (spec 14.2)

`es_native` pyo3 확장(`crates/es-py`) 위의 얇은 Python 프론트엔드이며, 그 확장은 다시
언어 중립적인 Rust 빌더 코어를 호출한다. 레이어링에 대해서는 `docs/design/python-builder.md`
참고.

## Rust 쪽이 정본(canonical)이다

`builder.py`의 몇몇 작은 공식들은 `es_native`를 호출하는 대신 로컬에서 Rust 로직을
재구현한다. 노드가 아직 만들어지기 전에 클라이언트 측 타입 추론을 위해 그 값이 필요하기
때문이다(에디터는 그래프를 한 번 실행하기도 전에 포트의 shape를 알아야 한다):

- `stable_id(path)`는 `es_core::StableId::from_path`(`crates/es-core/src/id.rs`)를
  미러링한다: `blake3(path)`를 16바이트로 자르고 hex로 인코딩한 것이다.
- `_feature_ty(dim, tokens)`와 `_chunk_ty(horizon, action_dim)`은
  `crates/es-ir/src/learning.rs`의 `feature()`/`chunk()`를 미러링한다.

**세 가지 모두에 대해 Rust가 진실의 원천이다.** 여기의 값이 Rust 구현과 어긋나면 Rust
구현이 옳고 고쳐야 할 쪽은 이 파일이다. `python/es/selfcheck.py`는 함수마다 골든 벡터
하나씩을 고정하며, 이는 `crates/es-core/src/id.rs`의 `from_path_is_stable_and_distinct`
테스트에 고정된 리터럴과 일치한다. 따라서 어느 쪽이든 어긋나면 잘못된 바디에 대해 검증되는
IR을 조용히 만드는 대신 요란하게 실패한다 (M4 리뷰 S-14).

## 셀프 체크 실행

```
maturin develop --release --features python   # from this directory, once, to build es_native
PYTHONPATH=python <venv>/Scripts/python.exe -m es.selfcheck
```

## `encode_video.py`

위 빌더와는 무관하다: `encode_video.py`는 `es video mosaic`가 만든 raw 프레임 출력을
`.mp4`로 변환하는 독립 스크립트다 (M5 V4, 설계 노트 `docs/design/visible-learning.md`
2.9절, 9절). `cv2.VideoWriter`를 쓰며 fourcc는 `mp4v`다 — 오라클 서버에는 `ffmpeg`
바이너리가 없고 이 코덱만 열린다. 이 패킷에서 유일한 Python 단계이며, `es video mosaic`
자체는 순수 Rust다. `opencv-python`이 설치된 Python이 필요하다:

```
<venv>/bin/python python/es/encode_video.py --frames <mosaic dir> --out demo.mp4 --fps 10
```

## `train_act.py`

이것도 위 빌더와는 무관하다: `train_act.py`는 스펙 2.3 학습 분할의 옵티마이저 쪽이다
(M5 V2/V2b, 설계 노트 `docs/design/visible-learning.md` 6절, 7.6절, 7.9절). 그 패킷에서
**유일한** Python이다 — `es policy lower`, `es dataset bake`, `es policy pack`은 Rust이고
인터프리터가 필요 없다.

```
es policy lower --policy untrained.esb --out build/
es dataset bake --policy untrained.esb --out baked/ --frames tiles/ ds/
<venv>/bin/python python/es/train_act.py --module build/ --baked baked/ --out model.safetensors \
    [--epochs N] [--batch N] [--lr F] [--seed N] [--device cuda] \
    [--checkpoint-at 1000,5000,20000] [--loss-curve curve.json]
es policy pack --policy untrained.esb --weights model.safetensors --out trained.esb
```

`es policy lower`가 쓴 `build/es_policy.py` — 번들 자신의 `LearningGraph`로부터
`es_policy::lower::lower_to_torch`가 생성한 모듈 — 을 `exec`하고, 그것만 최적화한다.
**레이어를 하나도 정의하지 않는다**: 아키텍처는 Learning IR에서 오거나 아예 오지 않는다.
이것이 스펙 1.4의 "같은 IR을 PyTorch로 돌린 것이 ground truth"를 근사적으로가 아니라
문자 그대로 참으로 만든다. `crates/es-policy/tests/ir_training.rs`가 양쪽을 모두 검사한다
(로워링과의 바이트 일치 검사, 그리고 이 파일에 대한 소스 스캔).

`build/contract.json`이 선언한 그대로의 키로 safetensors를 쓰므로, `es policy pack`이
번들에 받아들이기 전에 모든 키와 shape을 검사할 수 있다 (스펙 25.1). 이 경로 어디에서도
로드 시 코드를 실행할 수 있는 포맷은 읽지도 쓰지도 않는다 (`INV-16`).

알아둬야 할 세 가지, 모두 설계 노트에 기록되어 있다:

- **Observation IR 노드를 하나도 구현하지 않는다.** `--baked`는 `es dataset bake`의 출력이고,
  그 명령은 기록된 모든 프레임을 `es eval run`이 추론에서 돌리는 것과 같은 `CpuPlan`으로
  통과시켰다. V2는 parquet을 `pyarrow`로 읽고 여기서 `Op::Dequantize`를 다시 구현했으며,
  Observation IR이 정규화하는 상태 포트에 원본 `observation.state` 행을 먹였다 — 그래서 정책은
  한 관측으로 학습하고 다른 관측으로 평가받았다 (7.9절, 열린 질문 11). 이 파일이 이제 포맷
  하나만 읽는 이유다;
- 구운 텐서가 없는 contract 입력은 0으로 채워진 포트가 아니라 거부다. V2가 경고해야 했던
  "조용히 0" 실패 모드가 은퇴한다;
- 로워링된 모듈은 single-sample이므로, `--batch N`은 배치 forward 한 번이 아니라 N개 샘플을
  한 optimizer step으로 누적한다.

`torch`와 `torchvision`이 설치된 Python이 필요하다. `pyarrow`는 더 이상 여기서 읽지 않는다 —
데이터셋 읽기는 `es dataset bake`가 Rust로 한다.
