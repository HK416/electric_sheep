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
