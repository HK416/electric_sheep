<!-- Korean translation of docs/packets/M12/P-M12-R6-backend-last-words.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R6 — 죽은 백엔드 프로세스가 이유를 남긴다

리뷰 지적 S-6(`docs/reviews/M12.md`): 평가 워커 하나의 MuJoCo 프로세스가 한 번
`backend process died: 파이프가 닫히는 중입니다. (os error 232)`로 죽었고, 원인은 알 수 없다.
`crates/es-physics-backend/src/proc.rs`가 stderr를 버리고(`Stdio::null()`), 쓰기가 실패하면 Python
쪽이 stdout에 남긴 오류 줄을 읽지 않기 때문이다(`mujoco_ref.py`는 import가 실패하면
`{"ok": false, "error": "import failed: …"}`를 쓰고 종료한다 — 그 뒤의 쓰기는 os error 232로
실패하고 그 줄은 사라진다).

## 스펙

- `Process::call`: 쓰기가 실패하거나 읽기가 EOF를 돌려주면, 보고하기 전에 프로세스의 마지막 말을
  모은다. stdout에 이미 와 있는 완전한 줄(프로토콜 오류 줄이면 디코드해 그 `error` 텍스트를 쓴다),
  종료 상태(짧은 제한 대기 뒤의 `try_wait`), 그리고 stderr의 끝부분(크기 제한).
- stderr는 리더 스레드가 크기 제한이 있는 링(마지막 64줄 또는 16 KiB)에 받아 둔다. 그래서 읽지
  않은 파이프가 Python 쪽을 막는 일은 없다(원래 null로 돌린 이유). 링은 실패했을 때만 읽는다.
- 오류는 계속 `PhysicsError::ProcessDied`다. 텍스트는 예컨대 `backend process died (exit code 1):
  import failed: DLL load failed while importing … — stderr: …`가 된다. 성공 경로에서는 아무것도
  바뀌지 않는다. 해시 변화 없음, 리더 스레드 말고는 핫 패스에 시간 비용도 없다.
- `crates/es-policy`의 torch 런타임 프로세스가 같은 패턴(`the backend process died`)을 쓰면, 거기도
  같은 방식으로 고친다.

## context

```
crates/es-physics-backend/src/proc.rs
crates/es-physics-backend/tests/**
crates/es-policy/src/torch_runtime.rs
crates/es-policy/src/runtime.rs
crates/es-policy/tests/**
docs/packets/M12/P-M12-R6-backend-last-words.md
docs/packets/M12/P-M12-R6-backend-last-words.ko.md
```

## oracle

1. 대역 프로세스(기존 proc 테스트처럼 플랫폼의 인터프리터, 또는 설정돼 있으면 `ES_PYTHON`)를 띄우는
   테스트. 대역은 첫 요청 전에 프로토콜 오류 줄을 찍고 1로 종료한다. `call`은 os error 232가 아니라
   디코드한 오류 텍스트와 종료 코드를 보고한다.
2. 대역이 stderr에 1 MB 넘게 쓴 뒤 답하는 테스트: 교착 없이 답을 읽는다.
3. 기존 백엔드·정책 테스트가 통과한다. `ES_PYTHON`을 설정하면 MuJoCo 테스트가 실제로 돈다.
4. fmt, clippy `-D warnings`, `check-scope`.

## forbidden

프로토콜이나 `mujoco_ref.py`의 동작 변경, 해시 변경, 원격 서버 접속.
