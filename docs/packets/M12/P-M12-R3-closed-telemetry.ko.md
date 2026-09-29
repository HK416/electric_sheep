<!-- Korean translation of docs/packets/M12/P-M12-R3-closed-telemetry.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R3 — 에디터가 닫힌 텔레메트리 연결을 알아챈다

2026-09-29 Y-V 항목 1에서 발견했다. `docs/design/editor-redesign.md` 6.4절(다시 연 run, 신호등),
`docs/design/editor-shell.md` 13절(attach), 패킷 Y12에 적어 둔 한계("attach한 run의 프로세스가
죽으면 프로젝트를 다시 열 때까지 *진행 중*, 이어서 *응답 없음*으로 읽힌다").

## 결함

`crates/es-editor/src/model/telemetry_view.rs`의 `source_of`는 닫히거나 끊긴 연결을 영원히 "지금은
없음"으로 바꾼다(끊긴 소켓 때문에 탭이 지워지지 않게 한, M7/E4의 의도된 선택). `model/watch.rs`는
attach한 run을 살아 있는 것으로 센다(`Child::NotOurs`). 그래서 run의 `es`가 끝난 뒤에도 ③/④ 신호등은
끝없이 *응답 없음*을 가리키고, 에디터 밖에서 재개한 run(`telemetry.txt`에 적힌 새 `--telemetry`
주소)은 끝내 잡히지 않는다. 이 PC에서 본 일: 에디터가 카메라 전용 run의 첫 시도(포트 7812)에
attach했고, 그 시도는 실패해 종료했다. 재개한 시도는 7813에서 발행했는데, 새 run이 학습하는 동안
에디터는 이미 없는 run을 두고 *응답 없음*을 계속 띄웠다.

## 스펙

- `Source`가 닫혔다고 말할 수 있다: 텔레메트리 클라이언트의 `Err(_)` 중 `WouldBlock`이 아닌 것은
  모델이 읽을 수 있는 플래그를 세운다(탭에 이미 보이는 것은 그대로 둔다 — E4의 의도는 유지된다).
- `Watch`: 소스가 닫힌 attach는 더 이상 attach가 아니다. 단계들은 디스크(`RunFacts`)로 돌아간다.
  에디터가 attach한 적 없는 run과 똑같이 — 재개 지점을 가진 *중단됨*, *완료*, 또는 *실패*.
- 최신 run이 디스크상 끝나지 않았고 아무것도 attach되어 있지 않은 동안, `Watch`는 `telemetry.txt`를
  다시 읽어 그 주소로 다시 dial한다. 많아야 `REDIAL_S`(10초)에 한 번, 오늘의 dial처럼 UI 스레드
  밖에서. 그래서 다른 곳에서 재개한 run을 프로젝트를 다시 열지 않고도 잡는다.
- 에디터가 직접 시작한 run이 자식 프로세스가 살아 있는 동안 보여 주는 것은 바뀌지 않는다.

## context

```
crates/es-editor/src/model/telemetry_view.rs
crates/es-editor/src/model/watch.rs
crates/es-editor/src/ui/train.rs
crates/es-editor/src/app.rs
docs/packets/M12/P-M12-R3-closed-telemetry.md
docs/packets/M12/P-M12-R3-closed-telemetry.ko.md
```

## oracle

1. `watch.rs`의 헤드리스 테스트: 닫혔다고 보고하는 attach된 소스 → 신호등은 결코 *응답 없음*이
   아니다. 단계들은 `phases(Some(&facts), None)`과 같다. 나중에 새 주소가 적힌 `telemetry.txt`는
   `REDIAL_S` 뒤에 다시 dial되고, 그 전에는 되지 않는다(기존 테스트처럼 시계를 주입한다).
2. `telemetry_view.rs`의 테스트: 상대가 연결을 닫은 클라이언트가 닫혔다고 보고한다.
3. `cargo test -p es-editor`, clippy `-D warnings`, fmt, `check-scope`,
   `cargo xtask context-budget` — `es-editor`는 10,000 아래에 머물러야 한다(지금 9,715).

## forbidden

연결이 끊길 때 탭에 보이는 것을 지우는 것; UI 스레드에서 `telemetry.txt`를 폴링하는 것; `es`의
어떤 변경; 원격 서버 접속.
