<!-- Korean translation of docs/design/telemetry-protocol.md. The English file is the working copy; regenerate this when it changes. -->

# 텔레메트리 프로토콜 — 와이어 형식과 M1 루프백 트랜스포트

`es-telemetry::protocol`과 `es-telemetry::transport`에 대한 설계 노트다. Spec: spec 23.1
(백엔드 중립, 에디터는 클라이언트), spec 23.2–23.4 (계층 그래프 뷰, 성능 게이트),
spec 25.1 (보안), spec 25.3 (API 버전 관리), spec 12.4 (성능 지표 집합). 이 문서는 spec 28.3이
W8 패킷(`docs/packets/M1/W8-telemetry-transport.md`)의 선행 조건으로 지정하는 M1 설계 문서다.

## 1. 이 문서가 존재하는 이유

Spec 23.1: **"에디터는 학습을 호스팅하지 않는다. 실행 중인 프로세스에 접속하는 클라이언트다."** 이는
접속할 *대상*이 있어야만 성립한다: 프로세스(시뮬레이터, 임베디드 런타임, 또는 순수 Python 학습
루프)가 소켓을 열고 상태를 스트리밍하며, 임의 개수의 리더 — egui 에디터, CLI, 로깅 스크립트 —
가 프로듀서의 코드 경로를 바꾸지 않고 여기에 붙는다. Spec 23.1은 또한 이 프로토콜이 Electric
Sheep 전용이 아니어야 한다고 못박으며("얇은 Python 어댑터... 가 채택의 쐐기다"), 그래서 와이어
스키마(`es_telemetry::protocol`)는 ES 전용 개념을 하나도 담지 않는다: 스트림 id, 틱, 페이로드가
전부다.

이 문서는 의도적으로 분리된 두 계층을 다룬다.

- **스키마** (`protocol.rs`): 메시지가 어떻게 전달되는지와 무관하게, 메시지가 *무엇인지*;
- **트랜스포트** (`transport.rs`, W8에서 추가): 로컬호스트에서 스키마를 처음부터 끝까지
  검증하기에 충분한 `std::net::TcpListener`/`TcpStream` 구현. 최종 트랜스포트가 아님을
  명시적으로 밝힌다 — §6 참고.

## 2. 메시지 흐름

```
   client (editor / lerobot-adapter / CLI)              server (sim / embedded runtime)
            |                                                       |
            |  TCP connect ------------------------------------->   |
            |                                                       |
            |  Hello { versions_supported, token, client } ----->   |  negotiate() + token check
            |                                                       |
            |  <----------------------- HelloAck { version,         |  (accepted)
            |                            session_id,                |
            |                            execution_hash }           |
            |                    or                                 |
            |  <----------------------- Bye { reason }               |  (rejected, socket closes)
            |                                                       |
            |  Subscribe { streams: [..] } ----------------------->  |  replaces subscription set
            |                                                       |
            |  <====================================================|  Frame, Frame, Frame, ...
            |                                                       |  (only for subscribed streams)
            |  Subscribe { streams: [..] } ----------------------->  |  (may re-subscribe any time)
            |                                                       |
            |  Bye { reason } ------------------------------------>  |  (either side may end it)
```

TCP 연결 하나가 세션 하나다. 핸드셰이크 이후 관계는 비대칭적이다: 클라이언트가 내보내는 메시지는
`Subscribe`(뷰가 바뀔 때마다)와 `Bye`뿐이고, 서버가 내보내는 메시지는 `Frame`뿐이다(해당 프레임의
스트림을 구독 중인 클라이언트들에게 팬아웃된다) — 어느 스레드가 무엇을 쓰는지는
`crates/es-telemetry/src/transport.rs`의 모듈 문서를 참고하라. 두 스레드가 동기화 없이 같은
소켓에 쓰면 바이트가 뒤섞이기 때문이다.

## 3. 프레이밍

`protocol::encode`/`decode`: `codec (1 byte) | body_len (u32, LE) | body`. `codec = 0`은
현재 JSON이다; 바이너리 코덱은 와이어 브레이크가 아니라 추가 variant이므로, JSON이 빠른 것보다
`Codec`이 enum으로 남아 있는 것이 더 중요하다. `decode_with_max`는 바이트를 건드리기 전에
과도하게 큰 `body_len`을 거부하므로, 손상된 길이 프리픽스로 큰 할당을 강제할 수 없다.

이는 트랜스포트가 M1 TCP shim이든 이후의 QUIC 스트림이든 의도적으로 동일한 프레이밍이다:
`transport.rs`는 `TcpStream` 주위에 "하나의 `decode`가 성공할 때까지 읽고, 반복" 로직만 추가할
뿐, 프레임 바이트의 어떤 부분도 TCP를 가정하지 않는다.

## 4. 인증 (spec 25.1)

`Server::bind(addr, token: Option<String>)`. `token`이 `Some`일 때, 모든 `Hello.token`은
정확히 일치해야 하며 그렇지 않으면 서버는 `Bye { reason: "missing or invalid token" }`을
회신하고 클라이언트를 등록하지 않은 채 연결을 닫는다. `token`이 `None`일 때는 아무 검사도
실행되지 않는다 — 이것이 로컬 개발 형태다(`es --check-deps`가 토큰이 설정되지 않은 로컬
시뮬레이터에 대해 에디터를 실행하는 경우).

비교 자체는 `String`에 대한 `PartialEq`가 아니라 `transport::ct_eq`이며, 조기 반환 없이
`max(a.len(), b.len())` 바이트에 걸친 상수 시간 폴드(fold)다(`subtle` 의존성 없이 — 다섯
줄): 순진한 `!=` 비교는 처음 다른 바이트에서 멈추는데, 이는 타이밍을 통해 추측한 토큰이 얼마나
맞았는지를 유출시킨다. 길이 불일치는 `!=`로 조기 종료되는 대신 같은 누산기에 미리 접어 넣어진다.

토큰 검사 앞에는 두 가지 자원 제한이 더 있으며, 둘 다 `Server::bind_with(addr, token, cfg:
ServerConfig)` 위에 있다(`Server::bind`는 이를 `ServerConfig::default()`로 호출한다):

- **연결 상한** (`ServerConfig::max_clients`, 기본값 `DEFAULT_MAX_CLIENTS = 64`): accept
  루프 자체에서, 연결에 대해 스레드가 스폰되기도 전에 검사된다 — 즉 `Hello`를 읽기도 전이며,
  이는 버전/토큰 검사보다도 전이라는 뜻이다. `in_flight = handshaking.load() +
  clients.len()`은 미드핸드셰이크든 이미 등록됐든 현재 "슬롯"을 차지하고 있는 모든 연결을
  센다; 도착한 연결이 `>= max_clients`이면 accept 루프 스레드가 동기적으로 `Bye { reason:
  "too many clients" }`를 쓰고, 스레드를 스폰하거나 바이트 하나 읽는 일 없이 소켓을 드롭한다
  (닫는다). 이는 원래 형태(토큰/버전 검사 이후, 연결별 스레드 안에서 상한을 검사하던 방식)와
  다른 동작 변경이다: 상한을 초과한 연결의 `Hello` — 잘못된 토큰을 가진 것이라도 — 는 결코
  읽히지 않으므로, 토큰별 또는 버전별 사유가 아니라 오직 "too many clients"만 받는다. accept
  루프는 단일 스레드이므로 이 검사-후-증가에는 자기 자신과의 경합이 없다 — 두 연결이 경계를
  넘어 경합할 수 있다던 앞선 `ponytail` 노트는 더 이상 적용되지 않는다; 여기서 정확함을
  만드는 것은 (유지되는 뮤텍스가 아니라) 이 순서다.
- **핸드셰이크 타임아웃** (`ServerConfig::handshake_timeout`, 기본값 `HANDSHAKE_TIMEOUT =
  5s`): 개별 읽기 단위가 아니라 절대 데드라인이다. accept 루프는 연결이 admit될 때
  `Instant::now() + handshake_timeout`을 기록한다; `read_message_until`은 블록될 수 있는
  모든 소켓 읽기 전에 *남은* 시간을 다시 계산해 그것을 해당 읽기의 타임아웃으로 사용하며,
  남은 시간이 없어지면 오류를 낸다. 몇 초마다 한 바이트씩 흘리는 피어 — 각 개별 읽기는 자신의
  읽기별 타임아웃 안에 넉넉히 들어가더라도 — 는 전체 경과 시간이 데드라인을 넘는 순간 여전히
  끊긴다. 이는 한 번 적용된 단일 `set_read_timeout(Some(duration))`(원래 형태)로는 할 수
  없던 일이다: 그 호출은 하나의 읽기만 경계 지을 뿐, 여러 부분 읽기의 합을 경계 짓지 않는다.
  성공적인 `Hello` 읽기 직후 해제되어(`set_read_timeout(None)`), 세션의 이후 읽기
  (`Subscribe` 루프)는 연결이 살아있는 동안 정상적으로 블록된다.
- **관측 가능성**: `Server::stats()`는 클라이언트별 `sent`/`dropped` 카운터와 더불어
  `handshaking`(상한 검사를 통과해 admit됐지만 아직 핸드셰이크를 통과하지 못한 연결)과
  `threads_live`(핸드셰이크 중이든 등록됐든, 현재 실행 중인 모든 `serve_client` 스레드)를
  보고하므로, 호출자 — 또는 테스트 — 는 상한이 그저 신뢰의 대상이 아니라 실제로 지켜지고
  있음을 볼 수 있다. 둘 다 공유 `AtomicUsize` 카운터이며, `threads_live`는 스레드 종료 시
  RAII 가드에 의해 감소되므로 모든 경로(정상 반환, 이른 `?`, 패닉 언와인드)가 정확히 한 번씩
  계산에 반영된다.

**이것이 아닌 것**: 토큰은 JSON 본문 안에 평문으로 전달되며, 채널 암호화는 존재하지 않는다.
같은 사용자가 소유한 프로세스 간 `127.0.0.1`(spec 25.1의 기본 바인드)에서는 문제가 없지만,
신뢰할 수 없는 네트워크에서는 그렇지 않다. TLS는 이 표면에 대한 spec 25.1의 또 다른 요구사항이며
여기서는 구현되지 않는다 — §6 참고.

## 5. 버전 관리 (spec 25.3)

`negotiate(client_versions, server_versions) -> Option<u32>` (이 패킷 이전부터 변경 없음)는
양쪽이 모두 나열한 것 중 가장 높은 버전을 선택한다. 서버는 `{PROTOCOL_VERSION,
PROTOCOL_VERSION - 1}` — "N과 N-1" — 을 제시하므로, 이전 릴리스에 맞춰 빌드된 에디터도 롤아웃
중 더 새로운 런타임에 접속할 수 있다; `PROTOCOL_VERSION`이 `1`일 때는 아직 N-1이 없으므로
서버는 `{1}`만 제시한다. 공유 버전이 없으면 핸드셰이크 거부(`Bye`, reason에 양쪽 목록을 명시)일
뿐, 조용한 다운그레이드나 패닉이 아니다.

`Message`, `Frame`, `Payload`, `PerfMetrics` 자체에는 버전 필드가 없다: spec 25.3은 버전을
*핸드셰이크 안에* 두는데, 세션은 한 번 협상한 뒤 그 수명 동안 하나의 방언만 말하기 때문이다.
향후 호환되지 않는 스키마 변경은 `PROTOCOL_VERSION`을 올린다; 추가적인 변경(새 `Payload`
variant, 새 옵션 필드)은 그럴 필요가 없다 — `docs/design/policy-bundle.md`의 매니페스트
형식이 이미 `serde`의 `Option` 필드로 이 논리를 취하고 있는 것과 같다.

## 6. 백프레셔

Spec 23.4의 게이트 9 — 학습/시뮬레이션 루프에 대한 텔레메트리 오버헤드는 1% 미만이어야 한다 —
는 네트워크가 아니라 *프로듀서*에 대한 진술이다. `Server::publish`는 어떤 구독자가 무엇을
하고 있든 상관없이 유한한 시간 안에 반환해야 한다:

- 각 클라이언트는 자신만의 유한 큐(`CLIENT_QUEUE_CAPACITY = 16` 프레임)를 가진다;
- `publish`는 논블로킹 `try_send`를 수행한다; 큐가 가득 차면 프레임은 드롭되고 해당
  클라이언트의 `ClientStats::dropped`에 집계될 뿐, 결코 대기하지 않는다;
- 느린 클라이언트 — 읽기를 멈췄거나 바쁜 이미지 스트림에서 뒤처진 클라이언트 — 는 오직 자신의
  데이터만 잃는다. 시뮬레이션이나 다른 구독자, 심지어 자기 연결의 생존성조차 늦출 수 없다(큐는
  해당 클라이언트의 라이터 스레드에서 독립적으로 비워진다).

16이라는 값은 "몇 틱 정도의 여유"를 위한 추정치이지 측정치가 아니다 — M1 패킷의 테스트
(`a_slow_client_drops_frames_without_blocking_the_publisher`)는 정책의 *형태*(드롭이 발생하고
프로듀서는 결코 블록되지 않는다)만 증명할 뿐, 16이 실제 그래프 뷰 세션에 맞는 깊이라는 것까지
증명하지는 않는다. 그 숫자를 조율하고 게이트 9의 실제 "< 1%" 질문에 답하려면 실행 중인 학습
루프에 대해 실행 중인 에디터가 필요하다 — 단위 테스트가 만들어낼 수 있는 범위 밖이다.
**`Target / Status: 미검증 (unverified)`** — < 1% 수치 자체에 대해서다; 이 문서는 그것이
참이 되게 해줄 메커니즘(논블로킹 팬아웃)이 갖춰져 있다는 것만 확립한다.

## 7. Python 어댑터에 필요한 것

Spec 23.1의 백엔드 중립 요구사항은 LeRobot/Isaac Lab/Newton 학습 루프가 어떤 Rust 코드도
링크하지 않고 프레임을 내보낼 수 있어야 한다는 뜻이다. `protocol.rs`의 크레이트 레벨 문서
주석이 이미 JSON 본문 형태를 전부 설명하고 있으며, W8은 다운스트림 Rust 리더에게 그저
파싱되는 것이 아니라 실행 중인 `Server`에 *푸시*하고자 하는 Python 클라이언트를 위해 요구사항을
하나 더 추가한다:

1. 런타임이 출력했거나 설정된 주소로 TCP 연결을 연다.
2. 길이 프리픽스가 붙은 `Hello` 하나를 보낸다(`{"type": "hello", "versions_supported": [1],
   "token": null_or_the_configured_token, "client": "lerobot-adapter"}`), §3의 5바이트
   `codec | len` 헤더를 사용한다(`codec = 0`).
3. 길이 프리픽스가 붙은 회신 하나를 읽는다. `"type": "hello_ack"`이면 계속 진행하고,
   `"type": "bye"`이면 연결이 닫히는 중이며 `reason`이 그 이유를 말해준다(잘못된 토큰 또는
   버전 불일치 — 서버는 거부된 소켓에서 계속 듣지 않으므로, 새 연결에서 수정된 `Hello`로
   재시도한다).
4. `Frame` 메시지를 직접 보낸다 — 프로듀서는 아무것도 `Subscribe`할 필요가 없다; `Server`의
   팬아웃을 구독하는 것은 *리더*뿐이다. 스칼라와 `Metrics` 페이로드만 계속 푸시하는 학습
   루프는 텐서/이미지 인코딩을 전혀 필요로 하지 않는다.

protobuf도, 스키마 컴파일러도, 소켓과 JSON 인코더 외의 어떤 의존성도 필요 없다 — 이들
프레임워크 모두 이미 손에 쥐고 있는 것들이다.

## 8. 알려진 한계 (이 패킷)

- **`Server`에 종료 핸드셰이크가 없다.** `Server::bind`의 accept 스레드는 프로세스 수명
  내내 실행되며, 그것을 조인하는 `Server::close`가 없다. 리스너가 살아야 하는 만큼만 정확히
  사는 임베디드 런타임이나 시뮬레이터 프로세스에는 문제없다; 같은 주소를 나중에 다시 바인딩해야
  하는 호출자는 셧다운 플래그와 셀프-커넥트 웨이크업이 필요하지만, 여기서는 구축되지 않았다.
- **TLS 없음, QUIC 없음.** Spec 23.4는 원격 에디터 경로에 QUIC을, spec 25.1은 TLS를
  원하지만, 둘 다 `es-telemetry::transport`에는 존재하지 않는다. 둘 다 `es-transport`
  (layer 11, `CLAUDE.md`에 따르면 CUDA/HIP을 링크할 수 있는 유일한 크레이트)의 작업으로
  나중에 처리된다 — 이 트랜스포트는 그 크레이트가 존재하기 전에 M1 게이트가 스키마와
  백프레셔 정책을 검증할 수 있게 해주는 루프백/신뢰된 네트워크용 shim이다.
- **큐 깊이가 예산이 아니라 고정값이다.** `CLIENT_QUEUE_CAPACITY`는 상수이지, spec 23.3이
  상태 스트리밍에 대해 서술하는 "스텝당 20 µs 샘플링 예산"이 아니다. 그 예산을 실제 레이트
  리미터(깊이 리미터가 아니라)로 바꾸는 것은 측정할 실제 프로듀서가 생긴 이후의 작업이다.
- **재접속/재개 없음.** TCP 연결이 끊기면 클라이언트의 구독 상태를 잃는다; 재접속 후 처음부터
  다시 구독한다. `session_id` 기반의 재개는 존재하지 않는다.

## 9. 이 저장소의 프로듀서 (패킷 M7/E4)

§7은 *외부* 프로듀서가 무엇을 보내야 하는지를 말한다. 이것은 이 저장소 안의 프로듀서가 보내는
것이다: `es eval run --telemetry <addr> [--telemetry-token <t>] [--telemetry-image-every <N>]`
는 번들이나 씬, 두 파이썬 인터프리터 중 어느 것도 열기 전에 `Server`를 바인드하고 — 그래서
출력된 주소로 붙는 뷰어가 첫 셀 전에 구독을 마친다 — 네 스트림으로 발행한다:

| 스트림 | 페이로드 | 언제 |
|---|---|---|
| `1` | `Event { kind: "cell.begin" \| "cell.end" \| "suite.end", fields }` | `cell.begin`은 에피소드 시작(`cell`, `suite`, `seed`, `episode`), `cell.end`는 그 파일들이 쓰인 뒤(`cell`, `outcome`, `steps`, `frames`, `traj`), `suite.end`는 §10.1 행이 측정된 뒤(`suite`, `n_episodes`, `metric.<이름>`은 `MetricValue` 자신의 JSON) |
| `2` | `Scalars([frame, tick, source, 위반 비트])` | 관측을 뚴 모든 제어 틱 |
| `3` | `Metrics(PerfMetrics)` | `cell.end`마다: `actions_per_sec`, `policy_inferences_per_sec`, `chunk_underrun_rate`. 나머지는 전부 `None`이다 — 이 실행은 종단간 지연도 GPU 메모리도 측정하지 않고, §12.4는 0보다 빈칸을 원한다 |
| `4` | `Image { format: "rgb8" }` | `--telemetry-image-every N` 틱마다; `0`(기본값)은 하나도 발행하지 않는다 |

스트림 id는 **스키마가 아니라 데이터**다: `protocol.rs`는 그 어느 것도 이름짓지 않고, 모르는
스트림을 받은 소비자는 그냥 무시하며(`es_editor::model::live_run`이 정확히 그렇게 한다), 다른
프로듀서가 다른 번호를 골라도 여기서 깨지는 것은 없다. `Frame::stream`은 `u32`다.

스트림 2는 한 샘플당 `es_eval::runner::StepEvent` 하나를 펼친 것이다: 셀 안의 프레임 인덱스,
그것이 돌은 `PhysTick`, `es_data::ActionSourceCode`의 번호로 된 행동의 출처(`Policy 0, Clamped 1,
Fallback 2, Human 3`), 그리고 `es_safety::EventSet::bits()`. 같은 스텝을 두 번 측정한 것이 아니라
`events.json`이 받는 바로 그 기록이고, 그것이 에디터가 와이어로부터 끝난 실행의 행을 그대로
다시 세울 수 있는 이유다(`docs/design/editor-shell.ko.md` §13).

프로듀서가 지키는 규칙 둘과 와이어가 지킬 수 없는 규칙 하나:

- **텔레메트리를 위해 계산하는 것은 없다.** 평가기는 이미 가지고 있던 것을 클로저에 건네고,
  이미지 바이트는 다시 렌더하거나 복사하지 않고 플랜의 입력 버퍼에서 *빌린다*. `--telemetry`가
  없으면 아무것도 바인드하지 않고 §10.5의 두 산출물은 바이트 단위로 같다.
- **기다리는 것은 없다.** 발행은 §6의 논블로킹 팬아웃이므로, 리페인트에서 멈췄 에디터는 프레임을
  잃고 실행은 그것을 모른다. `CLIENT_QUEUE_CAPACITY`가 측정이 아니라 추측이라는 §6의 메모는
  그대로다 — M7/E4가 더한 것은 플로드사이드 비용의 측정(`editor-shell.ko.md` §13, 게이트 9)이지
  튜닝된 큐가 아니다.
- **토큰은 여전히 신뢰된 소켓 위의 평문이다**(§4). `--telemetry 127.0.0.1:7777`이 이것이 위하는
  모양이다; 토큰을 달고 `0.0.0.0`에 바인드한다고 안전한 원격 채널이 되지는 않으며,
  그것을 고치는 곳은 §6/§8의 TLS/QUIC 천장이다.

### 9.1 한 주소 위의 전체 루프 (패킷 M7/E7)

E4 뒤로는 `es eval run`만이 발행했다. 이제 세 프로듀서가 더 발행하고, 그중 하나는 *사이클*이다 —
전체 루프에서 시작을 누른 사람이 로그 줄이 아니라 실행 자체를 본다. 프레임을 만드는 자리는
`crates/es/src/cmd/telemetry.rs` 한 파일이며, 저장소에서 무언가를 `Frame`으로 바꾸는 곳은 거기뿐이다.

**모든 스트림 1 이벤트는 `stage` 필드를 지닌다**: `collect`, `expert-gate`, `train`, `eval`,
`showcase`, `cycle`. 소켓 하나로 충분한 이유가 그것이다. `es loop cycle --telemetry`는 첫 단계
전에 **한 번** 바인드하고 같은 퍼블리셔를 프로세스 안의 모든 단계에 넘긴다
(`cmd::r#loop::collect`, `cmd::eval::run`, `cmd::train::run`이 각각 `Option<&mut Publisher>`를
받는다). 독립 명령은 자기 것을 바인드하고 자기 이름을 붙인다. 사이클의 단계는
`stage.begin { name }`과 `stage.end { name, seconds, code }`로 감싸이므로, 에디터의 스트립은
계획이 가진 순서 그대로이고 각 단계가 쓴 벽시계 시간을 함께 지닌다.

주소는 **사이클의 것이지 단계의 것이 아니다**: 단계는 프로세스 안의 호출이므로
`es loop cycle --dry-run`이 찍는 계획의 어느 줄에도 `--telemetry`는 없고 `plan-cycle.txt`는
움직이지 않는다(`cycle_telemetry_is_one_address`).

| 스트림 | 페이로드 | 보내는 쪽 | 언제 |
|---|---|---|---|
| `1` | `Event { kind, fields }` | 전부 | `cell.begin` / `cell.end` / `suite.end`(평가, §9), `episode.begin { episode, seed }` / `episode.end { episode, outcome, steps }`(수집), `train.begin { total_steps }`와 `checkpoint { step, policy_hash }`(학습), `stage.begin` / `stage.end`(사이클). 모두 `stage`도 함께 지닌다 |
| `2` | `Scalars([frame, tick, source, violation bits])` | 평가, 수집 | 제어 틱마다 |
| `3` | `Metrics(PerfMetrics)` | 평가 | `cell.end`마다 |
| `4` | `Image { format: "rgb8" }` | 평가, 수집, 학습 | `--telemetry-image-every` 틱마다. 학습에서는 `--sample-every` 옵티마이저 스텝마다의 **샘플 이미지** |
| `5` | `Scalars([step, loss, lr, samples_per_s])` | 학습 | `--progress-every` 옵티마이저 스텝마다 |

스트림 5가 새로 생긴 유일한 id이고 그것도 여전히 스키마가 아니라 데이터다: `protocol.rs`는 자기
버전에 고정되어 있고, 5를 모르는 소비자는 그냥 무시한다.

**수집은 이미 가지고 있던 것을 발행한다.** `Collector::run_with_sink`는 클로저 하나
(`es_data::collect::CollectSink` — 여덟 번째 확장점이 아니라 클로저다, `INV-17`)를 에피소드
경계마다, 그리고 제어 틱마다 데이터셋의 `action_source` 열이 기록하는 바로 그 값과 그 스텝의
`EventSet` 비트와 함께 부른다. 비트는 그 스텝을 사이에 둔 **플레인 자신의 종류별 카운터 차이**다:
env가 하나이므로 스텝당 `validate`는 정확히 한 번이고, 수가 움직인 종류가 그 스텝이 올린 종류다 —
`SafeAction::events`가 지니는 것과 같은 비트셋을, 수집기가 볼 수 있는 쪽에서 읽은 것이다
(`SafeAction` 자체는 `es_env::DomainRunner::emit_actions`가 가지고 있다). 그 배열은 싱크가 있을
때만 읽히고, `--out` 아래 데이터셋은 플래그가 있든 없든 바이트 단위로 같다
(`collect_telemetry_publishes_every_episode`가 두 트리의 모든 파일을 비교한다).

**학습은 자기 stdout을 발행한다.** `train_act.py --progress-every N`은 `N` 옵티마이저 스텝마다
`{"progress": {…}}` 한 줄을 찍고, `--sample-every N`은 `--loss-curve` 옆에
`sample-<step>.bin` + `.json`을 쓰고 `{"sample": "<경로>"}`를 찍는다. 요약은 여전히 **마지막**
줄이고 바이트 단위로 같으며, `es train`은 그것을 같은 방법으로 읽는다 — `--telemetry`가 있으면
종료 시점이 아니라 `Stdio::piped` + `BufReader::lines`로 읽고, stderr는 파이프가 아니라 상속한다.
한 스레드로 파이프 둘을 읽으면 어느 한쪽이 차는 순간 교착하기 때문이다.

이 프로듀서가 지키는 규칙 셋:

- **트레이너의 두 플래그는 계획에 들어가지 않는다.** `config.json`이 계획을 지니고
  `identity_hash`가 `config.json`을 덮으므로(§19.3), 실행이 계산하는 것을 바꾸지 않는 플래그가
  실행의 정체성을 움직여서는 안 된다 — 누가 보고 있었느냐에 따라 달라지는 `training.lock`은
  똑같은 두 실행을 서로 다른 둘로 보이게 만든다. 두 플래그는 spawn 시점에 트레이너의 argv에
  덧붙고 자기 줄에 찍힌다. `train_telemetry_streams_the_curve`가 `training.lock`,
  `training/config.json`, `metrics/loss.json`, 포장된 `checkpoints/40.esb`를 플래그 없는 실행과
  바이트 단위로 비교한다.
- **스트림 3 프레임은 없다.** §12.4의 `PerfMetrics`에는 학습 자리가 없고 `protocol.rs`는 고정되어
  있다. 학습의 samples/s를 `actions_per_sec`에 실으면 어떤 양을 쟀는지에 대한 거짓말이 된다.
  처리량은 스트림 5의 네 번째 스칼라이고 다른 어디에도 없다.
- **학습 프레임의 `tick`은 0이다.** 옵티마이저 스텝은 물리 틱이 아니고, 스텝은 페이로드의 첫
  스칼라다.

**샘플 이미지는 보여 주기 위한 것이고 그렇다고 말한다.** 지금 묶음의 이미지 입력 하나를, 증강을
*거친 뒤* — 신경망이 실제로 맞추고 있는 텐서 — `clamp(v, 0, 1) * 255`로 `Rgb8`에 옮긴 것이다.
이 매핑은 데모의 체인(`Dequantize` ÷255 다음 `Normalize{Range 0..1}`, 즉 항등)의 정확한 역이다.
`contract.json`도 베이크의 `manifest.json`도 정규화의 수를 지니지 않으므로, 매핑은 가정되지 않고
**사이드카에 기록된다**: 평균/표준편차로 정규화하는 문서라면 `"mapping": "clamp(v, 0, 1) * 255"`가
옆에 붙은 채 색이 바랜 그림으로 드러나지, 아무도 의심할 수 없는 틀린 그림으로 드러나지 않는다.
임의의 체인을 되돌리려면 체인이 필요하고, 그것은 이 패킷이 하지 않는 매니페스트 변경이다.
