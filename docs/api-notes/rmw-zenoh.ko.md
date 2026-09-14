<!-- Korean translation of docs/api-notes/rmw-zenoh.md. The English file is the working copy; regenerate this when it changes. -->

# rmw_zenoh -- `es_ros2::names`, `es_ros2::attachment`, `es_ros2::session`를 위한 고정된 wire 표면

순수 Rust node가 zenoh 네트워크 위에 무엇을 올려야 `rmw_zenoh_cpp` node들과 `ros2` CLI가 그것을
발견하고 메시지를 주고받을 수 있는지(모드 A, §24.1). CDR payload와 타입 해시:
`docs/api-notes/ros2-cdr.md`. zenoh 라이브러리 API: `docs/api-notes/zenoh-rs.md`.

## 버전

| | |
|---|---|
| sources | `https://raw.githubusercontent.com/ros2/rmw_zenoh/<commit>/...`, 2026-09-14 조회: rolling `35d4509b11d4eefc3df7eca11a839aad1e0d012f`(rmw_zenoh_cpp 0.13.0), lyrical `4e17ccfb9e114c87ac87b802e705b60e283b4115`(0.10.6), kilted `ee5f35bbf71259fd77042a3cb0d523398d207c24`(0.6.8) |
| interop target | RoboStack `ros-kilted-rmw-zenoh-cpp 0.6.6`(`np2py312ha80d210_21`, `libzenohc >=1.7.2,<1.7.3` 링크)을 통한 ROS 2 **Kilted**. Lyrical(`ros-lyrical-rmw-zenoh-cpp 0.10.4`/`0.10.5`, prefix.dev에만 있음, `python_abi 3.14`, `libzenohc >=1.9.0,<1.9.1`)은 두 번째 타겟이며 CI에는 없음 |
| zenoh vendored upstream | `zenoh_cpp_vendor/CMakeLists.txt`(rolling, lyrical, kilted): `set(zenoh_c_commit 05bd370343b5161ca9269649b9a914c9c2dc4170)`, "corresponding to zenoh-c 1.8.0 plus few fixes"; zenoh-c `Cargo.lock`: `2687c51352121f006e3a603ce07925a8ad0b295c`의 `zenoh 1.8.0`. zenoh-cpp: rolling/lyrical `481b71bf...`(1.9.0), kilted `af381b42...`(1.8.0); serialization header는 둘 다 동일 |
| branch differences | rolling = lyrical design.md; kilted = jazzy. Rolling은 선택적 `/<backends>` 토큰 접미사를 추가한다, "only present for buffer-aware (i.e. `rosidl::Buffer`-carrying) publishers and subscribers and is omitted entirely for plain ROS message types". `es-ros2`는 그것을 절대 내보내지 않는다; 그것을 담은 remote 토큰은 parse되며, 여분의 chunk는 그대로 보존된다 |
| tier | REP-2000("up to Kilted Kaiju"에 적용됨): `rmw_zenoh_cpp | Eclipse Zenoh | Tier 1 | All Platforms | All Architectures`. Lyrical `supported-platforms.rst`: `rmw_zenoh_cpp | Eclipse Zenoh | Tier 1 | All Architectures`, `Zenoh | 1.8.0`, "The default middleware in ROS Lyrical is **rmw_fastrtps_cpp**." |

## Topic key expression

design.md: `<domain_id>/<fully_qualified_name>/<type_name>/<type_hash>`

```
0/chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18
0/robot1/chatter/std_msgs::msg::dds_::String_/RIHS01_...        (name /robot1/chatter)
```

- `topic_keyexpr_ = std::to_string(domain_id); += "/"; += strip_slashes(name_); ...`.
  `strip_slashes`는 앞뒤로 **하나씩**의 `/`만 제거한다; 안쪽의 `/`는 그대로 남는다(여기서는
  mangle되지 않음).
- 타입 이름(`type_support_common.cpp`): `ss << message_namespace << "::"; ss << "dds_::" << message_name << "_";`
  -> `std_msgs::msg::dds_::String_`.
- Hash: `RIHS01_PREFIX[] = "RIHS01_"`, `RIHS01_STRING_LEN = 71`, 소문자 hex
  (`rosidl_runtime_c/src/type_hash.c`).

## Liveliness 토큰 (`rmw_zenoh_cpp/src/detail/liveliness_utils.cpp`; kilted = rolling)

```
static const char ADMIN_SPACE[] = "@ros2_lv";
static const char NODE_STR[] = "NN";  PUB_STR[] = "MP"; SUB_STR[] = "MS"; SRV_STR[] = "SS"; CLI_STR[] = "SC";
static const char KEYEXPR_DELIMITER = '/';
static const char SLASH_REPLACEMENT = '%';
static const char QOS_DELIMITER = ':';
static const char QOS_COMPONENT_DELIMITER = ',';
```

Chunk: `AdminSpace / DomainId / Zid / Nid / Id / EntityStr / Enclave / Namespace / NodeName`, 그다음
`MP`/`MS`/`SS`/`SC`에 대해서는: `/ TopicName / TopicType / TopicTypeHash / TopicQoS`(rolling:
`[/Backends]`).

- `mangle_name`은 enclave, namespace, node 이름, topic 이름, 타입, hash 안의 `/`를 **모두** `%`로
  바꾼다. Namespace `/` -> `%`("An empty namespace from rcl will contain "/" but zenoh does not
  allow keys with "//""). Enclave: "just `%` if not set".
- `Zid` = `zid.to_string()`: zenoh-c -> uhlc 0.8.2 `write!(f, "{:x}", u128::from_le_bytes(self.0))`
  (소문자 hex, 앞자리 0 없음). zenoh-rs 1.8.0의 `ZenohId`의 `Display`도 같은 문자열을 낸다:
  **VERIFIED, W1b 2026-09-14** — `demo_talker_reaches_our_subscriber`가 실행한, 라이브
  `rmw_zenoh_cpp` talker 자신의 토큰(그 C++ 쪽 `Zid`로부터 만들어짐)을 `LivelinessToken::
  parse` / `to_key_expr`로 왕복시킨 결과가 바이트 단위로 동일했으며, 이는 두 `Display` 구현이
  일치할 때만 가능하다.
- `Nid`, `Id`: 십진수, 0부터 시작하는 context 전역 카운터 하나(`next_entity_id_(0)`,
  `fetch_add(1)`). node 토큰은 자신의 id를 반복한다: `.../<nid>/<nid>/NN/...`.
- Discovery: 각 node는 `@ros2_lv/<domain_id>/**`에 대해 `liveliness_get`을 수행한 다음, 같은
  key에 `history = true`로 liveliness subscriber를 선언한다.

원문 그대로(design.md):

```
@ros2_lv/0/aac3178e146ba6f1fc6e6a4085e77f21/0/0/NN/%/%/listener
@ros2_lv/0/8b20917502ee955ac4476e0266340d5c/0/10/MP/%/%/talker/%chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18/::,7:,:,:,,
```

### QoS chunk (`qos_to_keyexpr`)

`<reliability>:<durability>:<history>,<depth>:<deadline.sec>,<deadline.nsec>:<lifespan.sec>,<lifespan.nsec>:<liveliness>,<lease.sec>,<lease.nsec>`

- 구성 요소는 **rmw_zenoh 기본값과 같으면 비워진다**(`qos.cpp`): `RELIABLE`, `VOLATILE`,
  `KEEP_LAST`, depth `42`, deadline / lifespan / lease `RMW_DURATION_INFINITE`, liveliness
  `AUTOMATIC`. talker(depth 7, 나머지 기본값)는 `::,7:,:,:,,`를 낸다.
- 기본값이 아닌 값은 rmw enum의 `std::to_string`이다. `rmw/types.h`: reliability는
  `SYSTEM_DEFAULT, RELIABLE, BEST_EFFORT, UNKNOWN`으로 암묵적으로 선언됨(0..3), history와
  durability도 마찬가지; liveliness는 `AUTOMATIC = 1`, `MANUAL_BY_TOPIC = 3`. `BEST_AVAILABLE`의
  번호: **미검증**. `es-ros2`는 기본값과 명시적 depth만 내보내며, remote 구성 요소는 불투명한
  정수로 유지한다.

### GID

`simplified_XXH3_128bits(liveliness_keyexpr)`, `memcpy(gid, &low64, 8)`로 저장한 다음
`&high64`(x86_64: `low64.to_le_bytes() ++ high64.to_le_bytes()`). 표준 XXH3-128(seed 0)과의
동일성: **VERIFIED, W1b 2026-09-14** -- W1a의 golden은 PyPI `xxhash`(표준)를 사용한다;
`demo_talker_reaches_our_subscriber`는 추가로 라이브 `demo_nodes_cpp` talker의 attachment
`gid`를 이 crate가 계산한 `gid_of(<그 talker의 MP 토큰>)`과 비교하며, 실제 C++
`rmw_zenoh_cpp` publisher를 상대로(Python golden뿐 아니라) 일치한다.

## Attachment: 33바이트 (`attachment_helpers.cpp`; rolling = kilted = jazzy)

```
serializer.serialize(this->sequence_number_);   // int64_t
serializer.serialize(this->source_timestamp_);  // int64_t
serializer.serialize(this->source_gid_);        // std::array<uint8_t, RMW_GID_STORAGE_SIZE>, 16u
```

zenoh-cpp는 `std::array<T,N>`을 sequence로 직렬화한다(`ze_serializer_serialize_sequence_length`
-> `VarInt<usize>`, LEB128); zenoh-ext 1.8.0의 숫자는 `to_le_bytes()`다. 따라서:

| offset | size | 필드 |
|---|---|---|
| 0 | 8 | sequence number, i64 LE, publisher별, **1부터 시작**(`sequence_number_(1)`, post-increment) |
| 8 | 8 | source timestamp, i64 LE, UNIX epoch 이후 ns(`std::chrono::system_clock`) |
| 16 | 1 | `0x10`(LEB128 length = 16) |
| 17 | 16 | GID |

design.md의 "1 byte for the length of the publisher GID"가 바로 이 LEB128 접두사다. CHANGELOG
0.6.0(2025-04-18)에서 도입됨, "Change serialization format in attachment_helpers.cpp
(#601)"; 더 오래된 레이아웃은 지원되지 않는다.

## Payload와 데이터 경로

- Payload: encapsulation이 있는 CDR(`ser.serialize_encapsulation()`; Fast-CDR 2.2.x는
  little-endian 호스트에서 `00 01 00 00`을 쓴다).
- **zenoh `Encoding`은 설정되지 않는다**("all key expressions will be encoded with CDR so it
  does not really matter").
- attachment가 **없는** sample을 받으면 에러 로그와 함께 **버려진다**. 우리 publisher는 항상
  attach한다.
- Publisher는 zenoh-ext `AdvancedPublisher`다(`congestion_control = DROP`; RELIABLE +
  KEEP_ALL에는 `BLOCK`; TRANSIENT_LOCAL은 `publisher_detection`과 `max_samples = depth`
  캐시를 추가). Subscriber는 `AdvancedSubscriber`다(TRANSIENT_LOCAL:
  `detect_late_publishers`). Plain put / plain subscriber는 VOLATILE에 대해서는
  상호운용된다; TRANSIENT_LOCAL에 필요한 `@adv/...` 리터럴은 **미검증**이므로, `es-ros2`는
  TRANSIENT_LOCAL을 거부한다.

## Service (W1에는 구현되지 않음)

Server: `declare_queryable(service_key)`, `complete = true`; reply attachment은 request의
sequence number와 client GID를 새 timestamp와 함께 그대로 돌려준다(echo). Client:
`declare_querier`, `target = ALL_COMPLETE`, `consolidation = NONE`, request attachment
`(seq, ts, client_gid)`.

## 기본 설정 (rolling = lyrical = kilted)

| | session (`DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5`) | router (`DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5`) |
|---|---|---|
| `mode` | `"peer"` | `"router"` |
| `connect.endpoints` | `["tcp/localhost:7447"]` | 비어 있음 |
| `listen.endpoints` | `tcp/localhost:0`(design.md 표) | `tcp/[::]:7447`(design.md 표) |
| `scouting.multicast.enabled` | `false` | `false` |
| `scouting.gossip` | `enabled: true, multihop: false, target: { router: ["router", "peer"], peer: ["router"] }` | 동일 |
| `timestamping.enabled` | `{ router: true, peer: true, client: true }` | 동일 |
| `adminspace` | `enabled: true, permissions: { read: true, write: false }` | 동일 |

환경 변수(`zenoh_config.cpp`): `ZENOH_SESSION_CONFIG_URI`, `ZENOH_ROUTER_CONFIG_URI`,
`ZENOH_ROUTER_CHECK_ATTEMPTS`("The default is to check only once"; `0`은 무한정 대기, `<0`은
건너뜀; 1초마다 poll한 다음 "Proceeding with initialization"), `ZENOH_CONFIG_OVERRIDE`
(`'listen/endpoints=["tcp/127.0.0.1:7448"];scouting/multicast/enabled=true'`).

discovery에는 router가 필요하다(README): "Without the Zenoh router, nodes will not be able to
discover each other since multicast discovery is disabled by default" -- `ros2 run rmw_zenoh_cpp
rmw_zenohd`.

## 레퍼런스 명령 (라이브 오라클)

- rmw_zenoh README: `ros2 run rmw_zenoh_cpp rmw_zenohd`; `export RMW_IMPLEMENTATION=rmw_zenoh_cpp`;
  `ros2 run demo_nodes_cpp talker` / `listener`.
- ros2cli kilted `ros2topic/verb/echo.py`: `'--once', action='store_true', help='Print the first message received and then exit.'`;
  `pub.py`: `'-1', '--once', action='store_true', help='Publish one message and exit'`.

## zenoh-plugin-ros2dds (모드 B)

`eclipse-zenoh/zenoh-plugin-ros2dds` `main`, workspace `version = "1.10.1"`, 2026-09-14 조회.

- 모드 A와 상호운용되지 않는다(rmw_zenoh jazzy README, 328번째 줄): "`rmw_zenoh` utilizes Zenoh
  differently, particularly in terms of key expressions. Consequently, `rmw_zenoh` cannot
  interoperate with `zenoh-plugin-ros2dds`."
- Liveliness(`zenoh-plugin-ros2dds/src/liveliness_mgt.rs`):

```
pub ke_liveliness_all: "@/${zenoh_id:*}/@ros2_lv/${remaining:**}",
pub ke_liveliness_plugin: "@/${zenoh_id:*}/@ros2_lv",
pub(crate) ke_liveliness_pub: "@/${zenoh_id:*}/@ros2_lv/MP/${ke:*}/${typ:*}/${qos_ke:*}",
pub(crate) ke_liveliness_sub: "@/${zenoh_id:*}/@ros2_lv/MS/${ke:*}/${typ:*}/${qos_ke:*}",
pub(crate) ke_liveliness_service_srv: "@/${zenoh_id:*}/@ros2_lv/SS/${ke:*}/${typ:*}",
pub(crate) ke_liveliness_service_cli: "@/${zenoh_id:*}/@ros2_lv/SC/${ke:*}/${typ:*}",
pub(crate) ke_liveliness_action_srv: "@/${zenoh_id:*}/@ros2_lv/AS/${ke:*}/${typ:*}",
pub(crate) ke_liveliness_action_cli: "@/${zenoh_id:*}/@ros2_lv/AC/${ke:*}/${typ:*}",
const SLASH_REPLACEMSNT_CHAR: &str = "§";
```

  QoS chunk(`qos_to_key_expr`): `["K"] ":" [reliability] ":" [durability] ":" [history "," depth]`,
  Iron부터는 `":" user_data`가 추가된다.
- **모드 판별자:** rmw_zenoh 토큰은 `@ros2_lv/<domain>/`로 시작하고, bridge 토큰은
  `@/<zenoh_id>/@ros2_lv`로 시작한다. `@` chunk는 리터럴이라(wildcard가 절대 매치하지 않음)
  `@ros2_lv/**`는 bridge 토큰과 매치할 수 없고 `@/*/@ros2_lv/**`는 rmw_zenoh 토큰과 매치할
  수 없다.
- Topic 매핑(`src/ros2_utils.rs`, `ros2_name_to_key_expr`): namespace가 `"/"`면 ->
  `&ros2_name[1..]`; 그 외에는 `namespace[1..] / ros2_name[1..]`. `ros2_message_type_to_dds_type`:
  `std_msgs/msg/String` -> `std_msgs::msg::dds_::String_`.
- bridge 경로가 attachment를 갖는지: **미검증**(라이브 bridge 오라클은 계획되어 있지 않음).
- 배포: `robostack-kilted`의 linux-64에는 `ros-kilted-zenoh-bridge-ros2dds`가 없다(repodata,
  2026-09-14); README는 download.eclipse.org에서 받는 standalone `.zip`과 `cargo build
  --release`를 제공한다; "The bridge relies on CycloneDDS and has been tested with
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`"(`ros-kilted-rmw-cyclonedds-cpp 4.0.2`는 채널에
  있음).

## 미해결 항목과 이를 닫는 패킷

| Item | Closed by |
|---|---|
| zenoh-rs `ZenohId` 문자열 = zenoh-c의 것 | **W1b, 2026-09-14, VERIFIED**: `demo_talker_reaches_our_subscriber`가 실제 RoboStack Kilted `rmw_zenohd`를 상대로 라이브 talker의 `MP` 토큰을 왕복시킨다(`LivelinessToken::parse` / `to_key_expr` 바이트 단위로 동일) |
| GID = 표준 XXH3-128 | **W1b, 2026-09-14, VERIFIED**: 같은 테스트가 받은 attachment `gid`가 `gid_of(<talker의 MP 토큰>)`과 같다 |
| zenoh 1.8.0(우리) <-> 1.7.2(RoboStack router) wire 호환성 | **W1b, 2026-09-14, VERIFIED**: `rmw_zenoh_interop.rs`의 다섯 시나리오 모두 `ros-kilted-rmw-zenoh-cpp 0.6.6`(`libzenohc 1.7.2`)을 상대로 라이브로 통과 |
| 실제 rmw_zenoh CDR의 trailing padding / options bit | **W1b, 2026-09-14, VERIFIED (디코더 변경 불필요)**: `ros2_topic_pub_joint_state_reaches_our_subscriber`와 `capture_reference_goldens`가 실제 `ros2 topic pub` `JointState`(빈 `velocity`/`effort`)를 기존 `CdrReader`로 바이트 단위로 정확히 디코딩; `docs/api-notes/ros2-cdr.md`의 "Encapsulation header" 참고 |
| rmw_zenoh `CdrVersion` | 위와 동일 — 실제 네트워크 바이트가 plain XCDR1, `PLAIN_CDR`과 정확히 일치, W1a의 `rclpy` 근거가 예측한 그대로 |
| 모드 B attachment / 데이터 경로 | 계획되어 있지 않음(`Target / Status: unverified`) |
