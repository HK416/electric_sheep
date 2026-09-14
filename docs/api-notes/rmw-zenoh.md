# rmw_zenoh -- pinned wire surface for `es_ros2::names`, `es_ros2::attachment`, `es_ros2::session`

What a pure-Rust node must put on a zenoh network so that `rmw_zenoh_cpp` nodes and the `ros2`
CLI discover it and exchange messages with it (mode A, §24.1). CDR payloads and type hashes:
`docs/api-notes/ros2-cdr.md`. zenoh library API: `docs/api-notes/zenoh-rs.md`.

## Version

| | |
|---|---|
| sources | `https://raw.githubusercontent.com/ros2/rmw_zenoh/<commit>/...`, retrieved 2026-09-14: rolling `35d4509b11d4eefc3df7eca11a839aad1e0d012f` (rmw_zenoh_cpp 0.13.0), lyrical `4e17ccfb9e114c87ac87b802e705b60e283b4115` (0.10.6), kilted `ee5f35bbf71259fd77042a3cb0d523398d207c24` (0.6.8) |
| interop target | ROS 2 **Kilted** via RoboStack `ros-kilted-rmw-zenoh-cpp 0.6.6` (`np2py312ha80d210_21`, links `libzenohc >=1.7.2,<1.7.3`). Lyrical (`ros-lyrical-rmw-zenoh-cpp 0.10.4`/`0.10.5`, prefix.dev only, `python_abi 3.14`, `libzenohc >=1.9.0,<1.9.1`) is a second target, not in CI |
| zenoh vendored upstream | `zenoh_cpp_vendor/CMakeLists.txt` (rolling, lyrical, kilted): `set(zenoh_c_commit 05bd370343b5161ca9269649b9a914c9c2dc4170)`, "corresponding to zenoh-c 1.8.0 plus few fixes"; zenoh-c `Cargo.lock`: `zenoh 1.8.0` at `2687c51352121f006e3a603ce07925a8ad0b295c`. zenoh-cpp: rolling/lyrical `481b71bf...` (1.9.0), kilted `af381b42...` (1.8.0); serialization header identical at both |
| branch differences | rolling = lyrical design.md; kilted = jazzy. Rolling adds an optional `/<backends>` token suffix, "only present for buffer-aware (i.e. `rosidl::Buffer`-carrying) publishers and subscribers and is omitted entirely for plain ROS message types". `es-ros2` never emits it; a remote token carrying it parses, with the extra chunk kept verbatim |
| tier | REP-2000 (applies "up to Kilted Kaiju"): `rmw_zenoh_cpp | Eclipse Zenoh | Tier 1 | All Platforms | All Architectures`. Lyrical `supported-platforms.rst`: `rmw_zenoh_cpp | Eclipse Zenoh | Tier 1 | All Architectures`, `Zenoh | 1.8.0`, "The default middleware in ROS Lyrical is **rmw_fastrtps_cpp**." |

## Topic key expression

design.md: `<domain_id>/<fully_qualified_name>/<type_name>/<type_hash>`

```
0/chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18
0/robot1/chatter/std_msgs::msg::dds_::String_/RIHS01_...        (name /robot1/chatter)
```

- `topic_keyexpr_ = std::to_string(domain_id); += "/"; += strip_slashes(name_); ...`.
  `strip_slashes` removes **one** leading and one trailing `/`; inner `/` stay (not mangled here).
- Type name (`type_support_common.cpp`): `ss << message_namespace << "::"; ss << "dds_::" << message_name << "_";`
  -> `std_msgs::msg::dds_::String_`.
- Hash: `RIHS01_PREFIX[] = "RIHS01_"`, `RIHS01_STRING_LEN = 71`, lowercase hex
  (`rosidl_runtime_c/src/type_hash.c`).

## Liveliness tokens (`rmw_zenoh_cpp/src/detail/liveliness_utils.cpp`; kilted = rolling)

```
static const char ADMIN_SPACE[] = "@ros2_lv";
static const char NODE_STR[] = "NN";  PUB_STR[] = "MP"; SUB_STR[] = "MS"; SRV_STR[] = "SS"; CLI_STR[] = "SC";
static const char KEYEXPR_DELIMITER = '/';
static const char SLASH_REPLACEMENT = '%';
static const char QOS_DELIMITER = ':';
static const char QOS_COMPONENT_DELIMITER = ',';
```

Chunks: `AdminSpace / DomainId / Zid / Nid / Id / EntityStr / Enclave / Namespace / NodeName`, then
for `MP`/`MS`/`SS`/`SC`: `/ TopicName / TopicType / TopicTypeHash / TopicQoS` (rolling: `[/Backends]`).

- `mangle_name` replaces **every** `/` with `%` in enclave, namespace, node name, topic name, type
  and hash. Namespace `/` -> `%` ("An empty namespace from rcl will contain "/" but zenoh does not
  allow keys with "//""). Enclave: "just `%` if not set".
- `Zid` = `zid.to_string()`: zenoh-c -> uhlc 0.8.2 `write!(f, "{:x}", u128::from_le_bytes(self.0))`
  (lowercase hex, no leading zeros). Same string from zenoh-rs 1.8.0 `ZenohId`'s `Display`:
  **unverified** (W1b live oracle).
- `Nid`, `Id`: decimal, one context-wide counter from 0 (`next_entity_id_(0)`, `fetch_add(1)`). A
  node token repeats its id: `.../<nid>/<nid>/NN/...`.
- Discovery: each node does `liveliness_get` on `@ros2_lv/<domain_id>/**`, then declares a
  liveliness subscriber on the same key with `history = true`.

Verbatim (design.md):

```
@ros2_lv/0/aac3178e146ba6f1fc6e6a4085e77f21/0/0/NN/%/%/listener
@ros2_lv/0/8b20917502ee955ac4476e0266340d5c/0/10/MP/%/%/talker/%chatter/std_msgs::msg::dds_::String_/RIHS01_df668c740482bbd48fb39d76a70dfd4bd59db1288021743503259e948f6b1a18/::,7:,:,:,,
```

### QoS chunk (`qos_to_keyexpr`)

`<reliability>:<durability>:<history>,<depth>:<deadline.sec>,<deadline.nsec>:<lifespan.sec>,<lifespan.nsec>:<liveliness>,<lease.sec>,<lease.nsec>`

- A component is **empty when it equals the rmw_zenoh default** (`qos.cpp`): `RELIABLE`, `VOLATILE`,
  `KEEP_LAST`, depth `42`, deadline / lifespan / lease `RMW_DURATION_INFINITE`, liveliness
  `AUTOMATIC`. The talker (depth 7, rest default) gives `::,7:,:,:,,`.
- Non-default values are `std::to_string` of the rmw enum. `rmw/types.h`: reliability
  `SYSTEM_DEFAULT, RELIABLE, BEST_EFFORT, UNKNOWN` declared implicitly (0..3), history and
  durability likewise; liveliness `AUTOMATIC = 1`, `MANUAL_BY_TOPIC = 3`. The number of
  `BEST_AVAILABLE`: **unverified**. `es-ros2` emits defaults plus an explicit depth only and keeps
  remote components as opaque integers.

### GID

`simplified_XXH3_128bits(liveliness_keyexpr)`, stored `memcpy(gid, &low64, 8)` then `&high64`
(x86_64: `low64.to_le_bytes() ++ high64.to_le_bytes()`). Equality with stock XXH3-128 (seed 0):
**unverified** -- W1a's golden uses PyPI `xxhash` (stock), W1b compares against a live talker's GID.

## Attachment: 33 bytes (`attachment_helpers.cpp`; rolling = kilted = jazzy)

```
serializer.serialize(this->sequence_number_);   // int64_t
serializer.serialize(this->source_timestamp_);  // int64_t
serializer.serialize(this->source_gid_);        // std::array<uint8_t, RMW_GID_STORAGE_SIZE>, 16u
```

zenoh-cpp serializes `std::array<T,N>` as a sequence (`ze_serializer_serialize_sequence_length` ->
`VarInt<usize>`, LEB128); zenoh-ext 1.8.0 numbers are `to_le_bytes()`. Hence:

| offset | size | field |
|---|---|---|
| 0 | 8 | sequence number, i64 LE, per publisher, **starts at 1** (`sequence_number_(1)`, post-increment) |
| 8 | 8 | source timestamp, i64 LE, ns since UNIX epoch (`std::chrono::system_clock`) |
| 16 | 1 | `0x10` (LEB128 length = 16) |
| 17 | 16 | GID |

design.md's "1 byte for the length of the publisher GID" is this LEB128 prefix. Introduced by
CHANGELOG 0.6.0 (2025-04-18), "Change serialization format in attachment_helpers.cpp (#601)";
older layouts are not supported.

## Payload and data path

- Payload: CDR with encapsulation (`ser.serialize_encapsulation()`; Fast-CDR 2.2.x writes
  `00 01 00 00` on little-endian hosts).
- **No zenoh `Encoding` is set** ("all key expressions will be encoded with CDR so it does not really matter").
- A received sample **without an attachment is dropped** with an error log. Our publishers always attach.
- Publishers are zenoh-ext `AdvancedPublisher`s (`congestion_control = DROP`; `BLOCK` for
  RELIABLE + KEEP_ALL; TRANSIENT_LOCAL adds `publisher_detection` and a `max_samples = depth`
  cache). Subscribers are `AdvancedSubscriber`s (TRANSIENT_LOCAL: `detect_late_publishers`).
  Plain put / plain subscriber interoperate for VOLATILE; the `@adv/...` literals needed for
  TRANSIENT_LOCAL are **unverified**, so `es-ros2` rejects TRANSIENT_LOCAL.

## Services (not implemented in W1)

Server: `declare_queryable(service_key)`, `complete = true`; reply attachment echoes the request's
sequence number and client GID with a fresh timestamp. Client: `declare_querier`,
`target = ALL_COMPLETE`, `consolidation = NONE`, request attachment `(seq, ts, client_gid)`.

## Default configuration (rolling = lyrical = kilted)

| | session (`DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5`) | router (`DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5`) |
|---|---|---|
| `mode` | `"peer"` | `"router"` |
| `connect.endpoints` | `["tcp/localhost:7447"]` | empty |
| `listen.endpoints` | `tcp/localhost:0` (design.md table) | `tcp/[::]:7447` (design.md table) |
| `scouting.multicast.enabled` | `false` | `false` |
| `scouting.gossip` | `enabled: true, multihop: false, target: { router: ["router", "peer"], peer: ["router"] }` | same |
| `timestamping.enabled` | `{ router: true, peer: true, client: true }` | same |
| `adminspace` | `enabled: true, permissions: { read: true, write: false }` | same |

Environment (`zenoh_config.cpp`): `ZENOH_SESSION_CONFIG_URI`, `ZENOH_ROUTER_CONFIG_URI`,
`ZENOH_ROUTER_CHECK_ATTEMPTS` ("The default is to check only once"; `0` waits forever, `<0` skips;
polls every 1 s, then "Proceeding with initialization"), `ZENOH_CONFIG_OVERRIDE`
(`'listen/endpoints=["tcp/127.0.0.1:7448"];scouting/multicast/enabled=true'`).

A router is needed for discovery (README): "Without the Zenoh router, nodes will not be able to
discover each other since multicast discovery is disabled by default" -- `ros2 run rmw_zenoh_cpp rmw_zenohd`.

## Reference commands (live oracle)

- rmw_zenoh README: `ros2 run rmw_zenoh_cpp rmw_zenohd`; `export RMW_IMPLEMENTATION=rmw_zenoh_cpp`;
  `ros2 run demo_nodes_cpp talker` / `listener`.
- ros2cli kilted `ros2topic/verb/echo.py`: `'--once', action='store_true', help='Print the first message received and then exit.'`;
  `pub.py`: `'-1', '--once', action='store_true', help='Publish one message and exit'`.

## zenoh-plugin-ros2dds (mode B)

`eclipse-zenoh/zenoh-plugin-ros2dds` `main`, workspace `version = "1.10.1"`, retrieved 2026-09-14.

- Not interoperable with mode A (rmw_zenoh jazzy README, line 328): "`rmw_zenoh` utilizes Zenoh
  differently, particularly in terms of key expressions. Consequently, `rmw_zenoh` cannot
  interoperate with `zenoh-plugin-ros2dds`."
- Liveliness (`zenoh-plugin-ros2dds/src/liveliness_mgt.rs`):

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

  QoS chunk (`qos_to_key_expr`): `["K"] ":" [reliability] ":" [durability] ":" [history "," depth]`
  plus `":" user_data` from Iron on.
- **Mode discriminator:** rmw_zenoh tokens begin `@ros2_lv/<domain>/`, bridge tokens begin
  `@/<zenoh_id>/@ros2_lv`. `@` chunks are verbatim (wildcards never match them), so
  `@ros2_lv/**` cannot match a bridge token and `@/*/@ros2_lv/**` cannot match an rmw_zenoh token.
- Topic mapping (`src/ros2_utils.rs`, `ros2_name_to_key_expr`): namespace `"/"` ->
  `&ros2_name[1..]`; otherwise `namespace[1..] / ros2_name[1..]`. `ros2_message_type_to_dds_type`:
  `std_msgs/msg/String` -> `std_msgs::msg::dds_::String_`.
- Whether bridge routes carry an attachment: **unverified** (no live bridge oracle scheduled).
- Distribution: no `ros-kilted-zenoh-bridge-ros2dds` in `robostack-kilted` linux-64 (repodata,
  2026-09-14); the README offers standalone `.zip` downloads from download.eclipse.org and
  `cargo build --release`; "The bridge relies on CycloneDDS and has been tested with
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`" (`ros-kilted-rmw-cyclonedds-cpp 4.0.2` is in the channel).

## Open items and the packet that closes each

| Item | Closed by |
|---|---|
| zenoh-rs `ZenohId` string = zenoh-c's | W1b `demo_talker_reaches_our_subscriber` (token round trip) |
| GID = stock XXH3-128 | W1b, talker attachment GID vs `gid_of(talker MP token)` |
| zenoh 1.8.0 (us) <-> 1.7.2 (RoboStack router) wire compatibility | W1b live oracle |
| trailing padding / options bits in real rmw_zenoh CDR | W1a `rclpy` cross-check, W1b capture |
| rmw_zenoh `CdrVersion` | same |
| mode B attachment / data path | not scheduled (`Target / Status: unverified`) |
