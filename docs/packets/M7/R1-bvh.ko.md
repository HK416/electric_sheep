# M7 R1 — 80 ms 프레임: 캐시된 테셀레이션, 영속 버퍼, 소프트웨어 BVH

스펙: §15.3(RS가 비전의 기본값), §15.4(가속 구조 — TLAS 하나, 공유 BLAS; 이 패킷은
`es-gpu`의 컴퓨트 전용 디바이스가 허용하는 소프트웨어 버전이다), §3.4(결정성: 원자적 연산
없음, 고정된 순회 순서, 워크그룹 수 의존 없음), §12.4(`camera_frames_per_sec`,
`pixels_per_sec`; 단일 `step/s`는 절대 안 됨), §28.10 규칙 1(커밋된 문서의 픽셀은 움직이지
않는다). 확장할 설계 노트: `docs/design/renderer.md`(+ `.ko.md`) — 새 8절과, 플랫 스캔
한계를 언급하는 0, 2.1, 3절의 수정. 선행: M4 W8(렌더러), M5 V0b/V9(env 루프와 쇼케이스
호출부).

## 질문

`es video showcase`는 RTX 4090에서 삼각형 2,978개에 대해 1280×720에서 **프레임당 80 ms**를
측정했고(`visible-learning.md` 7.17절 항목 4), 평가 안의 모든 96×96 관측 프레임이 같은
프레임당 경로의 대가를 치른다. 알려진 두 원인이 있으나 정량화되지 않았다: 장면 전체가 매
프레임 다시 테셀레이트되고(f64, 정점마다 `approx::sin/cos`) **새로 할당된** 버퍼로 다시
업로드되며, 셰이더가 픽셀마다 삼각형을 전부 스캔한다 — `O(pixels × triangles)`. **시간은
어디로 가고, 각 원인을 출력 비트 하나 움직이지 않고 제거하면 얼마나 떨어지는가?**

## spec

**단계 0 — 먼저 측정한다.** 한 프레임의 네 단계 — 테셀레이트, 업로드, 디스패치+대기,
리드백 — 를 데모 장면에서 1280×720과 96×96 각각 100프레임에 걸쳐 중앙값과 p95로 재는
`#[ignore]`가 붙은 테스트(또는 `es video showcase`의 `--profile` 플래그, 둘 중 더 작은
쪽)를 추가한다. 아무것도 바꾸기 **전에** 그 표를 설계 노트에 기록한다. 이것이 수용 기준이
비교할 기준선이다.

**단계 1 — 캐시된 테셀레이션.** `TriScene::from_scene_with_poses`는 호출마다 모든 geom의
로컬 정점을 다시 계산한다. geom별 로컬 테셀레이션(`tessellate(geom)`이 반환하는
`Vec<[Vec3; 3]>`)을 새 `SceneCache`(또는 `Renderer` 위)에 geom id로 키를 매겨 캐시하고,
자세는 오늘과 정확히 같은 방식으로 프레임마다 적용한다 — f64로 같은
`pose.transform_point`, 같은 `to_f32`, 같은 `face_normal`, 같은 순서 — 그래서 월드 공간
삼각형이 캐시하지 않은 경로와 **비트 단위로 같다**. 오라클이 그것을 검사한다.

**단계 2 — 영속 버퍼.** `Renderer::upload_tris`와 `render`는 호출마다 새 `Buffer`를
할당한다(`Buffer::new`는 Vulkan 할당이다). 삼각형, 파라미터, 출력, BVH 버퍼를 프레임에
걸쳐 유지하고 프레임이 더 필요로 할 때만 키운다(절대 줄이지 않는다). 리드백은 그대로 둔다.

**단계 3 — 단일 레벨 BVH, 프레임마다 CPU에서 재구축.** 프레임의 월드 공간 삼각형들에 대해:
이진 BVH, 최장축을 따른 centroid 중앙값 분할, 리프는 ≤ 4 삼각형, 노드는 평평한 배열
(`[min.xyz, max.xyz, left/first, count]`, f32/u32)에, 결정적 루틴(안정적 순서, `HashMap`
없음, 스레드 없음)으로 짓는다. 삼각형 옆에 업로드한다. `common.slang`(`es_nearest`, 그리고
`restir.slang`이 쓰는 any-hit/그림자 스캔)과 `es_render::cpu::nearest_hit`에서, 플랫 스캔을
스택 기반 순회로 바꾼다(고정 64엔트리 스택; 빌드가 깊이를 제한하고 테스트가 그것을
단언한다). **히트 규칙은 변하지 않는다**: 엄격한 `<`로 가장 가까운 `t`, 동점은 더 낮은
*전역 삼각형 인덱스*로 깨진다 — 그래서 순회는 고정된 순서로 노드를 방문하고 `(t, index)`를
비교하며, 승자는 플랫 스캔의 승자와 같다. 레이-삼각형 교차 코드는 손대지 않는다(같은
`Möller–Trumbore`, 같은 식 순서) — 이것이 깊이와 노멀을 0 ULP로 유지하는 것이다.

2레벨 TLAS/BLAS(geom마다 한 번 지어지는 BLAS, 프레임마다 바디 인스턴스 위의 TLAS)는
§15.4의 설계이자 업그레이드 경로다; 이 패킷의 것은 **아니다** — 프레임마다의 f64 CPU
자세 잡기가 정점을 오늘과 비트 단위로 같게 유지하는 것이고, 삼각형 3,000개짜리 재구축은
80 ms가 가는 곳이 아니기 때문이다. 빌드 지점에 그 업그레이드를 지목하는 `ponytail:` 주석을
남긴다.

**단계 4 — 다시 측정**한다. 같은 표를, 기준선 옆에 기록한다. 두 해상도에 대해 §12.4의
`camera_frames_per_sec`과 `pixels_per_sec`을 보고한다; 1280×720에서의 목표
`< 5 ms/frame`은 서버 수치가 나올 때까지 `Target / Status: unverified`로 남는다.

## context (허용 범위)

`crates/es-render/src/{scene.rs,renderer.rs,cpu.rs,lib.rs}`, `crates/es-render/src/bvh.rs`
(신규), `crates/es-render/slang/{common.slang,restir.slang,pt.slang,raster.slang}`(순회만),
`crates/es-render/tests/render.rs`(새 테스트; 기존 테스트와 골든 생성기는 그대로),
`crates/es-render/benches/` 또는 `#[ignore]`가 붙은 타이밍 테스트,
`crates/es-env/src/render.rs`(캐시를 쓴다; 동작 변화 없음), `crates/es/src/cmd/showcase.rs`
(캐시를 쓴다; 프레임당 루프는 그 외에는 불변), `docs/design/renderer*.md`,
`docs/packets/M7/R1-bvh*.md`.

## oracle

1. `cargo test -p es-render cpu_reference_reproduces_the_goldens_bit_for_bit`와 기존의 모든
   GPU 테스트 — **파일도 결과도 변하지 않는다**. `cargo xtask verify-goldens`가 변경 0건을
   보고한다. 이것이 이 패킷의 처음이자 마지막 오라클이다.
2. `cargo test -p es-render cached_tessellation_is_bit_identical` — 캐시를 거친
   `from_scene_with_poses`와 캐시하지 않은 경로가 Cornell과 SO-101 장면 각각에 대해,
   픽스처 `.estraj`의 기록된 세 틱에서 같은 `TriScene`을 낸다(`PartialEq`, `f32`에 대해
   비트 단위 — E2의 픽스처 궤적이 착륙해 있으면 그것을 재사용하고, 아니면 장면의 홈 자세와
   테스트가 직접 지은 자세로 손으로 설정한 관절 벡터 둘을 `TriScene::from_scene_with_poses`에
   통과시킨다).
3. `cargo test -p es-render bvh_traversal_is_the_flat_scan` — 두 장면 각각에 대해 10,000개의
   레이(`es_render::rng`가 주는 결정적 카운터 기반 방향)로, BVH를 거친 `nearest_hit`이
   남겨둔 플랫 스캔(옛 스캔을 `nearest_hit_flat`로 테스트 전용 또는 `pub(crate)`로 남긴다)과
   같은 `Option<Hit>`(비트 단위 `t`, 같은 삼각형 인덱스)을 반환한다; any-hit 변형은
   불리언에서 일치한다; 모든 레이에 걸친 최대 순회 깊이는 ≤ 64이고 출력된다.
4. `cargo test -p es-render gpu_bvh_matches_the_cpu_bvh`(GPU; 디바이스 없으면 `SKIP`) —
   GPU를 거친 SO-101 장면 256×256에서의 `Rs`와 `Pt` 1 spp가 기존 허용 오차에서 CPU
   레퍼런스와 같다(`Rgb8`/`Seg`는 비트 단위, `Depth`/`Normal`은 ≤ 1 ULP, 출력됨), 그리고
   `gpu_renders_are_bit_identical_across_runs`가 BVH를 넣고도 여전히 성립한다.
5. `cargo test -p es-render --release -- --ignored frame_profile`가 두 해상도에 대해
   네 단계 표를 출력한다; 전후로 실행해 두 표 모두 설계 노트에 담는다.
6. `cargo xtask ci` 통과; `cargo xtask check-scope docs/packets/M7/R1-bvh.md` 깨끗함.

## acceptance

오라클 1–6이 로컬(RTX 3060)에서 통과하고 4–5는 오라클 서버(RTX 4090, 헤드리스)에서도
통과한다. `~/artifacts/plan-v/v19b`의 `nominal-00`에 대한 1280×720의 `es video showcase`가
V9의 비트 동일성 오라클(`showcase_replay_of_a_real_run_is_bit_identical`, 96×96,
`--ignored`)을 재현하고, 그 ms/frame이 전후로 기록된다. 설계 노트 8절이 두 프로파일 표,
BVH 레이아웃, 비트 동일성에 대한 순회 순서 논증, 2레벨 업그레이드 경로를 담는다.

## forbidden

골든이나 픽스처를 바꾸는 것; `Möller–Trumbore`나 어떤 셰이딩 식을 바꾸는 것; 테셀레이션
개수 변경; 원자적 연산, 공유 메모리, 서브그룹 연산, 워크그룹 수에 대한 어떤 의존도(§3.4);
Vulkan 확장(`es-gpu`는 컴퓨트 전용이고 계속 그렇다; `crates/es-gpu/**`는 범위 밖);
`HashMap`; 빌드 안의 스레드; `docs/ARCHITECTURE*.md`; `RenderConfig`의 기본값(룩은 R2의
것). INV-17: 새 트레이트 없음.
