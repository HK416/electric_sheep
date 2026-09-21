# M7 R4 — 정지 카메라를 위한 시간 누적, 그리고 SVGF의 V

스펙: §15.3(포토리얼리스틱 데이터셋과 쇼케이스를 위한 PT), §28.6(SVGF가 이름 붙음),
§3.4(고정된 누적 순서, 데이터 의존적 루프 경계 없음, 카운터 기반 RNG), §28.10 규칙 1과 (R4).
확장할 설계 노트: `docs/design/renderer.md`(+ `.ko.md`) — 지금 참인 것을 말하도록 다시 쓰인
4.3절, 새 11절. **R3**(NEE, 톤맵 → `Rgb8`, `ssim`, PT 쇼케이스)와 **R1**(영속 버퍼)에
의존한다. 먼저 읽을 것: Schied 외 2017 "Spatiotemporal Variance-Guided Filtering"
4.1–4.3절(시간 누적, 모멘트, luminance 가중치, 짧은 히스토리를 위한 7×7 분산 사전 필터), 그리고
`docs/design/renderer.md` 4.3절의 건너뛴 것 목록 — 그 목록이 이 패킷의 체크리스트다.

## 질문

쇼케이스 카메라는 움직이지 않는다; 팔은 움직인다. 오늘은 매 프레임이 샘플 0개에서
시작한다. **PT가 카메라마다, 픽셀마다, 장면이 바뀌지 않은 픽셀에서는 spp를 대신하는
히스토리를 유지하고, 바뀐 곳에서는 정확히 그것을 버리며, 픽셀별 분산이 à-trous 필터를
이끌게 할 수 있는가 — 1 spp짜리 `N`프레임이 누적되어 `N` spp짜리 1프레임이 만드는
바이트가 되면서?**

## 사양

* **`RenderConfig.temporal: Option<Temporal { max_history: u32 }>`**, 기본값 `None`(오늘의
  바이트; 모든 골든과 `frames` 픽스처는 불변). `Pt` 경로에서만 읽힌다. 카메라 슬롯마다
  렌더러가 유지하는 것: radiance 합, 첫째와 둘째 luminance 모멘트, 히스토리 길이
  `n ≤ max_history`, 그리고 이전 프레임의 depth / normal / primitive id.
* **정지 카메라, 픽셀별 유효성.** 리프로젝션 없음: 카메라 슬롯의 `CameraView`가 비트
  단위로 이전 것과 같고 **그리고** 픽셀의 depth, normal, primitive id가 비트 단위로 이전
  프레임의 것과 같을 때 그 픽셀의 히스토리는 유효하다; 그렇지 않으면 그 픽셀에 대해
  `n := 0`. 바뀐 `CameraView`는 슬롯 전체를 리셋한다. 이것은 움직이지 않는 카메라 아래에서
  움직이는 장면을 위한 disocclusion 규칙이다; 움직이는 카메라는 *범위 밖*이며 노트가 그렇게
  말한다.
* **샘플 인덱싱이 누적을 정확하게 만든다.** 슬롯의 프레임 `f`는 `rng::key` 안의
  `sample = n · spp + s`로 `spp`개의 샘플을 뽑는다 — 그러므로 1 spp짜리 `N`프레임은
  `N` spp짜리 한 프레임과 같은 샘플을, 같은 순서로 뽑는다. 합은 커널 안 루프가 쓰는 것과
  같은 왼쪽에서 오른쪽 순서로 누적되는 **합**(누적 평균이 아니라)으로 유지되고, 출력에서
  한 번만 나뉜다 — 이것이 오라클 2를 허용 오차가 아니라 CPU에서의 *비트 단위* 진술로
  만드는 것이다.
* **분산.** 모멘트로부터: `var = max(0, E[l²] − E[l]²)`; `n < 4`에 대해서는 논문의 공간적
  7×7 bilateral 추정(depth/normal-가중)이 대신한다. à-trous 패스는
  `w_l = es_exp(−|l_p − l_q| / (σ_l · sqrt(var_p) + ε))`(`σ_l = 4`)를 얻어 기존
  `w_depth · w_normal`에 곱해진다; 분산은 논문이 하는 대로 제곱된 가중치로 색상과 나란히
  필터링된다. `sqrt`는 IEEE-정확하다; `es_exp`는 `approx`의 것이다. 히스토리 길이가
  이끄는 커널 확장은 여전히 건너뛴다(나열됨).
* **쇼케이스.** `es video showcase --path pt --accumulate [--max-history N]`(R3의 플래그에
  이것을 더한 것); 영속 히스토리는 쇼케이스가 틱들에 걸쳐 이미 유지하는 그 하나의
  `Renderer` 안에 산다.
* **관측 가능성.** `Channel::History`(픽셀당 `u32`, 이 프레임 이후의 `n`)가 PT 채널에
  합류한다 — 스키마가 아니라 데이터로서 — 그래서 테스트와 사람이 히스토리가 어디서
  버려졌는지 볼 수 있다.

## context

`cargo xtask check-scope`가 읽는 글롭, 그다음 같은 범위를 산문으로:

```
crates/es-render/src/view.rs
crates/es-render/src/cpu.rs
crates/es-render/src/renderer.rs
crates/es-render/src/atlas.rs
crates/es-render/src/lib.rs
crates/es-render/slang/common.slang
crates/es-render/slang/pt.slang
crates/es-render/slang/svgf.slang
crates/es-render/slang/accumulate.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_pt_accum8_rgb8.*
crates/es/src/cmd/showcase.rs
crates/es/tests/video.rs
crates/es-sensor/src/channel.rs
crates/es-env/src/render.rs
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M7/R4-temporal-accumulation.md
docs/packets/M7/R4-temporal-accumulation.ko.md
```

`view.rs`(`Temporal`, `Channel::History`), `cpu.rs`(레퍼런스: 누적, 모멘트, 분산, luminance
가중치), `renderer.rs`(슬롯별 히스토리 버퍼, 유효성 패스, dispatch 순서), `atlas.rs`는
**오직** `u32` 채널에 슬롯이 필요할 때만, `slang/{pt,svgf,accumulate}.slang`
(`accumulate.slang` 신규: 유효성 + 합 + 모멘트), `tests/render.rs`, 새 골든(8프레임 × 1
spp, NEE 켜짐, Reinhard — CPU 생성기의 바이트), `showcase.rs`/`video.rs`(플래그), 설계
노트, 이 패킷.

**패킷이 예상하지 못했던 두 파일, 구현 시점에 추가.** `Channel`은 `es-render`가 아니라
`es-sensor`(layer 3)의 타입이고 `es-render`는 재수출만 한다. 그래서 `Channel::History`는
`crates/es-sensor/src/channel.rs`의 match 4개이고, `crates/es-env/src/render.rs`의
`channel_format`에 있는 `Channel` 전수 `match`가 컴파일되려면 arm 하나가 더 필요하다.
둘 다 `EnvRendererCfg`에 PT 손잡이를 다는 일이 아니고 관측 경로에서 도달하지도 않는다.
`forbidden` 목록은 그대로다.

## 오라클

1. 기존의 모든 골든과 GPU 테스트가 불변; `cargo xtask verify-goldens` 변경 0건.
2. `cargo test -p es-render accumulation_of_n_frames_is_n_spp` — Cornell, CPU: 1 spp짜리
   8프레임을 누적한 것이 `PtRadiance`에서 8 spp짜리 1프레임과 **비트 단위로** 같다; 그리고
   4 spp짜리 8프레임은 32 spp짜리 1프레임과 같다.
3. `cargo test -p es-render history_drops_where_the_scene_moved` — 프레임 5에서 키 큰 블록이
   이동한 Cornell: depth/normal/id가 바뀌지 않은 모든 픽셀에서 `Channel::History`는 5이고
   바뀐 모든 픽셀에서는 1이며, 각 집합에 최소 한 픽셀이 있다; 바뀐 `CameraView`는 모든
   곳에서 1을 준다.
4. `cargo test -p es-render variance_falls_with_history` — `n = 1, 4, 16`에서의 픽셀별 평균
   `var`는 엄격히 감소한다(출력됨).
5. `cargo test -p es-render gpu_accumulation_matches_the_cpu` — 새 골든이 CPU에 의해 비트
   단위로 재현된다; GPU는 기존 PT 허용 오차 안에서, 출력됨; GPU `History` 채널은 비트
   단위.
6. `cargo test -p es-render luminance_weight_narrows_the_filter_where_variance_is_low` — 절반은
   수렴하고 절반은 노이즈가 있는 합성 타일에서, 필터링된 출력은 luminance 항이 없을 때보다
   수렴한 절반에서 덜 움직인다(절반마다의 RMSE 출력됨).
7. `cargo test -p es --test video showcase_accumulate_flags_are_parsed`; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R4-temporal-accumulation.md`.

## 수용 기준

오라클 1–7(GPU는 RTX 3060과 서버에서). 서버에서: V19b `nominal-00`의 PT 쇼케이스를
`--spp 4 --accumulate --max-history 32`로 R3의 `--spp 64` 옆에(둘 다 ms/frame; *정지*
픽셀과 *팔* 픽셀 각각에 대해 R3의 1,024 spp 레퍼런스에 대한 SSIM — History 채널이 마스크다),
`~/artifacts/plan-v/m7-r4/` 아래에; 프레임 0, 8, 32의 PNG를 worktree의
`target/plan-u/r4/`에. 11절이 샘플 인덱싱 규칙, 유효성 규칙, 수치, 그리고 남은
건너뜀(리프로젝션, 커널 확장, ReSTIR-GI)을 기록한다.

## 금지

어떤 골든이든, R3의 기본값이든 옮기는 것; 커널의 것 말고 다른 누적 평균이나 어떤 누적
순서든; 움직이는-카메라 리프로젝션; `approx` 밖의 초월함수; 데이터 의존적 루프 경계;
원자적 연산; `crates/es-gpu/**`; `EnvRendererCfg` 안의 PT 노브; `docs/ARCHITECTURE*.md`.
INV-17: 새 트레이트 없음.
