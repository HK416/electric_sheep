# M7 R3 — 성장하는 패스 트레이서: next-event 추정, MIS, 비편향 ReSTIR, 톤맵

스펙: §15.3(PT: 포토리얼리스틱 데이터셋과 골든; RS와 PT 사이의 RGB는 SSIM
임계값 — 결코 측정된 적 없음), §15.1(같은 채널 계약), §28.6(ReSTIR/SVGF가 이름 붙음),
§3.2/§3.4(`approx` 초월함수, 고정된 누적 순서, 카운터 기반 RNG), §1.9 항목 2(PT는
잘라낼 수 있다 — 그래서 여기의 모든 단계는 자신이 무엇을 건너뛰는지 정직하다),
§28.10 규칙 1. 확장할 설계 노트: `docs/design/renderer.md`(+ `.ko.md`) 4, 4.2, 6절과 새
10절. **R1**(순회)과 **R2**(`Shading::Full`이 SSIM 비교의 RS 쪽이다; 톤맵을 공유한다)에
의존한다. 먼저 읽을 참고문헌: Pharr, Jakob, Humphreys, *Physically Based Rendering* 4판
13.10장(광원 샘플링, balance/power heuristic을 쓴 MIS); Bitterli 외 2020 "Spatiotemporal
reservoir resampling for real-time ray tracing with dynamic direct lighting" 4.3절(비편향
`1/Z` 조합)과 Wyman 외 2023 "A Gentle Introduction to ReSTIR" 5절(pairwise MIS 가중치);
Reinhard 외 2002(`x/(1+x)` 연산자).

## the question

오늘의 PT: 모두 디퓨즈, 방출하는 삼각형만이 유일한 광원, 바운스가 우연히 광원에 닿을 때만
`Le`를 누적하는 히트(next-event 추정 없음), `ReSTIR`은 편향된 `1/M`으로 리저버를 결합하고,
출력은 톤맵 없는 선형 radiance라서 PT는 `Rgb8`을 만들 수 없고 쇼케이스는 결코 그것을 쓰지
않는다(설계 노트 7.17절 항목 5). **PT가 사람이 포토리얼이라 부를 `Rgb8` 프레임을 만들고,
쇼케이스에 쓰기에 충분히 빠르게 수렴하며, §15.3의 SSIM 계약을 처음으로 측정 가능하게 만들
수 있는가?**

## spec

* **광원.** 세 종류, 모두 CPU 위의 하나의 `Light` 열거형과 GPU의 파라미터/장면 버퍼 안의
  작은 광원 테이블을 통해: 방출하는 삼각형(오늘처럼), 방향광 `light_dir`(`RenderConfig`가
  이미 갖고 있다 — radiance `light_rgb`를 부여하고, PT의 기본 출력이 불변이도록 기본값은
  `[0,0,0]`), 그리고 하늘(`sky`는 이미 존재한다: 미스는 그것을 반환한다 — 그대로 유지;
  `Full` PT에서는 코사인 가중 반구 pdf로 광원으로도 샘플링된다).
* **NEE + MIS.** 각 디퓨즈 히트에서: 광원 하나를 샘플링하고(방출하는 삼각형에 대해서는
  광원 목록 위에서 균등하게, 그다음 삼각형의 면적 위에서 균등하게; 방향광은 델타이므로 —
  MIS가 필요 없고 그림자 레이 하나만), R1의 any-hit으로 그림자 레이를 추적하고,
  `f * Le * G / p_light * w_light`를 더한다; BSDF 바운스는 오늘처럼 계속하고, 방출하는
  삼각형에 닿으면 `Le * w_bsdf`를 더하는데, 여기서 `w_light`/`w_bsdf`는 두 전략의 pdf에
  대한 **power heuristic**(β = 2)이다. `Pt { nee: bool }` 뒤에 있으며 기본값은 `false`라
  `cornell_pt1spp`는 손대지 않는다. 고정된 샘플 순서; RNG 스트림은 `key(…, bounce,
  stream)`이고 광원 선택, 면적 샘플, 그림자 테스트에 대한 새 스트림 id를 갖는다 — 스트림
  표를 문서화할 것.
* **비편향 ReSTIR.** `1/M`을 비편향 기여 가중치 `W = (1/p̂_y) · (w_sum / Z)`로
  교체한다(`Z`는 목적지에서 목표 함수가 0이 아닌 재사용된 리저버를 센다), 그리고
  공간 패스에서 pairwise MIS를 써서("Gentle Introduction" 형태) 샘플을 생성할 수
  없었던 이웃이 가중치 0을 기여하게 한다. 이전처럼 `Pt { restir: true }` 뒤에 있으며 —
  현재의 편향된 경로는 유지되는 게 아니라 *교체된다*: 그것은 이 패킷 자신의
  `unverified` TODO였고, ReSTIR을 고정하는 테스트
  (`gpu_restir_and_svgf_match_the_cpu_within_tolerance`)는 골든이 아니라 허용 오차를
  가지므로 이 변경은 합법이다. Cornell에서 4,096 spp 레퍼런스에 대한 전후 평균-radiance
  편향을 기록한다(수치가 있는 관측; `unverified` → 측정됨).
* **톤맵 → `Rgb8`.** `Channel::Rgb8`이 `PT_CHANNELS`에 합류한다. `RenderConfig`는
  `exposure: f32`(기본값 1.0)와 `tonemap: Tonemap { Reinhard, Aces }`(기본값
  `Reinhard`)를 얻는다: `Reinhard(c) = c·e / (1 + c·e)`를 채널마다; `Aces`는 Narkowicz
  2015의 유리함수 근사 `(x(2.51x + 0.03)) / (x(2.43x + 0.59) + 0.14)`를 클램프한 것 —
  둘 다 채널당 덧셈, 곱셈, 나눗셈 한 번이고 초월함수가 없으므로 CPU == GPU **비트
  단위**가 오라클이다; 그다음 기존의 정확한 sRGB 변환과 반올림. 같은 두 함수는 R2의
  `Full` 셰이딩에서도 쓸 수 있다(오늘은 클램프한다) — R2의 기본값은 바꾸지 말 것.
* **SSIM.** `es_render::ssim(a: &[u8], b: &[u8], w, h) -> f64` — 두 `Rgb8` 타일의 luma에
  대한 표준 8×8 윈도 SSIM(Wang 외 2004 상수 `K1 = 0.01, K2 = 0.03`), 순수 Rust, f64,
  결정적. Cornell과 V9 쇼케이스 카메라에서의 SO-101 장면에서, NEE를 켠 1,024 spp의
  `Pt`와 `Rs` `Full`(R2 프리셋, `ssaa 2`) 사이에서 측정한다; 수치는 설계 노트에 실린다;
  §15.3의 임계값은 **그 뒤에 그 수치로부터 정해진다**, 여기서 단언되지 않는다(그때까지는
  `Target / Status: unverified` — 이 패킷은 그 수치를 존재하게 만들 뿐이다).
* `es video showcase --path pt --spp N [--bounces B] [--exposure E] [--tonemap
  reinhard|aces]`가 PT로부터 `Rgb8` 프레임을 쓴다(NEE 켜짐, ReSTIR은 기본값으로 꺼짐,
  SVGF는 선택).

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
crates/es-render/src/view.rs
crates/es-render/src/cpu.rs
crates/es-render/src/renderer.rs
crates/es-render/src/lib.rs
crates/es-render/src/rng.rs
crates/es-render/src/ssim.rs
crates/es-render/slang/common.slang
crates/es-render/slang/pt.slang
crates/es-render/slang/restir.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_pt_nee_rgb8.*
crates/es/src/cmd/showcase.rs
crates/es/tests/video.rs
crates/es-env/src/render.rs
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M7/R3-pt-quality.md
docs/packets/M7/R3-pt-quality.ko.md
```

`crates/es-render/src/{view.rs,cpu.rs,renderer.rs,lib.rs,rng.rs}`, `crates/es-render/src/ssim.rs`
(신규), `crates/es-render/slang/{common.slang,pt.slang,restir.slang}`,
`crates/es-render/tests/render.rs`, `tests/golden/render/cornell_pt_nee_rgb8.*`(신규,
생성됨: 4 spp, 바운스 3, NEE, Reinhard, 노출 1 — CPU에서 비트 단위로 고정됨; GPU는
명시된 허용 오차에서 일치), `crates/es/src/cmd/showcase.rs`(플래그),
`crates/es/tests/video.rs`, `crates/es-env/src/render.rs`는 **오직** 새
`RenderConfig` 필드를 `config()`를 통해 (기본값 그대로) 통과시키기 위해서만,
`docs/design/renderer*.md`, `docs/packets/M7/R3-pt-quality*.md`.

## oracle

1. 기존의 모든 골든과 GPU 테스트가 불변; `cargo xtask verify-goldens` 변경 0건
   (`cornell_pt1spp`는 NEE가 꺼져 있고, `light_rgb`는 0이며, `PtRadiance`에 톤맵이 없다).
2. `cargo test -p es-render tonemap_is_bitwise_on_both_sides`(GPU; 디바이스 없으면
   `SKIP`) — 64×64 `PtRadiance` 타일을 CPU와 GPU에서 `Reinhard`와 `Aces`로: `Rgb8`이
   비트 단위로 같고, CPU에서 `RgbF32Linear` 입력 → `Rgb8` 출력이 채널별로 단조롭다.
3. `cargo test -p es-render nee_converges_to_the_same_image` — Cornell, CPU, 4,096
   spp에서 NEE 켜짐 대 꺼짐(작은 타일, 16×16): 선형 radiance의 평균 절대 차이가 평균
   radiance의 1% 미만이고 출력된다 — 두 추정량이 기댓값에서 일치한다.
4. `cargo test -p es-render nee_at_low_spp_has_lower_variance` — 16 spp에서 NEE 켜짐 대
   꺼짐 각각을 4,096 spp NEE 레퍼런스와 비교: NEE의 RMSE가 더 작다(출력됨).
5. `cargo test -p es-render restir_is_unbiased_within_tolerance` — 256개의 독립적인
   시드에 걸쳐 평균한 1 spp에서의 ReSTIR(temporal 꺼짐, spatial 켜짐)을 4,096 spp
   레퍼런스와 비교: |bias| < 평균 radiance의 1%(출력됨; 옛 `1/M` 가중치는 이것을
   통과하지 못한다 — 그 수치도 기록할 것).
6. `cargo test -p es-render gpu_pt_nee_matches_the_cpu` — 새 골든을 CPU가 비트 단위로
   재현한다; GPU는 기존 ReSTIR 스타일 허용 오차 안에서, 출력됨.
7. `cargo test -p es-render ssim_is_one_for_identical_and_falls_with_noise` —
   `ssim(a, a) == 1.0`, `ssim(a, a + noise)`는 노이즈 진폭에 따라 단조롭게 감소한다.
8. `cargo test -p es --test video showcase_pt_flags_are_parsed`; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R3-pt-quality.md`.

## acceptance

오라클 1–8(GPU 것들은 RTX 3060과 서버에서). 서버에서: `~/artifacts/plan-v/m7-r3/` 아래
V19b `nominal-00`의 PT 쇼케이스(`--spp 64 --bounces 3`, Reinhard, R1의 카메라), 그
ms/frame을 R1/R2의 것 옆에, 그리고 설계 노트 10절 안의 SSIM 표(Cornell, SO-101). 건너뛴
모든 것(ReSTIR GI, 환경 맵, 광택 BSDF, 스펙트럴한 모든 것)이 오늘 4.2절이 자신의 건너뜀을
나열하는 방식으로 노트에 나열된다.

## forbidden

`cornell_pt1spp`나 어떤 골든이든 옮기는 것; `Rs` `Lambert`를 바꾸는 것; `approx` 밖의
초월함수; 러시안 룰렛이나 어떤 데이터 의존적 루프 경계든(§3.4: 픽셀당 고정된 작업량);
원자적 연산/공유 메모리; `crates/es-gpu/**`; `EnvRendererCfg`에 PT 노브를 주는 것(관측
경로는 R2가 남긴 곳에 그대로 있다); `docs/ARCHITECTURE*.md`. INV-17: 새 트레이트 없음.
