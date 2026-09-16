# M7 R2 — RS의 룩: 그림자, 반구 하늘, 하이라이트와 슈퍼샘플링, 옵트인

스펙: §15.3(RS는 비전의 기본값이다; 경로 사이의 RGB는 유사도 임계값이다), §15.1(채널
계약은 불변), §3.2/§3.4(모든 초월함수는 `es_math::approx`를 거친다; 고정된 누적 순서),
§28.10 규칙 1(기본 룩은 단 1바이트도 움직이지 않는다). 확장할 설계 노트:
`docs/design/renderer.md`(+ `.ko.md`) — 3.1절이 `Full` 셰이딩을 얻고, 새 9절. **R1**(그림자
레이가 필요로 하는 any-hit 순회는 BVH의 것이다)에 의존한다.

## the question

RS는 `albedo * (ambient + max(0, n·L) * (1 - ambient))`로 셰이딩한다: 그림자 없음, 하이라이트
없음, 상수 ambient, 픽셀당 샘플 하나. 정직하고, 1995년처럼 보인다. **RS가 커밋된 관측
문서가 렌더하는 것의 1바이트도 바꾸지 않으면서 접촉 그림자, 하늘, 하이라이트, 안티앨리어싱을
가질 수 있는가?**

## spec

* `RenderConfig`는 `#[derive(Default)]`가 붙은 `shading: Shading`을 얻는다:

  ```rust
  pub enum Shading {
      /// Today's flat Lambert + constant ambient. The default; every golden pins it.
      #[default] Lambert,
      Full {
          /// One shadow ray towards `light_dir` per shaded pixel (any-hit, R1's traversal).
          shadows: bool,
          /// Blinn-Phong: specular weight in [0, 1] and shininess exponent; 0.0 disables.
          specular: f32,
          shininess: f32,
          /// Hemisphere ambient: sky colour for n.z = +1, ground colour for n.z = -1, lerped
          /// on (n.z + 1) / 2. Replaces the constant `ambient` when `Full`.
          sky_rgb: [f32; 3],
          ground_rgb: [f32; 3],
          /// Supersampling factor per axis (1 = off). The atlas is rendered at
          /// (w*ssaa, h*ssaa) and box-filtered in fixed row-major order before encoding.
          ssaa: u32,
      },
  }
  ```

  `RenderConfig::rs()`는 `Shading::Lambert`를 유지한다; `RenderConfig::rs_full(atlas)`가
  주관이 담긴 프리셋이다(`shadows: true, specular: 0.25, shininess: 32.0, sky [0.55, 0.65,
  0.85], ground [0.25, 0.22, 0.20], ssaa: 2`).
* `Full`에서의 셰이딩은 CPU(`cpu.rs`)와 `common.slang`에서 동일하며, 이 순서와 이 함수로
  이루어진다: `n`을 face-forward한다; `vis = shadows ? (any_hit(p + n*RAY_EPS, L) ? 0 : 1) :
  1`; `diffuse = max(0, n·L) * vis`; `h = normalize(L - d)`; `spec = specular * pow(max(0,
  n·h), shininess) * vis`, 여기서 `pow`는 `x <= 0`에서 가드된 `es_exp(es_ln(x) *
  shininess)`다; `hemi = lerp(ground, sky, (n.z + 1) * 0.5)`; `rgb_lin = albedo * (hemi +
  diffuse) + spec + emission`. 그다음 기존의 정확한 sRGB 변환과 `u8` 반올림. SSAA:
  슈퍼샘플된 선형 색은 **`ssaa × ssaa` 블록에 대해 행 우선 순서로** 평균된다(고정된 순차
  `+=` 다음 `1/(ssaa*ssaa)` 곱셈 한 번), 그러고 나서야 인코딩된다 — 양쪽 모두에서.
* 파라미터 버퍼는 **끝에서** 자란다(새 슬롯은 기존 것 뒤에 오므로 `Lambert`의 인덱스는
  움직이지 않는다); `Rs` 커널은 파라미터에서 읽은 셰이딩 플래그로 분기한다 — 커널 두 개가
  아니라 하나.
* `es video showcase --look lambert|full`(기본값 `lambert`, 그래서 V9의 비트-동일성
  오라클이 의미를 유지한다)이 프리셋을 고른다. `EnvRendererCfg`와 관측 경로는 이 패킷에서
  그 옵션을 **얻지 않는다**: 규칙 1 — 커밋된 문서의 룩은 그 문서의 것이고, 거기에 노브를
  제공하는 것은 나중의, 해시를 아는 패킷의 일이다.
* 골든: CPU 생성기(`generate_goldens`, 확장됨)가 한 번 생성하는 `cornell_rs_full_rgb8`
  (u8, 64×64, `ssaa: 2`를 쓴 `rs_full` 프리셋); `Full`의 depth/seg/normal은 `Lambert`와
  같은 버퍼이며 새 골든이 필요 없다 — 테스트가 그것들이 `cornell_rs_*`와 비트 단위로 같음을
  단언한다. `ssaa: 2`에서 지오메트리 채널은 각 블록의 **첫** 서브샘플로부터 쓰이므로 위의
  동등성이 여전히 성립한다(이를 노트에 명시할 것).

## context

`cargo xtask check-scope`가 읽는 글롭(파서는 정확히 `## context` 제목과 펜스 블록 또는 불릿 목록을 원한다), 그 아래는 같은 범위를 산문으로:

```
crates/es-render/src/view.rs
crates/es-render/src/cpu.rs
crates/es-render/src/renderer.rs
crates/es-render/src/lib.rs
crates/es-render/slang/common.slang
crates/es-render/slang/raster.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_rs_full_rgb8.*
crates/es/src/cmd/showcase.rs
crates/es/tests/video.rs
docs/design/renderer.md
docs/design/renderer.ko.md
docs/packets/M7/R2-rs-look.md
docs/packets/M7/R2-rs-look.ko.md
```

`crates/es-render/src/{view.rs,cpu.rs,renderer.rs,lib.rs}`, `crates/es-render/slang/{common.slang,
raster.slang}`, `crates/es-render/tests/render.rs`(새 테스트 + 확장된 생성기),
`tests/golden/render/cornell_rs_full_rgb8.*`(신규, 생성됨), `crates/es/src/cmd/showcase.rs`
(`--look` 플래그), `crates/es/tests/video.rs`(`--look` 인자 테스트), `docs/design/renderer*.md`,
`docs/packets/M7/R2-rs-look*.md`.

## oracle

1. 기존의 모든 골든과 GPU 테스트가 불변; `cargo xtask verify-goldens` 변경 0건. 처음이자
   마지막.
2. `cargo test -p es-render lambert_is_the_default_and_is_byte_identical` — `RenderConfig::rs()`가
   `Shading::Lambert`를 갖고, 새 코드 경로를 거친 `Lambert` 렌더가 골든과 비트 단위로
   같다(이것이 오라클 1을 API 수준의 단위 테스트로 말한 것이다).
3. `cargo test -p es-render cpu_full_shading_reproduces_its_golden`과
   `gpu_full_shading_matches_the_cpu`(GPU; 디바이스 없으면 `SKIP`) — `Rgb8`은 비트
   단위, `Full`의 세 지오메트리 채널은 `Lambert`의 것과 비트 단위로 같다.
4. `cargo test -p es-render a_shadow_ray_darkens_only_occluded_pixels` — Cornell에서
   `shadows: true`와 `false`(그 밖에는 동일)를 비교해, 바뀐 픽셀 집합이 비어 있지
   않고 바뀐 모든 픽셀에서 `L`을 향한 `any_hit`이 CPU에서 참이다.
5. `cargo test -p es-render ssaa_is_a_fixed_order_box_filter` — 삼각형 하나짜리 장면에서
   `ssaa: 2`가 행 우선 순서의 네 서브샘플에 대한 손 계산 평균과 같다(선형 값에서 비트
   단위, 그다음 인코딩됨).
6. `cargo test -p es --test video showcase_look_flag_is_parsed`; `cargo xtask ci`;
   `cargo xtask check-scope docs/packets/M7/R2-rs-look.md`.

## acceptance

오라클 1–6(3은 로컬 RTX 3060과 서버에서). 서버의 `~/artifacts/plan-v/m7-r2/` 아래에서
`--look full`로 렌더된 V19b `nominal-00`의 쇼케이스 프레임 세트 하나, 그 ms/frame이 R1의
`lambert` 수치 옆에(관측). 설계 노트 9절이 셰이딩 방정식, SSAA 순서, 파라미터 레이아웃,
관측 경로가 왜 그 노브를 얻지 않았는지를 기록한다.

## forbidden

`Shading::Lambert`의 산술이나 기본값을 바꾸는 것; `EnvRendererCfg` / 관측 경로에 새 룩을
주는 것; 기존의 어떤 골든이나 픽스처든; 호스트나 `GLSL.std.450`에서 온 `pow`/`exp`/`log`;
원자적 연산이나 공유 메모리; 두 번째 커널; `crates/es-gpu/**`; `docs/ARCHITECTURE*.md`.
INV-17: 새 트레이트 없음.
