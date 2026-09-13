<!-- Korean translation of docs/packets/M4/W8-renderer.md. The English file is the working copy; regenerate this when it changes. -->

# W8-renderer — `es-render`: 컴퓨트 패스 트레이서, ReSTIR DI, à-trous 디노이즈

Spec: §15.3(`PT` 렌더 경로; 두 경로가 공유하는 출력 계약), §15.4(가속 구조 — *구현되지 않음*),
§28.6(패스 트레이서 + ReSTIR + SVGF), §1.9 항목 2(패스 트레이서는 스코프 압박에서 두 번째로
잘리는 것 — 최소한의 정직한 버전을 만들 것), §1.4(골든 이미지가 오라클, PT/RS 채널 일치),
§3.1, §3.2 `DET-010`, §3.3, §3.4(전역 RNG 없음, 결정론적 실행), §16.2(스플랫 경로는 같은
crate의 세 번째 렌더 경로 — 훅만). 설계: `docs/design/renderer.md`.

`docs/packets/M1/W2-tile-atlas.md` 위에 지어지며, 같은 crate에서 아틀라스, 채널 계약,
래스터라이저를 소유한다.

## context (범위)

```
crates/es-render/src/{cpu,rng,view,renderer}.rs
crates/es-render/slang/{pt,restir,svgf,rng,common}.slang
crates/es-render/tests/render.rs
tests/golden/render/cornell_pt1spp.{bin,json}
docs/design/renderer.md
docs/packets/M4/W8-renderer.md
```

## spec (사양)

- `RenderPath::Pt { spp, bounces, restir, svgf }`. 디퓨즈만, 코사인 가중 반구 샘플링(가중치가
  `cos`와 `1/pi`를 상쇄하므로 throughput은 알베도의 단순 곱셈이다 — pdf 나눗셈도 0/0도 없음),
  고정 바운스 개수, **러시안 룰렛 없음**(이 바운스 개수에서는 아무 이득 없이 픽셀당 작업을
  데이터 의존적으로 만들 뿐이다). 샘플은 오름차순 인덱스로 순차 누적된다: 순서가 루프로
  고정되므로 저렴한 합이 곧 재현 가능한 합이다(§18.4는 순서가 *고정되지 않은* 리덕션을 위한
  것이다).
- 깊이, 세그멘테이션, 노멀은 어떤 RNG 추출보다도 먼저, 래스터라이저가 쓰는 것과 같은
  `nearest_hit`을 통해 샘플 0의 1차 히트에서 온다. 그래서 §15.3의 "깊이/세그/노멀은 경로
  사이에서 비트 동일"은 구성상 성립하고, `pt_and_rs_agree_on_geometry`가 그것을 깨는 사람을
  잡아낼 테스트다.
- RNG(`rng.rs` + `rng.slang`): 카운터 기반이며 `(seed, view, pixel, sample, bounce, stream)`으로
  주소되는, `es_env::rng`의 설계다(§3.4: 전역 RNG 없음). **32비트 Murmur3 `fmix32`이지
  splitmix64가 아니다** — Slang의 `uint64_t`는 `shaderInt64`가 필요한데, §3.3이 모든 타깃에서
  그것을 보장하지 않는다. 고정 바운스 디퓨즈 패스 트레이서를 넘어서는 통계적 품질은
  `미검증 (unverified)`이다.
- ReSTIR DI(`restir.slang`, 세 진입점, 세 디스패치, 패스당 버퍼 하나씩 기록): 이미셔티브
  삼각형 위에서 8개의 균일 후보로 비가려짐 타겟 함수에 대한 가중 예비 샘플링, 생존자에 대해
  그림자 스캔 한 번; 같은 픽셀에서 이전 `render()` 호출의 예비에 대한 시간적 재사용; 표준
  깊이/노멀 유사성 테스트를 가진 네 개의 *고정* 이웃 오프셋에 대한 공간적 재사용. 이미셔티브 =
  이름이 `_light`로 끝나는 geom(`SceneDesc`에는 이미션 필드가 없고 `es-assets`는 범위 밖).
  출력은 `PtRadiance`를 직접광 추정치로 대체한다.
  **건너뛰었고, 코드에도 그렇게 이름 붙어 있는 것:** MIS 가중치(GRIS pairwise MIS가 아니라
  편향된 `1/M` 결합), 재사용 시 편향 보정 가시성 재검사, ReSTIR GI 전체, 이미셔티브 삼각형
  외의 모든 광원 타입, 모션 벡터 재투영(그래서 시간적 재사용은 1프레임에서 아무 일도 하지
  않고 정적 카메라를 가정한다), `M` 클램프를 넘어서는 예비 노화.
- `svgf.slang`: 깊이/노멀 에지 스토핑을 가진 5탭 B-스플라인 웨이블릿의 스트라이드 `1 << i`
  à-trous 반복 `n`번; 노멀 가중치는 초월함수 없이 다섯 번의 제곱으로 `n^32`다.
  **이것은 SVGF가 아니다**: 색상의 시간적 누적도, 분산 추정도, 분산 유도 휘도 가중치도,
  분산 프리필터도, 디스오클루전 처리도, 히스토리 기반 커널 확장도 없다. §28.6의 결과물과
  커널 자신의 문서 주석 때문에 그렇게 이름 붙었을 뿐이다.
- 가속 구조 없음(§15.4): `es-gpu`는 컴퓨트 파이프라인만 노출한다. 두 경로 모두 평평한 삼각형
  배열을 인덱스 순서로 스캔한다 — 정직한 한계는 수백 개의 삼각형이다.
- 골든 씬에는 **동일 평면 위의 면이 없다**. 두 삼각형이 평면을 공유하는 곳에서는 광선이 같은
  `t`에서 둘 다 맞고, 승자는 barycentrics가 `u + v <= 1`의 어느 쪽에 떨어지는지로 정해지는데 —
  CPU와 GPU가 이를 1 ULP만큼 다르게 답할 수 있다.

## oracle (오라클)

```
cargo fmt -p es-render --check
cargo clippy -p es-render --all-targets -- -D warnings
cargo test -p es-render -- --nocapture
cargo xtask layering && cargo xtask verify-goldens && cargo xtask context-budget && cargo xtask check-spec-refs
```

`cornell_pt1spp`는 `es_render::cpu::path_trace`로 한 번 생성된다 — 1 spp, 바운스 2, ReSTIR와
SVGF는 꺼짐: RNG, 바운스 루프, 이미셔티브 히트를 운동시키는 가장 작은 구성이며, 64 샘플의
노이즈까지 고정하지 않고도 골든이 비트 정확하게 고정할 수 있는 유일한 구성이다. 디바이스나
`slangc`가 없으면 GPU 테스트는 이유를 출력하며 `SKIP`한다.

## acceptance (수용 기준)

NVIDIA RTX 4060 Laptop GPU(드라이버 592.82, Slang 2026.8), 64×64, 96삼각형에서 측정:

- `gpu_path_tracer_matches_the_cpu_reference_at_1spp` — **비트 동일, 최대 ULP 0**.
- `gpu_pt_and_rs_agree_on_geometry` — `Depth32`, `SegmentationId`, `Normal`이 경로 사이에서
  비트 동일(§15.3), GPU에서; 같은 단언이 `cpu::tests::pt_and_rs_agree_on_geometry`에서 CPU로도
  돌아가므로 디바이스 없이도 검사된다.
- `gpu_restir_and_svgf_match_the_cpu_within_tolerance` — 주장 ≤ 이미지 피크의 1e-5,
  **측정 1.3e-7(최대 10 ULP)**. SVGF만 하면 비트 동일하고, ReSTIR가 비트 동일하지 않은 유일한
  경로다. 재사용 패스가 이웃에 걸쳐 `p̂ · W · M`을 합산하고 컴파일러가 `NoContraction`
  아래에서도 식 내에서 재결합할 수 있기 때문이다. 허용 오차는 픽셀당이 아니라 이미지 피크에
  대해 명시된다: 렌더링된 이미지 대부분은 0에 가깝고, 1e-7 픽셀에서의 1e-7 차이는 아무도
  구별할 수 없는 이미지에 대해 100% 상대 오차다.
- CPU 단위 테스트: `path_trace`의 결정론, ReSTIR와 SVGF가 켜졌을 때의 유한성과 비음수성,
  반구 안에 있고 단위 길이인 코사인 샘플, RNG 스트림 분리.

## forbidden (금지)

`context` 밖의 모든 파일 — 특히 `crates/es-usd`, `es-script`, `es-data`, `es-eval`, `es-py`,
`es-compile`, `crates/es`, 루트 `Cargo.toml`. 테스트를 통과시키려고 골든을 편집하는 것
(§1.4); 발산은 ULP 개수로 보고한다. ReSTIR나 SVGF가 발표된 알고리즘이라고 주장하는 것 —
위의 건너뛴 목록은 코드에 계속 남아 있어야 한다. 이 패킷 아래서 빠진 절반을 추가하는 것(MIS
가중치, 분산 추정, ReSTIR GI): §1.9 항목 2는 이 경로가 삭감 가능하다고 말하므로, §28.6
패킷이 더 요구할 때까지 최소한으로 유지한다. 새 확장 포인트 트레이트(`INV-17`).
`HashMap`/`HashSet`, FP 원자적 연산, 커널 안의 `std` 초월함수(§3.4, §3.2). 레이 트레이싱이나
그래픽스 Vulkan 확장 — 그것은 `es-gpu`의 패킷이다. 스플랫 렌더링(§16.2). 센서 리얼리즘
(§18.3). `RS`와 `PT` RGB 사이의 SSIM 비교: 수렴된 PT 렌더와 SSIM 구현이 필요하며, 둘 다
여기서는 `미검증 (unverified)`이다.
