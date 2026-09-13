<!-- Korean translation of docs/packets/M3/W3-splat-importer.md. The English file is the working copy; regenerate this when it changes. -->
# W3-splat-importer — `es-splat` 오프라인 경로: 임포트, 정렬, 바인딩

스펙: §16 (3DGS를 통한 real-to-sim; §16.2 색상 및 위치 정렬, LBS 바인딩; §16.3
산출물), §28.5 W3 행, §15.1 (스플랫 자산이 궁극적으로 공급하는 대상), §3.1 (Z-up, 미터,
xyzw 쿼터니언), §5.3 (디코딩된 콘텐츠 위에서의 `asset_hash`), §1.9 3번 항목(이 경로는
세 번째로 잘려나간다 — 가볍게 유지할 것).
설계 노트: `docs/design/splat-real2sim.md`.
API 노트: `docs/api-notes/gaussian-splat-ply.md`(모든 필드가 `unverified`(미검증)).

§16의 오프라인 절반만 다룬다. §16.3의 스플랫 래스터라이저는 Vulkan과 `es-gpu`가 필요하다;
이 패킷에는 포함되지 않으며 여기서는 아무것도 렌더링하지 않는다.

## context (범위)

```
crates/es-splat/src/lib.rs
crates/es-splat/src/ply.rs
crates/es-splat/src/align.rs
crates/es-splat/src/bind.rs
crates/es-splat/tests/splat.rs
tests/fixtures/splat/cube20_ascii.ply
tests/fixtures/splat/cube20_binary.ply
docs/design/splat-real2sim.md
docs/api-notes/gaussian-splat-ply.md
docs/packets/M3/W3-splat-importer.md
```

## spec (사양)

1. **`SplatScene`** — 배열의 구조체, `f32`: `positions`(3N, m, §3.1 Z-up), `scales`
   (3N, m, `exp` 적용, 가우시안 로컬 프레임), `rotations`(4N, xyzw, 단위 노름, `w >= 0`),
   `opacities`(N, `sigmoid` 적용), `sh_dc`(3N), `sh_rest`(`3 * ((d+1)^2 - 1) * N`, 비어
   있을 수 있음), 그리고 `sh_degree`, `bounds`, `asset: AssetRef`, `warnings`.

2. **`import_ply(bytes) -> Result<SplatScene, SplatError>`** — `binary_little_endian 1.0`과
   `ascii 1.0`. 프로퍼티는 **이름으로** 위치를 찾으며, 오프셋은 선언된 타입 순서로부터
   계산되므로 순서가 바뀌거나 확장된 익스포터도 읽을 수 있다. 인식되지 않는 프로퍼티는
   오류가 아니라 경고다. `sh_degree`는 `f_rest_*` 개수로부터 추론되며 0/9/24/45 중
   하나여야 한다. COLMAP/OpenCV의 Y-down 월드에서 §3.1의 Z-up으로의 축 변환은 위치와
   쿼터니언 벡터 부분에 적용되는 고정 스위즐 `(x, z, -y)`다 —
   **`unverified`(미검증)**이며, 비트 단위로 정확하고 정확히 역변환 가능하도록 쿼터니언
   곱이 아니라 의도적으로 스위즐로 구현했다. `f_rest`는 밴드 회전을 *적용받지 않으며*,
   이는 경고를 발생시킨다. 잘못된 입력은 모두 해당 element나 프로퍼티를 명시하는 타입이
   있는 `SplatError`가 된다 — 패닉도, 조용한 기본값도 절대 없다.

3. **`SplatScene::write_ply(&self) -> Vec<u8>`** — 레퍼런스 프로퍼티 순서로 된
   `binary_little_endian`이며, 두 활성화 함수를 모두 역변환한다. 인코딩은 하나뿐이다:
   리더는 둘을 받아들이지만, 라이터는 교환용 하나를 선택한다.

4. **`Similarity { scale, rot, trans }`** — `fit(src, dst)`는 Horn의 쿼터니언 방법을
   사용하는 Umeyama 방식으로, 4x4 대칭 고유값 문제를 고정 스윕 순환 야코비(cyclic
   Jacobi)로 푼다(`sqrt`와 나눗셈만 사용; 초월함수 없음, 따라서 `es_math::approx`
   의존성도 3x3 SVD도 필요 없음). 대응점 `>= 3`개 필요; 퇴화 입력은 오류. `apply(p)`,
   `apply_scene(&mut)`.

5. **`ColorAffine { gain, bias }`** — 스플랫의 DC 색상과 레퍼런스 샘플 간의 채널별 일반
   최소제곱; 분산이 0인 채널은 순수 오프셋으로 축퇴한다. `SplatScene::apply_color`는
   `sh_dc`를 RGB 공간으로 매핑하고(`0.5 + C0 * dc`) 다시 되돌린다.

6. **`Binding::bind(&SplatScene, &SceneDesc, k)`** — 가장 가까운 geom **표면**까지의
   거리로 `k`개의 최근접 바디를 찾는다(`Sphere`/`Box`/`Capsule`은 정확, 그 외는 geom
   원점으로 대체), 역거리 가중치를 1로 정규화, `k`는 `1..=4`로 클램프. 표면에서
   `WELD_EPS` 이내인 가우시안은 용접 처리된다: 가중치 `1.0`, 바디 하나.
   `skin(&Binding, &SplatScene, &BTreeMap<StableId, Pose>) -> SkinnedSplats`는
   `delta_b = pose_now_b * rest_b^-1`을 선형 블렌드 스키닝으로 적용한다; 포즈가 주어지지
   않은 바디는 자신의 rest 포즈로 기여한다.

7. **`asset_hash`** — 도메인 태그, 개수, `sh_degree`, 그리고 **파일 순서**로 나열된 모든
   속성 배열(각각 길이 접두) 위에서 계산한 `blake3`; 디코딩된 값 위에서 계산하므로
   인코딩이 이를 바꿀 수 없다(§5.3).

## oracle (오라클)

```
cargo fmt -p es-splat --check
cargo clippy -p es-splat --all-targets -- -D warnings
cargo test -p es-splat
cargo xtask layering
cargo xtask context-budget
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `tests/fixtures/splat/cube20_binary.ply`(가우시안 20개, SH 3차)를 임포트하면 `write_ply`가
  파일을 **바이트 단위로 완전히 동일하게** 재현한다. 이 픽스처의 원본 스케일/불투명도
  값은 `exp`/`ln`과 `sigmoid`/`logit` 쌍의 고정점에서 의도적으로 뽑은 것이다 —
  `docs/design/splat-real2sim.md` §1.1 참조 — 그리고 별도 테스트가 정확히 일치하지 않는
  경우를 상대 오차 `1e-6`으로 고정한다.
- ascii 픽스처는 binary 픽스처와 비트 단위로 동일한 배열로 디코딩되며, 동일한
  `asset_hash`를 갖는다.
- `Similarity::fit`은 고정된 splitmix64 시드에 대한 의사난수 점 10개로부터 알려진
  `(s, R, t)`를 `1e-6` 이내로 복원한다.
- `ColorAffine::fit`은 알려진 채널별 gain과 bias를 복원한다.
- 두 바디 중 하나의 표면 위에 놓인 가우시안은 그 바디에 가중치 `1.0`을 받으며, 그 바디를
  이동시키면 정확히 같은 벡터만큼 이동한다.
- 잘린 헤더, 알 수 없는 `format`, 누락된 `x`, 잘못된 `f_rest` 개수, 짧은 바이너리
  페이로드는 각각 고유한 `SplatError` variant를 발생시킨다; 어느 것도 패닉하지 않는다.

## forbidden (금지)

- `crates/es-splat/**`, `tests/fixtures/splat/**`, 그리고 위 세 문서 외의 모든 것.
  `es-assets`, `es-math`, `es-core`는 소비만 될 뿐 수정되지 않는다 — `AssetKind`도
  포함되며, 여기서는 `Splat` variant를 얻지 않는다.
- 렌더링, Vulkan, `es-gpu`: §16.3 래스터라이저는 별도 패킷이다.
- 새 의존성. PLY 리더와 라이터는 여기서 직접 작성한다; 3DGS PLY는 텍스트 헤더와 패킹된
  `f32`일 뿐이다.
- `.splat` / `.ksplat` / `.spz` 디코딩 — API 노트에 미지원으로 문서화되어 있다.
- ICP와 RANSAC(§16.2). `fit`은 다른 누군가가 선택한 대응점을 입력으로 받는다.
- `HashMap`/`HashSet` — `BTreeMap`만 사용.
- 새 확장 지점 트레이트(INV-17); 이 패킷은 트레이트를 전혀 추가하지 않는다.
- 커밋. 오라클은 실행하고 보고할 뿐, 반영(land)하지 않는다.
