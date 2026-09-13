<!-- Korean translation of docs/design/splat-real2sim.md. The English file is the working copy; regenerate this when it changes. -->
# Splat real-to-sim: 오프라인 절반

`es-splat`(레이어 5)은 §16 real-to-sim 경로다. 이 노트는 M3 W3가 *오프라인*으로 구현하는 부분을 다룬다: 3D 가우시안 스플래팅 캡처를 임포트하고, 위치와 색상 면에서 물리 씬에 정렬하며, 관절체(articulated body)에 바인딩하여 비주얼이 물리를 따라가게 한다. 래스터라이저 — §16.3의 나머지 절반 — 는 Vulkan이 필요하며 뒤로 미뤄져 있다; 여기서는 아무것도 렌더링하지 않는다.

이 크레이트를 작게 유지하는 틀은 §16.3 자체가 제시한 것이다: **정확한 정렬은 불필요하고, 분포 일치가 목표다.** 아래의 모든 수치는 측정값이 아니라 피팅(fit) 값이며, §16.1의 결론은 색상과 위치를 대략 맞추는 것이 정책 전이(policy transfer)를 움직인다는 것이다. 따라서 폐형식(closed-form) 피팅만 사용하고, 반복 정제도 ICP도 아직 없다.

§1.9에 따라 이 경로 전체는 범위 압박 시 세 번째로 잘려나간다. 이는 각주가 아니라 설계 제약이다: 이 크레이트 바깥의 어떤 것도 여기에 의존해서는 안 된다.

## 1. `SplatScene`

배열의 구조체(structure-of-arrays), `f32`, 속성마다 `Vec` 하나씩, 모두 길이 `count`(또는 그 배수):

```
positions   3 * count   m, §3.1 Z-up world
scales      3 * count   m, per-axis standard deviation in the Gaussian's own frame (exp applied)
rotations   4 * count   xyzw, unit norm, w >= 0 (§3.1)
opacities   1 * count   alpha in [0, 1] (sigmoid applied)
sh_dc       3 * count   degree-0 SH coefficient, RGB
sh_rest     3 * ((d+1)^2 - 1) * count, or empty
```

여기에 `sh_degree`, `bounds`(`positions`에 대한 축 정렬 min/max), `asset: AssetRef`, `warnings`가 더해진다. 배열의 구조체(SoA)를 쓰는 이유는, 소비자가 GPU 업로드와 속성 하나씩만 건드리는 세 개의 패스이기 때문이고, 속성별 `Vec<f32>`가 직렬화 단계 없이 `blake3`로 해시할 수 있는 형태이기 때문이다.

디코딩 대상 파일 레이아웃은 `docs/api-notes/gaussian-splat-ply.md`에 있다(그곳의 모든 내용은 `unverified`(미검증)이다).

### 1.1 활성화 함수는 임포트 시점에 적용되며, 이는 손실을 수반한다

파일은 `log(sigma)`와 `logit(alpha)`를 저장한다; `SplatScene`은 `sigma`와 `alpha`를 저장한다. `write_ply`는 둘 다 역변환한다. `exp`와 `ln`은 `f32`에서 정확한 역함수 관계가 아니다: 무작위 입력에 대해 스케일 값의 약 10%, 불투명도 값의 약 50%가 라운드트립 후 1 ULP만큼 어긋난다. 따라서 **file -> scene -> file이 바이트 단위로 정확히 일치하는 것은 활성화 함수 쌍의 고정점(fixed point)인 값들뿐이다.** 체크인된 픽스처들은 의도적으로 그 고정점 집합에서 생성되었으므로, 라운드트립 테스트는 `libm`의 반올림이 아니라 원래 측정하려는 대상 — 헤더, 프로퍼티 순서, 바이트 레이아웃, 가우시안 순서 — 을 측정한다. 별도 테스트가 정확히 일치하지 않는 경우를 상대 오차 `1e-6`으로 고정한다.

대안(원본 파일 값을 그대로 저장하고 접근 시점에 활성화 함수를 적용)은 어떤 입력에 대해서도 라운드트립을 정확하게 만들지만, `exp`/`sigmoid`를 GPU 업로드 경로를 포함한 모든 소비자에게 밀어넣게 된다. 임포트 시점에 디코딩하는 방식은 glTF 임포터가 정점 데이터에 대해 하는 것과, §5.3이 말하는 "디코딩된 콘텐츠를 해시한다"는 것과 일치하므로 이 방식을 택한다.

**결정성(Determinism) (P-M3 리뷰, §3.2/§3.4).** `exp`, `ln`, `sigmoid`, `logit`은 모두 §5.3 체인 입력인 `SplatScene::asset_hash`에 반영되므로, 호스트의 `libm`에 의존해서는 안 된다: 동일한 캡처를 임포트하는 두 대의 기계는 반드시 동일하게 해시되어야 한다. 따라서 `sigmoid`/`logit`/`scale`의 디코딩과 인코딩은 `f64::exp`/`f64::ln`을 계산한 뒤 한 번 `f32`로 반올림하는 대신, GPU와 공유하는(`crates/es-math/slang/approx.slang`) 유일한 결정적 초월함수 구현인 `es_math::approx::exp` / `es_math::approx::ln`을 전적으로 `f32`로 거친다. `es_math::approx`는 오차를 IEEE 정확 반올림이 아니라 ULP 단위로 제한하므로, 이는 호스트 `libm`이 주던 것과는 엄밀히 다른(그리고 다소 덜 정확한) 근사다; `a_non_fixed_point_activation_round_trips_to_a_relative_1e_6`의 상대 오차 `1e-6` 허용치가 이미 그 차이를 포괄하며, 바이트 단위로 정확한 픽스처들이 뽑히는 고정점 집합도 그에 맞춰 바뀌었다 — `0.0`은 `es_math::approx` 아래에서 `exp`/`ln`과 `sigmoid`/`logit` 양쪽 모두의 고정점이므로(`exp(0) == 1`, `ln(1) == 0`, `sigmoid(0) == 0.5`, `logit(0.5) == 0`, 모두 비트 단위로 정확), 두 `cube20_*.ply` 픽스처는 모든 정점에 대해 `scale_0..2`와 `opacity`를 `0.0`으로 설정하여 재생성되었다; 다른 모든 필드(위치, 법선, SH 계수, 회전)는 변경되지 않았다.

### 1.2 `AssetKind`에는 `Splat` variant가 없다

`AssetKind`는 이 패킷이 소유하지 않는 `es-assets`에 있으므로, `AssetRef::kind`는 `Mesh`이고 `path`가 `.ply`를 담는다. `AssetKind::Splat`을 추가하는 패킷이 이를 바꿔야 한다; 오늘은 이 variant로 분기하는 곳이 없다.

## 2. 축 변환

3DGS 재구성은 COLMAP/OpenCV 카메라 관례를 물려받는다: 월드 프레임은 첫 번째 카메라의 것이므로 **+Y가 아래쪽, +Z가 전방**이다. §3.1은 오른손 좌표계 Z-up이다. 둘 사이의 고정 회전은 `R_x(-90 deg)`다:

```
x_es = +x_gs        y_es = +z_gs        z_es = -y_gs
```

이는 `unverified`(미검증)다 — COLMAP 기반 캡처와 레퍼런스 트레이너의 출력에는 성립하지만, SfM 단계에서 이미 up-벡터 보정을 적용한 캡처에는 틀린다. 이것은 *관례에 대한 추측*이며, 바로 이 때문에 §16.2가 그 뒤에 피팅된 `T_robot_scan`을 둔다: 남은 회전 오차는 아래 §3의 similarity에 흡수되므로, 잘못된 추측은 초기 추정값의 정확도만 깎아먹을 뿐 최종 정렬의 정확성에는 영향을 주지 않는다.

이는 쿼터니언 곱이 아니라 **부호 반전을 동반한 성분 스위즐(swizzle)**로 구현된다. 동일한 회전이면서 비트 단위로 정확하고 정확히 역변환 가능한데, 이것이 PLY 라운드트립을 의미 있게 만드는 요소다. 쿼터니언도 벡터 부분에 동일한 스위즐을 적용받는다(회전에 의한 켤레 연산은 벡터 부분에 그 회전을 작용시키고, 스칼라 부분은 불변이다). 스케일은 변환하지 *않는다*: 이들은 월드 방향이 아니라 가우시안 자체 프레임에서의 범위(extent)이기 때문이다.

## 3. 위치 정렬: `Similarity`

```
Similarity { scale: f64, rot: Quat, trans: Vec3 }     apply(p) = scale * (rot * p) + trans
Similarity::fit(src: &[Vec3], dst: &[Vec3]) -> Result<Similarity, SplatError>
```

회전에는 Horn의 쿼터니언 방법을 곁들인 Umeyama의 폐형식(closed form)을 사용하며, >= 3개의 대응점이 필요하다(공선이 아닌 세 점은 회전을 결정하지만, 두 점은 그렇지 않다):

1. 중심점(centroid)을 구한 뒤 점 집합을 중심화한다. 합산은 고정된 인덱스 순서를 사용한다.
2. `S = sum_i src_i_centred * dst_i_centred^T` (3x3).
3. `S`로부터 Horn의 대칭 4x4 `N`을 구성한다(`(w, x, y, z)` 순서 형태). 최대 고유값의 고유벡터가 회전 `src -> dst`이다.
4. `scale = sum_i dst_i_centred . (rot * src_i_centred) / sum_i |src_i_centred|^2`.
5. `trans = dst_mean - scale * (rot * src_mean)`.

**왜 SVD가 아니라 고유값(eigen) 방식인가.** 3x3 SVD는 의존성(`nalgebra`이며, 레이어 5는 이를 위해 의존성을 들이지 않는다)이거나, 반사(reflection)를 피하기 위한 부호 보정을 포함한 `A^T A`에 대한 야코비(Jacobi) 약 200줄이다. Horn의 `N`은 4x4 대칭 행렬이므로 **순환 야코비 고유값 스윕(cyclic Jacobi eigenvalue sweep)** — 스윕당 여섯 번의 회전, 고정된 `(p, q)` 순서, 고정된 스윕 횟수 — 로 `sqrt`와 나눗셈만으로 약 60줄에 풀 수 있다. 초월함수가 전혀 필요 없으므로 §3.2 / `es_math::approx` 규칙은 애초에 필요가 없어짐으로써 충족되고, 고정된 연산 순서 덕분에 별도의 결정성 계약 없이도 결과가 재현 가능하다. 쿼터니언 파라미터화는 또한 반사를 만들어낼 수 없는데, 이는 SVD가 `det` 보정을 필요로 하는 실패 모드다.

퇴화된(degenerate) 입력은 조용히 항등원으로 처리되지 않고 오류가 된다: 점이 3개 미만인 경우, 쌍의 개수가 일치하지 않는 경우, `src` 집합의 중심화된 점들이 퍼짐이 0인 경우(모두 동일하여 스케일이 정의되지 않음), 또는 유한하지 않은 좌표인 경우.

ICP와 RANSAC(§16.2)은 여기에 없다. `fit`은 다른 누군가가 선택한 대응점 — 지정된 마커, 등록 UI, 또는 검출기 — 을 입력으로 받으며, 그것이 M3 W3 범위의 전부다.

## 4. 색상 정렬: `ColorAffine`

§16.1이 이를 특별히 짚는다: 스캔의 색공간을 로봇의 실제 카메라에 매핑하는 것이 정책을 다시 in-distribution으로 되돌리는 요소였다. 스펙은 다항식 매핑을 요구하며, 그 첫 번째 유용한 항이 채널별 아핀(affine)이므로 이것이 바로 그 내용이다:

```
ColorAffine { gain: [f64; 3], bias: [f64; 3] }        apply(rgb_c) = gain_c * rgb_c + bias_c
ColorAffine::fit(src: &[[f32; 3]], dst: &[[f32; 3]]) -> Result<ColorAffine, SplatError>
```

채널별로 독립적인 일반 최소제곱(ordinary least squares)이다: `gain = cov(src, dst) / var(src)`, `bias = mean(dst) - gain * mean(src)`. `src`에서 분산이 0인 채널은 0으로 나누는 대신 `gain = 1, bias = mean(dst) - mean(src)` — 순수한 오프셋 — 이 된다. 교차 채널 항도, 감마도 없다: 대조할 실제 캡처가 없는 상태에서 이 둘 중 하나라도 추가하는 것은 아무것도 없는 데 대한 피팅이 될 것이다. 이 설계는 `ColorAffine`을 고정된 변환으로 굳히지 않고 씬에 저장되는 값으로 유지함으로써 여지를 남겨둔다.

`SplatScene::apply_color`는 `sh_dc`를 RGB로 변환하고(`0.5 + C0 * dc`), 매핑을 적용한 뒤 다시 변환해 되돌린다. 1..3차는 그대로 둔다: 아핀 매핑은 (바이어스 항 때문에) 선형이 아니므로 상위 밴드에 일관되게 작용할 방법이 없고, 뷰 의존적 색상을 보정하려면 아직 존재하지 않는 렌더러가 필요하다.

색상 정렬은 언젠가 `scene_hash`(§16.2)에 포함되어야 한다. 이 패킷에는 그것을 해시해 넣을 `SceneDesc`를 만드는 부분이 없으므로 아직은 포함되지 않는다.

## 5. LBS 바인딩

§16.2의 첫 번째 설계 결정: 스플랫은 물리를 대체하지 않는다. 각 가우시안은 하나 이상의 강체(rigid body)에 올라타며, 그 강체들이 어디에 있는지는 §17의 물리가 알려준다.

```
Binding {
    bodies:  Vec<StableId>,   // index space for the weights
    rest:    Vec<Pose>,       // each body's world pose at bind time
    indices: Vec<[u16; 4]>,   // per Gaussian, k <= 4 body indices
    weights: Vec<[f32; 4]>,   // per Gaussian, sums to 1
}
Binding::bind(&SplatScene, &SceneDesc, k: usize) -> Binding
skin(&Binding, &SplatScene, &BTreeMap<StableId, Pose>) -> SkinnedSplats
```

`bind`는 바디 트리를 따라가며 월드 포즈를 구한 뒤, 각 가우시안에 대해 각 바디의 **가장 가까운 geom 표면**까지의 거리를 측정하여 가장 작은 `k`개를 유지한다(`k`는 `1..=4`로 클램프). 가중치는 역거리(inverse distance)를 정규화한 값이다. 표면 위에 놓인 가우시안 — 거리가 `WELD_EPS` 미만 — 은 용접(weld) 처리된다: 그 바디에 가중치 `1.0`을 부여하고 나머지는 없음. 이는 `1/d`에 대한 퇴화 케이스 가드이자 동시에 흔한 경우이기도 한데, 어떤 링크의 사진들로부터 재구성된 스플랫은 실제로 그 링크 위에 *있기* 때문이다.

표면 거리는 `Sphere`, `Box`, `Capsule`에 대해서는 정확하며, 그 외(`Mesh`, `HeightField`, `Cylinder`, `Ellipsoid`, `Plane`)에 대해서는 geom 원점까지의 거리로 대체한다. 볼록 분해(convex-decomposition) 프록시 — §16.2가 제시하는 물체 지오메트리에 대한 실제 해법 — 는 메시에 대해서도 올바른 거리를 줄 것이며, 이를 만드는 패킷의 몫이다. 그 전까지는 정확해 보이지만 정확하지 않은 무언가로 근사하는 대신, 이 대체 방식을 그대로 문서화해 둔다.

`skin`은 바디마다 `delta_b = pose_now_b * rest_b^-1`을 계산한 뒤, 가우시안마다 위치에 대해서는 `sum_k w_k * (delta_k . p)`를 블렌딩하고, 회전에 대해서는 가중치가 가장 큰 바디에 부호를 맞추고 재정규화한 가중 쿼터니언 합을 사용한다. 이 쿼터니언 블렌딩은 표준적인 LBS 근사다: 엄밀한 회전 평균이 아니며 상대 회전이 클 때 축소(shrink)되는데, 스플랫의 경우 이는 서로 반대로 회전하는 두 바디 사이에서 타원체 방향이 약간 틀어지는 것을 의미한다. 이것이 `domain_gap` 수치(§10)에서 실제로 드러난다면 듀얼 쿼터니언 스키닝이 해법이지만, 그 전까지는 그런 코드를 들일 가치가 없다. 포즈가 주어지지 않은 바디는 해당 가우시안을 원점으로 붕괴시키는 대신 그 rest 위치에 그대로 둔다.

이는 CPU 경로다. GPU 경로는 컴퓨트 셰이더에서 동일한 산술을 수행하며 래스터라이저와 함께 다뤄질 것이다.

## 6. 해싱

`asset_hash`는 도메인 태그, 가우시안 개수, `sh_degree`, 그리고 파일 순서대로 나열된 모든 속성 배열(각각 길이 접두) 위에서 계산한 `blake3`다. **디코딩된** 값 위에서 계산하므로, 동일한 캡처를 ascii PLY로 읽든 `binary_little_endian` PLY로 읽든 동일하게 해시되며 — 테스트로 확인됨 — 향후 `.spz` 디코더도 이 성질을 별도 작업 없이 그대로 물려받는다. **파일 순서** 위에서 계산하는 이유는, 3DGS PLY에는 정렬 기준으로 삼을 표준적인 가우시안 순서가 없고 하나를 새로 만드는 것은 누구도 요청하지 않은 정렬이 될 것이기 때문이다; 파일의 순서가 곧 그 캡처의 정체성이다.

해시되지 않는 것: `warnings`, (`positions`로부터 유도되는) `bounds`, 그리고 `AssetRef` 자신(그 `hash` 필드가 *바로* 이 다이제스트다).
