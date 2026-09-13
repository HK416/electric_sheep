<!-- Korean translation of docs/packets/M1/P-M1-R1.md. The English file is the working copy; regenerate this when it changes. -->

# P-M1-R1 — 외부에서 생성된 observation golden

M1 리뷰 블로커(`docs/reviews/M1.md`, Blocker, 게이트 6)를 고친다:
`tests/golden/observation/*`는 `cargo test -p es-compile --test gen_goldens -- --ignored`로,
즉 그것이 검사하는 바로 그 커널들에 의해 생성되었으므로, 회귀는 잡을 수
있었지만 틀린 답은 결코 잡을 수 없었다 — 잘못된 반화소 컨벤션이 지적당하는 대신 그대로
굳어졌을 것이다(spec 1.4).

Spec: spec 1.4, spec 7.7, spec 11.3, spec 3.4. 설계 노트:
`docs/design/observation-lowering.md`. API 다이제스트: `docs/api-notes/torchvision.md`.

## context (범위)

```
crates/es-compile/python/gen_observation_goldens.py  (new — the oracle)
crates/es-compile/tests/gen_goldens.rs               (rewritten — provenance check, not generator)
crates/es-compile/tests/observation_cpu.rs           (sidecar-declared ULP tolerance)
tests/golden/observation/**                          (regenerated from torch)
docs/design/observation-lowering.md                  (§6, §12, §13)
docs/packets/M1/P-M1-R1.md                           (new)
docs/api-notes/torchvision.md                        (new)
```

## forbidden (금지)

`crates/es-compile/src/**` — 커널들은 P-M1-R1의 *대상*이지 그 범위가 아니다. 여섯
golden 중 다섯은 비트 단위로 재현되므로 어느 것도 변경이 필요하지 않았다; golden
집합 밖에서 발견된 유일한 측정된 불일치는 여기서 고쳐지는 대신 이후 패킷을 위해
설계 노트(§12 항목 6)에 기록된다. 범위 밖인 것도 마찬가지다: `es-policy`(그것은
P-M1-R2, 게이트 5)와 오라클을 설치하는 CI 계층(P-M1-R7).

## spec (사양)

- `crates/es-compile/python/gen_observation_goldens.py <out-dir>`는 오직 torch와
  torchvision만으로 같은 여섯 개의 `.bin` + `.json` 쌍을 쓴다. 이 workspace에서는 아무것도
  import하지 않는다. 결정적: CPU, 스레드 하나, `torch.manual_seed(0)`,
  `torch.use_deterministic_algorithms(True)`.
- golden별 오라클 호출: `TF.to_tensor`, `F.interpolate(mode="bilinear",
  align_corners=False, antialias=False)`, 텐서 슬라이싱, f64로 계산되어 f32로 반올림된
  IEC 61966-2-1 EOTF, `TF.normalize`, `torch.stack`. 각 사이드카는 새로운 `"oracle"`
  필드에 그 호출을 기록하고 `"generator"`에 스크립트 이름을 담는다.
- 사이드카는 `"tolerance_ulp": n`을 실을 수 있다. `observation_cpu.rs`는 이를 읽어
  정확히 그만큼의 ULP만 허용한다; 없으면 비교는 바이트 단위다. 사이드카 자체가
  golden이므로, 비교를 느슨하게 하려면 CI 읽기 전용 파일을 수정해야 한다.
- `gen_goldens.rs`는 더 이상 아무것도 생성하지 않는다. 스크립트를 임시 디렉터리로
  다시 실행하여 커밋된 파일과 바이트가 하나라도 다르면 실패한다. 다른 참조-오라클
  테스트들처럼, torch + torchvision을 가진 인터프리터가 없으면 요란하게(이유를 출력하며)
  SKIP한다; `ES_PYTHON`이 인터프리터를 선택한다.

## oracle (오라클)

```
cargo fmt -p es-compile --check
cargo clippy -p es-compile --all-targets -- -D warnings
ES_PYTHON=<venv>/Scripts/python cargo test -p es-compile
cargo xtask check-spec-refs
```

## acceptance (수용 기준)

- `cargo test -p es-compile --test observation_cpu`는 바이트가 torch에서 나온
  golden들에 대해 통과하며, `srgb_to_linear_lut256`이 선언한 7 ULP를 제외하고는 어디에도
  허용오차가 없다.
- `ES_PYTHON`이 설정되어 있으면, `gen_goldens`는 재생성하여 바이트 단위로 일치한다 —
  출처(provenance)가 주석으로 단언되는 대신 기계적으로 검사된다. 설정되어 있지 않으면
  테스트는 `SKIPPED`를 출력하고 다른 모든 테스트는 그대로 실행된다(spec 1.4: 오라클이
  설치되어 있든 아니든 하네스는 존재한다).
- 정확히 하나의 `.bin`이 바뀌었다: `srgb_to_linear_lut256.bin`인데, Rust LUT는 다항식
  피팅이고 golden은 이제 참조 공식이기 때문이다. 나머지 다섯 개의 `.bin` 파일은
  변경되지 않았으며, 이것이 바로 이 작업의 발견이다: 반화소 resize 컨벤션, `/255` 순서,
  CHW permute, `normalize`의 나눗셈, 오래된 것→새것 window가 전부 이미 올바랐다는 것.
