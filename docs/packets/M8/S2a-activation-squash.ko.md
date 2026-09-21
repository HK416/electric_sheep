# M8 S2a — `StateEncoder{Mlp}`가 활성 함수를, `PolicyHead{Regression}`이 스쿼시를 얻고, 커밋된 해시는 아무것도 움직이지 않는다

스펙: §8.3(노드 집합), §14.4("활성 함수와 스쿼시는 노드 파라미터다; 부재 = 기본값 = 오늘의
정규형"), §28.11 파동 1, §1.5(`es-ir`는 6,008 코드 줄이다: 6,000 목표는 넘고 10,000 상한
아래다 — 이 추가분을 작게 유지하고 새 줄 수를 보고하라). 선례: 패킷 M7/R5의
`SensorRender`(`crates/es-ir/src/task.rs` ~723–790: `#[serde(default,
skip_serializing_if)]` + 정규 바이트는 기본값이 아닐 때만 쓰인다). 이것이 닫는 간극:
`docs/design/quadruped-track.md` 3.4의 항목 1–3(ReLU 고정, 헤드 앞 활성 없음, `tanh` 없음).
확장할 설계 노트: `docs/design/learning-lowering.md`(+ `.ko.md`); 결정은
`docs/design/rl-continuation.md` 8절 질문 3에 기록되어 있다.

## 질문

brax의 MLP는 `Dense swish, Dense swish, Dense swish, Dense`이고; rsl_rl의 것은 `Linear ELU …
Linear`이며; 우리 것은 `Mlp { hidden }`을 `Linear ReLU … Linear`로 로워링하는데 헤드 앞에 활성이
없고 어디에도 `tanh`가 없다. **노드 집합이 어떤 활성 함수인지, 마지막 히든 레이어가 활성화되는지,
regression 출력이 스쿼시되는지를 말할 수 있는가 — 커밋된 어떤 문서의 `learning_hash`도, 어떤
로워링된 모듈의 바이트도 움직이지 않고?**

## 사양

* `StateEncoderKind::Mlp { hidden: Vec<u32>, activation: Activation, activate_output: bool }`와
  `enum Activation { Relu, Elu, Swish, Tanh }`; 기본값은 `Relu` / `false`, 둘 다
  `#[serde(default, skip_serializing_if = …)]`. **기본 노드의 정규 바이트는 오늘의 것과 바이트
  단위로 같다** — 오늘 `canonical`은 `format!("{kind:?}")`를 쓴다, 즉 문자열
  `Mlp { hidden: [256] }`; 기본값에는 그 정확한 문자열을 쓰고 두 필드는 기본값이 아닐 때만
  덧붙인다. 커밋된 `learning.toml`과 `learning-pretrained.toml`의 해시를 무엇에도 손대기
  **전에** 16진 리터럴로 고정한다(`main`에서 먼저 계산한다).
* `LearningNode::PolicyHead { …, squash: Squash }`와 `enum Squash { None, Tanh }`, 기본값
  `None`, 같은 serde/canonical 규칙. 검증: `Regression`이 아닌 어떤 헤드 종류에든
  `squash != None`이면 이름 붙은 진단이다(`learning.rs`의 기존 오류 코드 계열을 따른다).
* `NodeSchema`(인스펙터의 파라미터 표, 패킷 M7/E3 — `StateEncoder`와 `PolicyHead`가 자신의
  `ParamType`을 선언하는 곳을 찾아라)가 세 파라미터를 얻어 에디터의 인스펙터가 그것들을
  보여준다; `python/es/builder.py`는 이 노드들을 적는다면 같은 기본값의 키워드 인자를 얻는다.
* 로워링(`crates/es-policy/src/lower/torch.rs` ~648–665와 ~754): `Elu → nn.ELU()`,
  `Swish → nn.SiLU()`, `Tanh → nn.Tanh()`; `activate_output`은 인코더의 마지막 `nn.Linear` 뒤에
  활성을 덧붙인다; `Squash::Tanh`는 regression 헤드의 출력을 reshape 전에 `torch.tanh`로
  감싼다. 기본 그래프는 **바이트 단위로 같은** 소스로 로워링된다(`lowering_hash`도 움직이지
  않는다: 그것도 고정하라).
* 가중치 키와 형태는 셋 중 무엇으로도 바뀌지 않는다(활성 함수는 가중치를 갖지 않는다).

## context

```
crates/es-ir/src/learning.rs
crates/es-ir/src/schema.rs
crates/es-ir/src/**
crates/es-ir/tests/**
crates/es-policy/src/lower/torch.rs
crates/es-policy/tests/**
crates/es-policy/python/**
python/es/builder.py
crates/es-ir-types/src/codes.rs
crates/es-data/src/lerobot_config.rs
crates/es-policy/src/reference.rs
crates/es-editor/tests/common/mod.rs
crates/es-runtime-embedded/tests/embedded.rs
crates/es/tests/cli.rs
docs/design/learning-lowering.md
docs/design/learning-lowering.ko.md
docs/packets/M8/S2a-activation-squash.md
docs/packets/M8/S2a-activation-squash.ko.md
```

`learning.rs`(두 enum, 필드, canonical, 검증), `es-ir` 안에 그것이 사는 곳의 schema 파일,
`torch.rs`(세 개의 match arm), 테스트, 이 노드들을 적는다면 builder, 노트, 이 패킷.

## 오라클

1. `cargo test -p es-ir committed_learning_hashes_are_unmoved_by_activation_and_squash` — 두
   커밋된 문서가 파싱되고, 그 `learning_hash`들이 `main`에서 고정한 16진 리터럴과 같으며,
   `activation = "Relu"`, `activate_output = false`, `squash = "None"`을 명시적으로 적은 문서가
   그것들을 생략한 것과 같은 해시를 낸다; 기본값이 아닌 값은 해시를 움직인다.
2. `cargo test -p es-ir squash_is_refused_off_a_regression_head` — 그 이름 붙은 진단.
3. `cargo test -p es-policy lower_mlp_activations_source` — 커밋된 그래프의 생성된 소스가
   이전과 바이트 단위로 같다(고정된 `lowering_hash`); `Elu/Swish/Tanh × activate_output ×
   squash`의 각 조합에 대해 소스가 기대되는 `nn.` 표기를 담는다.
4. `cargo test -p es-policy lower_mlp_activations_match_torch -- --ignored`(`ES_PYTHON`,
   서버): 조합마다 15 → [32, 32] → 6 그래프, 무작위 가중치, 무작위 입력 64개 — 로워링된 모듈
   대 `crates/es-policy/python/`의 손으로 쓴 torch 레퍼런스, CPU에서 **비트 단위**.
5. `cargo xtask context-budget`(`es-ir`의 새 코드 줄 수를 보고; ≤ 6,100);
   `cargo xtask ci`; `cargo xtask check-scope docs/packets/M8/S2a-activation-squash.md`.

## 수용 기준

오라클 1–5. `learning-lowering.md`가 세 파라미터, `nn.` 매핑, 해시 규칙을 이름 짓는 짧은 하위
절을 얻는다; 한국어 자매 문서 갱신.

## 금지

커밋된 `learning_hash`나 `lowering_hash`를 움직이는 것; 새 노드(이것은 파라미터 세 개이지,
노드가 아니다); `es-policy/src/lerobot.rs`나 ACT 로워링을 건드리는 것; `docs/ARCHITECTURE*.md`;
`tests/golden/**`; `es-ir` 안의 Python 의존성. INV-17: 새 trait 없음.
