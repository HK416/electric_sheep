<!-- Korean translation of docs/packets/M12/P-M12-R8-backbone-fetch.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R8 — 학습 단계는 없는 고정 백본을 받아 온다

P-M12-R7(커밋 `e536c72`)이 찾은 문제: 힌트 카드의 레시피 `training-hint-u3.toml`은
`target/backbone/resnet18-imagenet1k-v1.safetensors`를 가리키는데, 새로 받은 체크아웃에는 이 파일이
없다. 받아 오는 것도 없었고 `needs`로 말할 수도 없었다(`es --check-deps`가 보고하지 않는 이름은
없는 것으로 치고 카드가 꺼진다). 그래서 사이클은 26분을 수집하고 전문가 게이트를 통과한 뒤에야
학습 단계가 `Backbone::verify`에서 멈췄다. 초보자가 이걸 겪어서는 안 된다.

## 스펙

- `[policy] base_model_fetch = "resnet18"` — 선택 필드, 가장 작은 모양이다. `fetch_backbone.py`의
  `--arch`이고 그 이상은 없다. `--out`은 `base_model`의 디렉터리, `--expect`는
  `RESNET18_IMAGENET1K_V1_BLAKE3`이라 레시피는 고정값의 두 번째 사본을 갖지 않는다. 이름으로
  거부되는 경우: `base_model`이 없을 때, `resnet18`이 아닌 arch일 때(고정값이 하나라 arch도 하나),
  `base_model`의 파일 이름이 스크립트가 쓰는 `<arch>-imagenet1k-v1.safetensors`가 아닐 때
  (`Recipe::fetch_args`. `Recipe::parse`, `Cycle::training`, `Plan::build`가 부른다).
- `Plan.fetch`: 명령 줄 전체 — 실행의 인터프리터(`ES_PYTHON`이 먼저), `python/es/fetch_backbone.py`,
  플래그. `# fetch:` 주석 줄로 렌더링된다(파일이 없을 때만 돌기 때문이다). `es train`의 계획에서는
  `# route:` 다음, 사이클에서는 모든 단계 위(중첩된 학습 계획에서 끌어올려 한 번만 찍는다). 필드가
  없으면 모든 계획, 골든, `identity_hash`가 그대로이고, 있으면 다른 필드처럼 `config.json`의 레시피
  안에 들어간다.
- `es train`: `base_model`이 없으면 백본 검사 전에 받아 오고, 그 다음은 이전처럼
  `Backbone::verify`가 파일을 판정한다. `es loop cycle`: 첫 단계 전에(`--from eval|showcase`가
  아니면) 파일이 없을 때 같은 받아 오기를 돌리고, 이름이 붙은 `base_model`을 `Backbone::verify`로
  확인한다. 받아 오기가 실패하거나 파일이 틀리면 사이클은 수집 전에 멈춘다. 실패한 받아 오기는
  파일, 종료 코드, 명령, 인터프리터에 필요한 것을 이름으로 대는 평이한 한 문장이다.
- 새 텔레메트리 단계는 없으므로 에디터는 바뀌지 않는다. 픽스처와 템플릿 머리말은 "`needs`로 말할
  수 없다" 대신 이제 무슨 일이 일어나는지를 적는다.

## context

```
crates/es-data/src/training.rs
crates/es/src/cmd/train.rs
crates/es/src/cmd/cycle.rs
crates/es/tests/cli.rs
tests/fixtures/visible-learning/training-hint-u3.toml
templates/cube-into-bin-hint.toml
tests/golden/train/plan-cycle-hint-u3.txt
docs/design/training-recipe.md
docs/design/training-recipe.ko.md
docs/packets/M12/P-M12-R8-backbone-fetch.md
docs/packets/M12/P-M12-R8-backbone-fetch.ko.md
```

## oracle

1. `cargo test -p es-data training::tests::base_model_fetch_is_one_line_above_collect`: 필드는
   `# fetch:` 줄 하나이고 계획의 다른 것은 움직이지 않는다. 사이클은 그것을 수집 위에 한 번 찍는다.
   거부 세 가지.
2. `cargo test -p es --test cli cycle_fetches_the_backbone_before_collect`: 대역 인터프리터(`.cmd` /
   `sh` 스크립트, Python 없음)가 서로는 맞지만 고정값과는 다른 아티팩트를 `--out`에 복사하면,
   사이클은 받아 온 뒤 고정값으로 거부하고 어떤 단계도 시작하지 않는다. 이미 있는 파일은 다시 받지
   않는다. 실패하는 대역은 평이한 메시지를 낸다. `es train`은 백본 검사 전에 받아 온다.
3. `plan-cycle-hint-u3.txt`는 `generate_cycle_hint_u3_golden`으로 다시 만들었다. 한 줄이 늘었고,
   `# cycle:` 머리 아래의 `# fetch:` 줄이다. 다른 계획 골든은 그대로다.
4. fmt, `cargo clippy -p es -p es-data --all-targets -- -D warnings`, `cargo test -p es-data`,
   `cargo test -p es --test cli -- cycle_ train_`, `cargo xtask verify-goldens`,
   `cargo xtask check-spec-refs`, `cargo xtask check-scope <이 파일> --base main`.

## forbidden

고정값이나 `fetch_backbone.py`의 동작 변경, 새 텔레메트리 단계, 다른 레시피·계획 골든·
`identity_hash`의 변화, 학습·평가 실행, 원격 서버 접속.
