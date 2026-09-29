<!-- Korean translation of docs/packets/M12/P-M12-R7-hint-u3.md. The English file is the working copy; regenerate this when it changes. -->
# M12 R7 — 힌트 템플릿은 통과했던 조합(U3)을 학습한다

소유자 결정, 2026-09-29: 연습 카드는 초보자에게 성공할 수 있는 정책을 주어야 한다.
지금 카드의 레시피는 `cycle.toml` → `learning.toml` + `observation.toml` 위의 `training.toml`이고,
그대로는 한 번도 측정되지 않았다. 같은 문서의 U0는 0.25를 받았고, 이 PC에서의 첫 실제 실행(short
프리셋)은 nominal 0/16이었으며 plane이 스텝의 99.6 %를 바꿨다(`docs/packets/M12/YV-verification.md`).
M7/U3 — `learning-pretrained.toml`(ImageNet ResNet18 백본) + `observation-augmented.toml`,
`training-u3.toml`의 실행 설정 — 은 데모의 acceptance를 통과했다(held-out 0.5625,
`docs/reviews/M7.md`, `docs/design/visible-learning.md` 7.31절).

## spec

- 새로 커밋하는 cycle 레시피(예: `tests/fixtures/visible-learning/cycle-hint-u3.toml`). 그
  `[train]`은 U3의 실행 설정을 인라인으로 담거나, 저장소 기준 상대 경로를 쓰는 새 training 레시피를
  가리킨다(커밋된 `training-u3.toml`은 서버 경로를 담고 있으며 그대로 둔다). 그 `[eval] config`는
  U3 번들이 `XIR-040`을 통과하는 Evaluation IR이다 — U3가 어떻게 평가되었는지 찾아라(증강 노드는
  학습 전용이다. 평가된 번들은 `observation-augmented.toml`을 실었나, `observation.toml`을 실었나?)
  그리고 정확히 그대로 따른다. 새 Evaluation IR을 커밋해야 한다면 그 문서는 `evaluation.toml`의
  suite, seed, metric, threshold를 **하나도** 바꾸지 않는다.
- 사전학습 백본: 새 머신에서 `base_model`을 어떻게 얻는가(`python/es/fetch_backbone.py` 또는
  문서화된 절차). 편집기의 실행은 이 PC의 새 체크아웃에서 동작해야 한다. 다운로드가 필요하면
  레시피/CLI 경로가 그것을 하거나, 템플릿의 `needs`가 무엇이 없는지 말한다.
- `templates/cube-into-bin-hint.toml`은 새 cycle을 가리킨다. `[bundle]`은 U3 문서를 적는다(그래서
  `es policy init`이 U3의 학습 전 번들을 만든다). 프리셋은 "5,000 / 20,000 / 60,000 스텝"을 유지하고
  IR 경로의 mark 규칙을 지킨다. `medium`은 U3의 20,000과 같다.
- `--resident-gpu`(U3의 `extra`): 데모 200개로 12 GB GPU에 들어갈 때만 유지한다(측정하라). 아니면
  빼고 레시피 헤더에 이유를 적는다.
- `plan-cycle-vision.txt`처럼 새 cycle의 dry-run golden을 그 생성기로 만든다.

## context

```
tests/fixtures/visible-learning/**
tests/golden/train/**
templates/cube-into-bin-hint.toml
crates/es/tests/cli.rs
crates/es-editor/src/model/template.rs
crates/es-editor-model/src/model/template.rs
python/es/fetch_backbone.py
docs/packets/M12/P-M12-R7-hint-u3.md
docs/packets/M12/P-M12-R7-hint-u3.ko.md
```

## oracle

1. `es loop cycle --recipe <새 cycle> --dry-run`이 새 golden과 일치한다(생성기로 만들고 검토한 것).
2. 템플릿의 `[bundle]`로 `es policy init` → 새 cycle의 `[eval] config`에 대해 게이트의 검사
   (Y5b의 테스트가 하는 검사)를 통과하는 번들, Python 없이.
3. 템플릿 테스트(`the_committed_templates_parse_and_name_real_files` 또는 그 후속)가 통과한다.
4. 이 패킷 안에서는 학습을 돌리지 않는다: 그동안 같은 PC에서 다른 패킷들이 컴파일한다(리뷰 S-6).
   short와 medium 종단 간 실행은 병합 뒤 조용한 PC에서 오케스트레이터가 돌린다.

## forbidden

`evaluation.toml`, 어떤 acceptance threshold, Safety Plane envelope(S-7은 소유자의 것),
`training-u3.toml`, 그 밖의 커밋된 문서를 제자리에서 바꾸는 것; 원격 서버 접속.
