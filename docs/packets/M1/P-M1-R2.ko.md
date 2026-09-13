<!-- Korean translation of docs/packets/M1/P-M1-R2.md. The English file is the working copy; regenerate this when it changes. -->

# P-M1-R2 — 실제 LeRobot ACT 체크포인트가 로드되고 자신의 액션을 재현한다

M1 리뷰 후속 과제(`docs/reviews/M1.md`, P-M1-R2, 게이트 5)를 닫으며, 이와 함께
spec §1.4의 M1 게이트도 닫는다: *"자체 추론 런타임이 LeRobot의 ACT/SmolVLA
체크포인트를 로드해 동일 출력을 내는지가 M1의 게이트다"*. 이 패킷 이전에는
모든 `es-policy` 동등성 테스트가, 자신이 검사하는 것과 같은 lowering이
생성한 합성 체크포인트에 대해 실행되었다 — P-M1-R1이 observation golden에
대해 고쳤던 것과 같은 순환성이다. `Tolerance::TIER4_FP32`는 다른 누군가가
만들어낸 가중치에 적용된 적이 한 번도 없었다.

Spec: spec 1.4, spec 8.4 (PolicyContract), spec 8.7 (lowering), spec 8.9
(tier 4), spec 5.3 (해시 체인), `INV-16`. Design note:
`docs/design/learning-lowering.md`. API digest: `docs/api-notes/lerobot-act.md`
(신규, **검증됨** — 재구성이 아님).

## context (범위)

```
crates/es-policy/src/lerobot.rs            (신규 — ActConfig, remap_act_keys, remap_checkpoint,
                                            lower_act, act_policy)
crates/es-policy/src/lib.rs                (모듈 + re-export)
crates/es-policy/src/lower/torch.rs        (lowering_hash를 추출해 lerobot.rs와 공유)
crates/es-policy/src/torch_runtime.rs      (TorchRuntime::load_lowered; load가 이를 위임)
crates/es-policy/python/act_ref.py         (신규 — LeRobot 쪽 오라클)
crates/es-policy/tests/act_checkpoint.rs   (신규 — 게이트)
docs/api-notes/lerobot-act.md              (신규)
docs/api-notes/lerobot-config.md           (ACT 절: unverified -> verified)
docs/packets/M1/P-M1-R2.md                 (신규)
.gitignore                                 (target/lerobot-cache/)
```

## forbidden (금지)

- `crates/es-ir/**`. 아래의 발견 — spec §8.3의 노드 파라미터가 ACT를 표현할
  수 없다는 것 — 은 보고될 뿐 고쳐지지 않는다: `TemporalEncoder`를 넓히는
  것은 패킷이 아니라 스펙 변경이다.
- `crates/es-data/**`. `lerobot_config.rs`는 config를 layer 10에서 Learning
  IR로 변환한다; `es-policy`는 layer 8이며 이에 의존할 수 없으므로
  (spec §4.2), `ActConfig`는 같은 파일을 독립적으로 읽는다. 둘은 서로에
  대해서가 아니라 하나의 실제 config에 대해 고정되어 있다.
- `crates/es-compile/**`와 Observation IR resize 경로: 이 체크포인트의
  `config.json`은 네이티브 카메라 해상도인 `[3, 480, 640]`을 선언하므로,
  ACT는 어떤 resize도 적용하지 않으며 여기서 `Resize`/`ImageSpec`가 할 일은
  없다(`INV-14`는 건드려지지 않는다).
- 가중치를 커밋하는 것. 197 MiB 체크포인트는 `target/lerobot-cache/` 아래에
  살며 `ES_ACT_CHECKPOINT`를 통해 접근된다.

## spec (사양)

- `ActConfig::parse`는 체크포인트 자체의 `config.json`을 읽는다. 이는
  근사하는 대신 **거부한다**: `act`가 아닌 type, `pre_norm`, `relu`가
  아닌 activation, `n_obs_steps != 1`, non-null인 `temporal_ensemble_coeff`,
  `MEAN_STD`가 아닌 정규화 모드, 하나보다 많은 카메라, 빠진 state나 action
  feature, 홀수인 `dim_model`. 이들 각각은 그렇지 않으면 조용히 잘못된
  action chunk가 될 것이며, spec §8.9는 정확히 이를 막기 위해 존재한다.
- `remap_act_keys(&ActConfig, keys) -> BTreeMap<String, String>`는
  longest-prefix-first로(그래서 `model.encoder.`가
  `model.encoder_latent_input_proj.`를 삼키지 않도록) LeRobot의 이름을
  `nodes.<id>.<param>`로 매핑한다. 항목이 없는 키는 결과에서 빠진다: 56개의
  `model.vae_encoder*` 텐서와 2개의 `normalize_targets.*` 버퍼, 모두
  학습 전용이다.
- `remap_checkpoint`는 safetensors를 **바이트 수준에서** 다시 쓴다 — 헤더가
  재구성되고, 살아남은 각 텐서의 바이트는 그대로 복사되며, `BTreeMap`
  순서를 따른다 — 그래서 어떤 값도 디코드되거나 다시 반올림되지 않으며
  결과의 `blake3`는 안정적인 `WeightsRef::hash`다(spec §3.4, §5.3).
- `lower_act`는 실제 ACT forward pass를 만들어낸다:
  `layer4`에서 잘린 `IntermediateLayerGetter(resnet18(
  norm_layer=FrozenBatchNorm2d))`, 1×1 conv projection, 계산된 2차원
  사인파 카메라 임베딩, `[latent, state, *300개의 이미지 토큰]` 인코더
  시퀀스, `chunk_size`개의 학습된 query에 대한 DETR 디코더, action head,
  그리고 체크포인트 자체의 MEAN_STD 버퍼를 노드 9/10/11로. `use_vae`는
  *생략*을 통해 준수된다: CVAE 분기는 `self.training`으로 게이트되므로
  latent는 0이고 모듈에는 샘플러도 RNG도 없다.
- 네 개의 transformer 레이어 클래스는 `nn.TransformerEncoderLayer`로부터
  빌드되는 대신 손으로 풀어 쓰여 있는데, 정확히 한 가지 이유 때문이다: ACT는
  위치 임베딩을 **query와 key에만** 더하고 value에는 절대 더하지 않는데,
  `torch.nn`의 레이어는 이를 표현할 수 없다. 그것들의 파라미터는 여전히
  `nn.MultiheadAttention`/`nn.Linear`/`nn.LayerNorm`이며, 이것이 접두사 이름
  변경만으로 가중치 리맵 전체가 끝나는 이유다.
- `act_policy`는 config를 spec §8.4의 `PolicyContract`에 투영하며,
  LeRobot 파일의 해시를 `base_model`에, 리맵된 파일의 해시를 `weights`에
  넣어 둘 다 해시 체인에 들어가게 한다.
- `TorchRuntime::load_lowered`는 `lower_to_torch` 밖에서 lowering된 모듈을
  받아, Python이 그 파일을 보기 전에 `load`가 하는 모든 검사 — 선언된
  가중치 해시, 그다음 키와 shape 검증 — 를 수행한다. `PolicyRuntime::load`는
  이제 이에 위임한다. `INV-16`은 건드려지지 않는다: safetensors가 들어가고
  safetensors가 나오며, `act_ref.py`는 그 파일을 `struct`와 `json`으로
  읽는다.

## oracle (오라클)

```
cargo fmt -p es-policy --check
cargo clippy -p es-policy --all-targets -- -D warnings
cargo test -p es-policy
ES_PYTHON=<venv>/Scripts/python.exe ES_ACT_CHECKPOINT=target/lerobot-cache/act_aloha_sim_transfer_cube_human \
  cargo test -p es-policy act -- --nocapture
cargo xtask check-spec-refs
```

체크포인트를 가져온다(절대 커밋되지 않음):

```
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('lerobot/act_aloha_sim_transfer_cube_human', \
    local_dir='target/lerobot-cache/act_aloha_sim_transfer_cube_human')"
```

## acceptance (수용 기준)

- **측정됨, `lerobot` 0.6.1 / `torch` 2.11.0+cpu / `torchvision`
  0.26.0+cpu:**

  ```
  RAN act_checkpoint: lerobot 0.6.1 torch 2.11.0+cpu shape [100, 14] max_abs 0e0 max_rel 0e0
  ```

  전체 `[100, 14]` chunk에 걸쳐 **비트 단위로 동일** — spec §8.9의 tier-4
  `1e-5` 안에 다섯 자릿수의 여유를 두고 들어간다. 테스트는 tier-4 문턱값이
  아니라 `Tolerance::BITWISE`를 단언(assert)하므로, 향후의 발산이 허용오차
  안에 숨을 수 없다. 어떤 모듈도 발산하지 않는다; 문서화할 한계가 없다.
- 비교는 전체 경로를 커버한다: `config.json` → `lower_act` → 키 리맵 →
  바이트 수준 재구성 → 해시 검사 → `validate_keys` → `TorchRuntime`
  서브프로세스 → `compare_actions`, 설치된 `lerobot`의
  `ACTPolicy.predict_action_chunk`에 대해.
- observation은 양쪽이 독립적으로 빌드하는 정수-유도 램프
  (`((i * step + offset) % 4096) / 2^k`)로 고정되어 있어, 양쪽 모두에서
  f32로 정확하며 어떤 텐서도 프로세스 경계를 건너지 않는다. Rust에서
  `torch.Generator`를 재현하거나 파이프로 3.7 MB 이미지를 보내는 것이
  대안이었다.
- `ES_ACT_CHECKPOINT`가 설정되지 않았을 때, `lerobot`이 없을 때, 또는
  `torch`를 가진 Python이 없을 때, 테스트는 조용히 통과하는 대신 이유를
  출력하며 큰 소리로 SKIP한다(spec §1.4). `lerobot.rs`의 단위 테스트
  여덟 개는 Python 없이도 리맵, 키/shape 선언, 결정성, 그리고 모든 거부를
  커버한다.
- `docs/api-notes/lerobot-act.md`는 `unverified` 표시를 전혀 달고 있지
  않다. 그 안의 모든 키 이름, shape, 동작에 관한 주장은 실제 체크포인트에서
  읽혔거나 실제 정책을 실행해 확인되었다; `lerobot-config.md`의 ACT
  절도 같은 읽기로부터 갱신되었으며, 그 "이미지 통계가 픽셀별일 수도
  있다"는 질문은 이제 답해졌다(`[3, 1, 1]`, 채널별).

## findings for later packets

1. **Spec §8.3는 ACT를 표현할 수 없다.** `TemporalEncoder { Transformer }`는
   폭 하나만 실을 뿐 그 이상은 없다 — 레이어 수도, feed-forward 폭도,
   디코더도, latent 차원도 없다 — 그래서 `lower_to_torch`는 `LearningGraph`로부터
   이 모듈을 만들어낼 수 없으며, 아키텍처 파라미터는 대신 체크포인트의
   `config.json`에서 온다. `act_policy`는 컴파일러가 실제로 검사하는
   §8.4 계약에 투영함으로써 그 간극에 다리를 놓지만, "IR이 실제 정책을
   표현한다"는 주장은 현재 *계약* 수준에서는 성립해도 *그래프* 수준에서는
   성립하지 않는다. 노드를 넓히는 것은 스펙의 결정이다.
2. **`lerobot` 0.6.x는 정규화를 정책 밖으로 옮겼다.** `ACTPolicy`는 더 이상
   `normalize_inputs`/`unnormalize_outputs`를 갖지 않는다; 체크포인트의
   여덟 개 통계 버퍼는 *예상치 못한 키*로 로드되어 무시된다. 우리의
   lowering은 이들을 그대로 담는데, 이는 spec §8.7이 원하는 동작이다(IR이
   전/후처리를 소유한다), 하지만 정책이 정규화를 한다고 가정하는 어떤
   `es-data` 코드든 이에 비추어 점검해야 한다.
3. **카메라는 하나뿐이다.** node-id 표는 state, image, action 정규화기를
   위해 9/10/11을 예약하는데, 이는 여분의 image 정규화기에 대한 규칙 없이는
   다중-카메라 ALOHA 체크포인트로 확장되지 않는다. 추측하는 대신 명시적으로
   거부되었다.
4. `es-data`의 평평한 `FeatureStats`는 여전히 `VISUAL` feature의 중첩된
   `[3, 1, 1]` 통계를 파싱하지 못한다(`lerobot-config.md`). 여기서는 범위
   밖이다; 그 로더는 실행되지 않았다.
