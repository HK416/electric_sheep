<!-- Korean translation of docs/packets/M4/P-M4-R5.md. The English file is the working copy; regenerate this when it changes. -->

# P-M4-R5 — Slang 캐시 키가 모든 include를 포괄한다

Spec: §2.3(콘텐츠 해시 캐시를 사용하는 오프라인 Slang → SPIR-V), §3.4 item 7(컴파일러도
정체성의 일부다), §11.4(`compile_hash`), §22(랭크 간에 공유되는 캐시).
리뷰 발견 사항: `docs/reviews/M4.md` S-4 — `es-gpu/src/slang.rs:236-256`은 include
디렉터리를 **비재귀적인** `read_dir`로 해싱하면서 읽기 오류를 삼켰다. 그 결과
`sub/helper.slang`으로 include된 헤더는 키에 아무것도 기여하지 않았고, 헤더가 바뀐
뒤에도 오래된 `.spv`가 그대로 제공되었다. 입력을 놓치는 콘텐츠 주소 기반 캐시는 캐시가
없는 것보다 나쁘다: `es task compile`이 패키징하는 번들(§11.4)이 결국 자신을 설명하지
못하는 해시로 명명되기 때문이다.

## context (범위)

```
crates/es-gpu/src/slang.rs
docs/api-notes/slang.md
docs/design/gpu-foundation.md
docs/packets/M4/P-M4-R5.md
```

## spec (사양)

- `SlangCompiler::cache_key`는 `Result<String, GpuError>`를 반환한다. include 디렉터리를
  순회하며 발생하는 모든 오류는 전파된다; 아무것도 삼켜지지 않는다.
- 자유 함수 `hash_include_tree(&mut Hasher, dir, prefix, depth)`가 각 include 디렉터리를
  **재귀적으로** 순회한다. 항목들은 정렬되므로(`Vec<PathBuf>` + `sort`) 키가 `read_dir`
  순서에 의존하지 않는다; 각 파일은 *include 루트에 대한 상대 경로*(`helper.slang`만이
  아니라 `sub/helper.slang`)를 자신의 바이트 뒤에 붙여 키에 기여하므로, 파일을 하위
  디렉터리 사이에서 옮기면 키도 바뀐다.
- `MAX_INCLUDE_DEPTH = 32`: 이보다 깊어지면 스택 오버플로가 아니라
  `io::ErrorKind::InvalidInput`이 된다. 심링크 루프만이 여기에 도달할 현실적인 방법이다.
- `compile`은 `?`로 전파한다; 그 외의 동작 변화는 없다.

## oracle (오라클)

```
cargo test -p es-gpu slang
```

테스트 두 개:

- `cache_key_covers_files_in_include_subdirectories` — `slangc`가 필요 없다:
  `sub/helper.slang`을 담은 임시 include 루트에서 키를 구하고, helper를 다시 쓴 뒤 키를
  다시 구하면 두 키가 다르다; 그 후 루트를 삭제하면 `cache_key`는 `Err`를 반환한다.
- `editing_an_included_subdirectory_file_recompiles` — `slangc`를 거치는 엔드투엔드
  테스트다(없으면 SKIP, 사양 §1.4): `#include "sub/helper.slang"`을 쓰는 커널을
  컴파일하고, 다시 컴파일하며(캐시 히트, `invocations() == 1`), helper를 수정한 뒤 세
  번째로 컴파일한다 — `invocations() == 2`이며, 해시와 SPIR-V 워드 모두 첫 컴파일과
  달라진다.

## acceptance (수용 기준)

- include *하위 디렉터리*의 파일을 수정하면 캐시 키가 바뀌고 실제 `slangc` 호출이
  일어난다. 관찰값: 컴파일 3회에 대해 호출 2회.
- 읽을 수 없거나 존재하지 않는 include 디렉터리가 있으면 `compile`은 키를 만들어내는
  대신 실패한다.
- 입력이 바뀌지 않은 경우는 여전히 캐시에 히트한다: `cache_hit_does_not_start_slangc`는
  손대지 않았으며 여전히 컴파일 2회에 대해 호출 1회를 보고한다.
- `docs/api-notes/slang.md`와 `docs/design/gpu-foundation.md`는 재귀적 순회와 오류
  전파를 설명한다.

## forbidden (금지)

- 순회를 위한 새 의존성 없음(`read_dir`에 재귀만 더하면 되므로 `walkdir` 등은 필요
  없다).
- 호출 간에 include 해시를 캐싱하지 않는다: 키의 요점은 지금 디스크에 있는 것으로부터
  다시 계산된다는 데 있다.
- `P-M4-R0-slang-scratch`가 확립한 캐시 레이아웃, 파일 명명, 동시 발행(concurrent-publish)
  로직에는 변경이 없다.
- `spirv.rs`, `buffer.rs`, `xtask`에는 손대지 않는다 — 이들은 R6와 R7의 몫이다.
