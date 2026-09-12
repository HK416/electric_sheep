# API notes

Pinned digests of external API surfaces that agent training data tends to be stale or
wrong on — the failure mode §1.7 calls "환각 API" (hallucinated API). One file per
library, named after it (`ash.md`, `slang.md`, `torch.md`, `lerobot-schema.md`, …).

## Convention

- Every digest names the **exact pinned version** it was taken from (the `Cargo.lock` /
  `pyproject.lock` / submodule commit hash), at the top of the file.
- A digest records only what was actually checked against that version — real type and
  function signatures, real field names — not what looks plausible. If a packet needs a
  signature that isn't here yet, add it (verified against the installed version) before
  writing code against it, not after.
- A digest is refreshed whenever its pinned version bumps. A stale digest is worse than no
  digest: delete or update it, don't leave it silently wrong.
- This file only indexes the convention. No library digest exists yet — none of `ash`,
  Slang, `torch`, or the LeRobot dataset schema is pulled into the workspace yet (Wave 0
  is verification infra only, per `docs/ARCHITECTURE.ko.md` §28.2).

## Planned files

- `ash.md` — Vulkan bindings surface actually used by `es-gpu`/`es-render`.
- `slang.md` — Slang → SPIR-V compiler invocation and reflection API.
- `torch.md` — the `tch`/libtorch C++ surface used by `es-policy` and the Learning IR
  PyTorch reference oracle (§1.4).
- `lerobot-schema.md` — LeRobot dataset and checkpoint schema used by the Observation IR
  and policy-equivalence oracles (§1.4).
