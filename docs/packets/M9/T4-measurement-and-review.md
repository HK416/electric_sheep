# M9 T4 — the measurement is read, and the review closes plan T

Spec: §28.12 wave 3, §13.4 (whether the RL default becomes the increment), §28.9 rule 2. Type A.

## spec

`docs/reviews/M9.md` + `.ko.md` in the M8 format: gate results, what plan T set out to do and
did (R1's noise table, R6's line count, T1's bitwise integration, T2's delta import, T3's
side-by-side), findings, human decisions (the §13.4 default; `EeDelta`'s IK; the two
`ActionSpace` enums; the still-open M8 decisions S-1, S-2 and the next source policy's scene),
ladder status, verdict, follow-ups. §28.12 gains its "M9 result" paragraph (ko + en, one commit);
`CLAUDE.md`'s status paragraph moves.

## context

```
docs/reviews/M9.md
docs/reviews/M9.ko.md
docs/ARCHITECTURE.ko.md
docs/ARCHITECTURE.md
CLAUDE.md
docs/packets/M9/T4-measurement-and-review.md
docs/packets/M9/T4-measurement-and-review.ko.md
```

## oracle

`cargo xtask ci` green on the reviewed commit; `check-spec-refs`.

## acceptance

The review exists with every number tagged measured or `unverified`; the orchestrator writes it.

## forbidden

Numbers without hashes; a verdict the tables do not support.
