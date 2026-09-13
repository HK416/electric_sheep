<!-- Korean translation of docs/packets/README.md. The English file is the working copy; regenerate this when it changes. -->

# 작업 패킷 (Work packets)

패킷당 파일 하나: `docs/packets/<milestone>/<id>.md`에 `context`, `spec`,
`oracle`, `acceptance`, `forbidden` 섹션을 둔다 (spec 1.2).
`cargo xtask check-scope <packet-file>`이 작업 트리 diff를 `context`와 대조해
검사한다.
