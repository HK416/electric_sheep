# Work packets

One file per packet: `docs/packets/<milestone>/<id>.md` with sections
`context`, `spec`, `oracle`, `acceptance`, `forbidden` (spec 1.2).
`cargo xtask check-scope <packet-file>` checks the working-tree diff against `context`.
