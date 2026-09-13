"""``from es import math as m`` (spec 14.2): free-function spelling of the `Expr` methods
(`m.norm(x)` reads better than `x.norm()` inside a reward expression). Every function here is a
one-line forward to `Expr`; the actual node construction lives in `builder.Expr`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .builder import Expr


def norm(x: "Expr", kind: str = "L2") -> "Expr":
    return x.norm(kind)


def sum(x: "Expr") -> "Expr":  # noqa: A001 - matches spec 14.2's `m.sum`/`.sum()` naming
    return x.sum()
