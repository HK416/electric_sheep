"""Electric Sheep authoring frontend (spec 14.2): ``from es import Task, Observation, Learning,
Deployment, math``.

This package is pure Python; the actual node construction happens in the ``es_native`` pyo3
extension (built from ``crates/es-py`` by maturin), which in turn calls the language-neutral
Rust builder core (``crates/es-py/src/builder.rs``). See ``docs/design/python-builder.md``.
"""

from . import math
from .builder import (
    Body,
    Contact,
    Deployment,
    EsDiagnosticError,
    Expr,
    ImageChain,
    Learning,
    Link,
    Observation,
    Pose,
    Robot,
    Sensor,
    StateChain,
    Task,
    stable_id,
)

__all__ = [
    "Task",
    "Observation",
    "Learning",
    "Deployment",
    "math",
    "Expr",
    "Pose",
    "Robot",
    "Link",
    "Body",
    "Sensor",
    "Contact",
    "ImageChain",
    "StateChain",
    "stable_id",
    "EsDiagnosticError",
]
