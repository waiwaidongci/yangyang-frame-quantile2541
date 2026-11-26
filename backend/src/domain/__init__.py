"""Domain package facade.

The implementation modules remain the stable import surface; this initializer
also exposes common release/version names at :mod:`src.domain`.
"""

from . import boosting as _boosting
from . import versions as _versions

__all__ = [
    "SPEC", "demo_rows", "normalize_rows", "analyze",
    "QuantileModel", "QuantilePair", "fit_quantile", "fit_model_pair", "predict_shot",
    "ModelVersion", "ReleaseRegistry", "ReleaseDecision",
    "ModelRegistry", "ModelVersionRegistry", "ReleaseGate",
    "evaluate_on_holdout", "both_pinballs_not_worse",
]


def __getattr__(name):
    if name in {"SPEC", "demo_rows", "normalize_rows", "analyze"}:
        from . import project
        return getattr(project, name)
    if hasattr(_versions, name):
        return getattr(_versions, name)
    if hasattr(_boosting, name):
        return getattr(_boosting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
