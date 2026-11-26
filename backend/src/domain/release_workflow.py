"""High-level retrain -> same-holdout evaluation -> atomic release workflow."""

from __future__ import annotations

from typing import Any

from .boosting import shot_features
from .project import analyze
from .records import normalize_rows
from .settings import parse_settings
from .versions import ModelVersion, ReleaseRegistry, evaluate_on_holdout, fingerprint_models
from .boosting import fit_model_pair


def train_version(rows: list[dict], settings: dict | None = None,
                  version: str | None = None) -> ModelVersion:
    options = parse_settings(settings or {})
    records = normalize_rows(rows)
    train = [row for row in records if row["split"] == "train"]
    holdout = [row for row in records if row["split"] == "holdout"]
    pair = fit_model_pair([shot_features(row) for row in train],
                          [row["actual_minutes"] for row in train], options, version=version)
    if version is None:
        version = f"mv-{fingerprint_models(pair)[:12]}"
    scores = evaluate_on_holdout(pair, holdout)
    return ModelVersion(version=version, models=pair, scores=scores, settings=options)


def retrain_and_release(rows: list[dict], settings: dict | None = None,
                        registry: ReleaseRegistry | None = None, *,
                        version: str | None = None) -> dict[str, Any]:
    """Train a candidate, compare on this ledger's holdout, then schedule active."""
    options = parse_settings(settings or {})
    candidate = train_version(rows, options, version)
    registry = registry or ReleaseRegistry()
    holdout = [row for row in normalize_rows(rows) if row["split"] == "holdout"]
    decision = registry.publish(candidate, holdout)
    active = registry.active
    result = analyze(rows, options, model_version=active.version, models=active.models)
    return {"candidate": candidate, "decision": decision, "registry": registry, "result": result}


# Common aliases.
retrain_and_switch = retrain_and_release
train_and_publish = retrain_and_release
train_and_compare = train_version
