"""Version-stamped scheduling facade.

``shift_queue`` already performs single-station sequencing.  This module makes
the one-version invariant explicit for callers that pass a trained release.
"""

from __future__ import annotations

from typing import Any, Sequence

from .boosting import QuantilePair, predict_shot
from .reports import shift_queue
from .versions import extract_models, version_identifier


def schedule_shots(shots: Sequence[dict], model: Any = None, *, models=None,
                   model_version: str | None = None, version: str | None = None,
                   shift_minutes: int = 240) -> dict:
    selected = models if models is not None else model
    version_id = model_version or version or version_identifier(selected)
    if isinstance(selected, QuantilePair) and model_version is None and version is None:
        version_id = selected.version
    if selected is not None:
        selected_models = extract_models(selected)
        predicted = [{**predict_shot(dict(shot), selected_models), "model_version": version_id} for shot in shots]
    else:
        predicted = list(shots)
    table, scheduled, reserved = shift_queue(predicted, shift_minutes, version_id)
    table["model_version"] = version_id or table.get("model_version")
    return {
        "table": table,
        "scheduled": scheduled,
        "reserved_minutes": reserved,
        "model_version": table["model_version"],
    }


# Verb/noun aliases used by different call sites.
schedule = schedule_shots
build_shift_queue = schedule_shots
create_schedule = schedule_shots
