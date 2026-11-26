"""Database-backed atomic model release operations.

The pure policy lives in :mod:`src.domain.versions`.  This module adds a
singleton lock row so separate worker/API processes serialize releases.
"""

from __future__ import annotations

import threading
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.domain.boosting import fit_model_pair, shot_features
from src.domain.records import normalize_rows
from src.domain.settings import parse_settings
from src.domain.versions import (
    both_pinballs_not_worse,
    evaluate_on_holdout,
    models_from_data,
    models_to_data,
)
from src.models.model_version import ModelReleaseState, ModelVersionRecord
from src.utils.time import utc_stamp

_release_lock = threading.RLock()


def ensure_state(session: Session) -> ModelReleaseState:
    state = session.get(ModelReleaseState, 1)
    if state is None:
        state = ModelReleaseState(id=1, active_version_id=None, updated_at=utc_stamp())
        session.add(state)
        session.flush()
    return state


def locked_state(session: Session) -> ModelReleaseState:
    ensure_state(session)
    statement = select(ModelReleaseState).where(ModelReleaseState.id == 1).with_for_update()
    state = session.scalar(statement)
    if state is None:  # pragma: no cover - defensive for unusual isolation behavior
        state = ensure_state(session)
    return state


def get_active_record(session: Session, *, lock: bool = False) -> ModelVersionRecord | None:
    statement = select(ModelVersionRecord).where(ModelVersionRecord.is_active.is_(True))
    if lock:
        statement = statement.with_for_update()
    return session.scalar(statement.order_by(ModelVersionRecord.id.desc()).limit(1))


def train_candidate(rows: list[dict], settings: dict | None = None, version: str | None = None):
    """Fit one immutable P50/P90 pair from the train split of a ledger."""
    records = normalize_rows(rows)
    train = [row for row in records if row["split"] == "train"]
    features = [shot_features(row) for row in train]
    actual = [row["actual_minutes"] for row in train]
    options = parse_settings(settings or {})
    return fit_model_pair(features, actual, options, version=version)


def publish_model(
    session: Session,
    *,
    version: str,
    models: Any,
    holdout_rows: list[dict],
    dataset_id: int | None = None,
    settings: dict | None = None,
    commit: bool = False,
) -> tuple[ModelVersionRecord, dict]:
    """Compare candidate with the switch-time active model and swap atomically."""
    with _release_lock:
        state = locked_state(session)
        active_record = None
        if state.active_version_id is not None:
            active_record = session.scalar(
                select(ModelVersionRecord)
                .where(ModelVersionRecord.id == state.active_version_id)
                .with_for_update()
            )
        if active_record is None:
            active_record = get_active_record(session, lock=True)

        candidate_scores = evaluate_on_holdout(models, holdout_rows)
        active_scores = None
        accepted = True
        reason = "initial_active"

        if active_record is not None:
            active_models = models_from_data(active_record.artifact)
            # Crucially this is scored on the exact same holdout_rows list.
            active_scores = evaluate_on_holdout(active_models, holdout_rows)
            accepted = both_pinballs_not_worse(candidate_scores, active_scores)
            reason = "pinball_not_worse_switched" if accepted else "pinball_worse_retained_active"

        record = ModelVersionRecord(
            version=version,
            status="active" if accepted else "rejected",
            is_active=accepted,
            artifact=models_to_data(models),
            p50_pinball=candidate_scores[50],
            p90_pinball=candidate_scores[90],
            dataset_id=dataset_id,
            settings=settings,
            message=reason,
            created_at=utc_stamp(),
        )
        session.add(record)
        session.flush()

        if accepted:
            if active_record is not None:
                active_record.is_active = False
                active_record.status = "retired"
            state.active_version_id = record.id
            state.updated_at = record.created_at

        decision = {
            "switched": accepted,
            "accepted": accepted,
            "reason": reason,
            "candidate_version": version,
            "active_version": version if accepted else active_record.version if active_record else None,
            "candidate_scores": candidate_scores,
            "active_scores": active_scores,
        }
        if commit:
            session.commit()
        return record, decision


# Aliases matching common release vocabulary.
compare_and_switch = publish_model
release_model = publish_model


class ModelReleaseService:
    """Small object facade for callers that prefer a service instance."""

    def __init__(self, session: Session):
        self.session = session

    def get_active(self, *, lock: bool = False):
        return get_active_record(self.session, lock=lock)

    def publish(self, **kwargs):
        return publish_model(self.session, **kwargs)

    compare_and_switch = publish
    release = publish
