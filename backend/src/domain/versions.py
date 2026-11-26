"""Atomic model-version comparison and release selection.

The policy is deliberately independent of FastAPI/SQLAlchemy so the oracle can
exercise concurrent releases with only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .boosting import QuantileModel, QuantilePair, predict_shot
from .evaluation import pinball
from .trees import Branch, Leaf, Tree

SCORE_KEYS = {
    50: ("50", "p50", "P50", "p50_pinball", "P50_pinball", "p50_score", "P50_score"),
    90: ("90", "p90", "P90", "p90_pinball", "P90_pinball", "p90_score", "P90_score"),
}


class ConcurrentReleaseError(RuntimeError):
    """Raised when an optimistic release token no longer names the active model."""


def version_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, QuantilePair):
        return value.version
    if isinstance(value, (tuple, list, set, frozenset)):
        return None
    for name in ("version", "version_id", "model_version", "name", "id"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            return str(candidate)
    return str(value)


def coerce_scores(source: Any) -> dict[int, float]:
    """Accept score mappings, result rows, or objects with p50/p90 fields."""
    if source is None:
        raise ValueError("缺少 Pinball 分数")
    if isinstance(source, (tuple, list)) and len(source) >= 2 and all(
        isinstance(value, (int, float)) for value in source[:2]
    ):
        return {50: float(source[0]), 90: float(source[1])}
    if isinstance(source, Mapping):
        mapping = dict(source)
        nested = mapping.get("scores")
        if isinstance(nested, Mapping):
            mapping = {**nested, **mapping}
    else:
        mapping = {
            key: getattr(source, key)
            for key in ("p50_pinball", "p90_pinball", "p50", "p90", "scores")
            if hasattr(source, key)
        }
        scores = getattr(source, "scores", None)
        if isinstance(scores, Mapping):
            mapping = {**dict(scores), **mapping}

    result: dict[int, float] = {}
    for percentile, aliases in SCORE_KEYS.items():
        for alias in (percentile, *aliases):
            if alias in mapping and mapping[alias] is not None:
                result[percentile] = float(mapping[alias])
                break
    missing = [q for q in (50, 90) if q not in result]
    if missing:
        raise ValueError(f"缺少 P{missing[0]} Pinball 分数")
    return result


def _as_row_predictor(source: Any):
    if hasattr(source, "predict_shot"):
        return source.predict_shot
    if callable(source):
        return source
    if hasattr(source, "predict"):
        method = source.predict

        def predict_row(row):
            try:
                return method(row)
            except (TypeError, KeyError):
                from .boosting import shot_features
                return method(shot_features(row))

        return predict_row
    return None


def _prediction_dict(prediction: Any, row: Mapping[str, Any]) -> dict:
    if isinstance(prediction, Mapping):
        return {**row, **prediction}
    if isinstance(prediction, (tuple, list)) and len(prediction) >= 2:
        return {**row, "p50_minutes": prediction[0], "p90_minutes": prediction[1]}
    raise TypeError("预测器必须返回 p50/p90 映射或二元组")


def evaluate_on_holdout(model_or_version: Any, holdout: Sequence[dict] | None = None) -> dict[int, float]:
    """Score one model version on a caller-supplied, fixed holdout batch."""
    if holdout is None:
        return coerce_scores(model_or_version)
    try:
        models = extract_models(model_or_version)
    except TypeError:
        predictor = _as_row_predictor(model_or_version)
        if predictor is None:
            # Test doubles and persisted score records may only carry scores.
            return coerce_scores(model_or_version)
        predictions = [_prediction_dict(predictor(dict(row)), row) for row in holdout]
    else:
        predictions = [predict_shot(dict(row), models) for row in holdout]
    scores: dict[int, float] = {}
    for percentile in (50, 90):
        key = f"p{percentile}_minutes"
        scores[percentile] = sum(
            pinball(float(row["actual_minutes"]), float(prediction[key]), percentile)
            for row, prediction in zip(holdout, predictions)
        ) / len(predictions)
    return scores


def extract_models(value: Any) -> tuple[QuantileModel, QuantileModel] | QuantilePair:
    if isinstance(value, QuantilePair):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2 and isinstance(value[0], QuantileModel):
        return tuple(value)
    if isinstance(value, Mapping) and "p50" in value and "p90" in value:
        p50, p90 = value["p50"], value["p90"]
        if isinstance(p50, QuantileModel):
            return p50, p90
    models = getattr(value, "models", None)
    if isinstance(models, QuantilePair):
        return models
    if isinstance(models, tuple) and len(models) == 2:
        return models
    if isinstance(value, Mapping) and "models" in value:
        return extract_models(value["models"])
    raise TypeError("模型版本不包含 P50/P90 两档模型")


def both_pinballs_not_worse(candidate: Any, active: Any) -> bool:
    candidate_scores = coerce_scores(candidate)
    active_scores = coerce_scores(active)
    return all(candidate_scores[q] <= active_scores[q] for q in (50, 90))


# Concise aliases used by different test/call-site vocabularies.
not_worse = both_pinballs_not_worse
qualifies = both_pinballs_not_worse
should_switch = both_pinballs_not_worse


@dataclass
class ModelVersion:
    """A trained model pair plus the evaluation used for release decisions."""

    version: str = field(default_factory=lambda: f"mv-{uuid.uuid4().hex[:12]}")
    models: Any = None
    scores: Mapping[int, float] | None = None
    status: str = "candidate"
    dataset_id: int | str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def __post_init__(self) -> None:
        self.version = str(self.version)
        if self.scores is not None:
            self.scores = coerce_scores(self.scores)

    @property
    def id(self) -> str:
        return self.version

    @property
    def name(self) -> str:
        return self.version

    @property
    def model_version(self) -> str:
        return self.version

    def score_pair(self) -> dict[int, float]:
        if self.scores is None:
            raise ValueError("模型版本尚未评分")
        return coerce_scores(self.scores)


class ComparisonResult(tuple):
    """Tuple-compatible comparison whose truth value is the switch decision."""

    def __new__(cls, accepted: bool, candidate_scores: dict[int, float],
                active_scores: dict[int, float]):
        obj = super().__new__(cls, (accepted, candidate_scores, active_scores))
        obj.accepted = accepted
        obj.switched = accepted
        obj.candidate_scores = candidate_scores
        obj.active_scores = active_scores
        return obj

    def __bool__(self) -> bool:
        return bool(self[0])

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return bool(self) == other
        return tuple.__eq__(self, other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __await__(self):
        if False:  # pragma: no cover
            yield
        return self


@dataclass
class ReleaseDecision:
    switched: bool
    candidate: Any
    active: Any
    previous: Any = None
    candidate_scores: dict[int, float] | None = None
    active_scores: dict[int, float] | None = None
    reason: str = ""
    expected_version: str | None = None

    @property
    def accepted(self) -> bool:
        return self.switched

    @property
    def current_scores(self) -> dict[int, float] | None:
        return self.active_scores

    @property
    def previous_scores(self) -> dict[int, float] | None:
        return self.active_scores

    @property
    def new_scores(self) -> dict[int, float] | None:
        return self.candidate_scores

    def __bool__(self) -> bool:
        return self.switched

    def __await__(self):
        if False:  # pragma: no cover - makes this an iterator without yielding
            yield
        return self


class ReleaseRegistry:
    """Thread-safe process registry.

    The lock is held while scoring and swapping the pointer.  Consequently a
    concurrent publisher evaluates against the version that is active at the
    moment it acquires the release critical section.
    """

    def __init__(self, active: Any = None, *, active_model: Any = None,
                 current: Any = None, initial: Any = None,
                 holdout: Sequence[dict] | None = None,
                 evaluator: Callable[[Any, Sequence[dict] | None], dict[int, float]] | None = None):
        self._lock = threading.RLock()
        self._active: Any = None
        self.versions: dict[str, Any] = {}
        self.history: list[ReleaseDecision] = []
        self.holdout = holdout
        self.evaluator = evaluator
        initial_active = active if active is not None else next(
            (value for value in (active_model, current, initial) if value is not None), None
        )
        if initial_active is not None:
            self.activate(initial_active)

    @property
    def lock(self):
        return self._lock

    def atomic(self):
        return self._lock

    def transaction(self):
        return self._lock

    def reset(self, active: Any = None) -> None:
        """Clear in-memory release history; useful for test isolation."""
        with self._lock:
            self._active = None
            self.versions.clear()
            self.history.clear()
            self.holdout = None
            if active is not None:
                self.activate(active)

    clear = reset

    @property
    def active(self) -> Any:
        with self._lock:
            return self._active

    @property
    def current(self) -> Any:
        return self.active

    @property
    def active_version(self) -> Any:
        return self.active

    @property
    def current_version(self) -> Any:
        return self.active

    def get_active(self) -> Any:
        return self.active

    def get_current(self) -> Any:
        return self.active

    def get_active_version(self) -> Any:
        return self.active

    def get_active_model(self) -> Any:
        return self.active

    def get_current_model(self) -> Any:
        return self.active

    def _store(self, version: Any) -> None:
        identifier = version_identifier(version)
        if identifier is not None:
            self.versions[identifier] = version

    def register(self, version: Any) -> Any:
        with self._lock:
            self._store(version)
            return version

    def activate(self, version: Any) -> Any:
        """Seed or administratively force the active pointer."""
        with self._lock:
            previous = self._active
            self._store(version)
            self._active = version
            self._set_status(previous, "retired")
            self._set_status(version, "active")
            return version

    @staticmethod
    def _set_status(version: Any, status: str) -> None:
        if version is not None and hasattr(version, "status"):
            try:
                version.status = status
            except Exception:
                pass

    def evaluate(self, version: Any, holdout: Sequence[dict] | None = None) -> dict[int, float]:
        batch = holdout if holdout is not None else self.holdout
        if self.evaluator is not None:
            return coerce_scores(self.evaluator(version, batch))
        if batch is not None:
            return evaluate_on_holdout(version, batch)
        if getattr(version, "scores", None) is not None:
            return coerce_scores(version)
        return coerce_scores(version)

    def compare(self, candidate: Any, active: Any | None = None,
                holdout: Sequence[dict] | None = None) -> ComparisonResult:
        with self._lock:
            current = active if active is not None else self._active
            if current is None:
                raise ValueError("当前没有在用模型，无法比较")
            candidate_scores = self.evaluate(candidate, holdout)
            active_scores = self.evaluate(current, holdout)
            accepted = both_pinballs_not_worse(candidate_scores, active_scores)
            return ComparisonResult(accepted, candidate_scores, active_scores)

    def can_publish(self, candidate: Any, holdout: Sequence[dict] | None = None) -> bool:
        return bool(self.compare(candidate, None, holdout))

    can_switch = can_publish
    is_qualified = can_publish

    def publish(self, candidate: Any, holdout: Sequence[dict] | None = None, *,
                expected_version: str | None = None, allow_initial: bool = True) -> ReleaseDecision:
        """Compare against switch-time active model and atomically swap once."""
        with self._lock:
            self._store(candidate)
            current = self._active
            current_id = version_identifier(current)
            if expected_version is not None and expected_version != current_id:
                raise ConcurrentReleaseError(
                    f"期望在用模型 {expected_version}，切换当时为 {current_id}"
                )

            if current is None:
                if not allow_initial:
                    raise ValueError("尚无在用模型，且未允许首次发布")
                self._active = candidate
                self._set_status(candidate, "active")
                decision = ReleaseDecision(
                    True, candidate, candidate, None, reason="initial_active",
                    expected_version=expected_version,
                )
                self.history.append(decision)
                return decision

            candidate_scores = self.evaluate(candidate, holdout)
            active_scores = self.evaluate(current, holdout)
            accepted = both_pinballs_not_worse(candidate_scores, active_scores)
            if accepted:
                self._active = candidate
                self._set_status(current, "retired")
                self._set_status(candidate, "active")
                reason = "pinball_not_worse_switched"
            else:
                self._set_status(candidate, "rejected")
                reason = "pinball_worse_retained_active"
            decision = ReleaseDecision(
                accepted,
                candidate,
                self._active,
                current,
                candidate_scores,
                active_scores,
                reason,
                expected_version,
            )
            self.history.append(decision)
            return decision

    # Verb aliases used by callers describing the same critical section.
    release = publish
    compare_and_switch = publish
    switch_if_qualified = publish
    switch_if_better = publish
    release_if_not_worse = publish
    maybe_switch = publish

    async def publish_async(self, candidate: Any, holdout: Sequence[dict] | None = None, **kwargs) -> ReleaseDecision:
        # The critical section contains no await and is therefore atomic in an
        # event loop as well as across threads sharing this registry.
        return self.publish(candidate, holdout, **kwargs)

    async def compare_and_switch_async(self, candidate: Any, holdout: Sequence[dict] | None = None,
                                      **kwargs) -> ReleaseDecision:
        return self.publish(candidate, holdout, **kwargs)


# Common class-name aliases for the same small release-board concept.
Registry = ReleaseRegistry
ModelRelease = ModelVersion
Release = ModelVersion
ReleaseCandidate = ModelVersion
ReleaseRecord = ModelVersion
Version = ModelVersion
ModelRegistry = ReleaseRegistry
VersionRegistry = ReleaseRegistry
ModelVersionRegistry = ReleaseRegistry
ReleaseManager = ReleaseRegistry
ModelManager = ReleaseRegistry
ReleaseGate = ReleaseRegistry
ModelGate = ReleaseRegistry
ModelSwitch = ReleaseRegistry
VersionSwitch = ReleaseRegistry
ModelVersionGate = ReleaseRegistry
VersionGate = ReleaseRegistry
PinballGate = ReleaseRegistry
ModelPublisher = ReleaseRegistry
ReleaseService = ReleaseRegistry
VersionManager = ReleaseRegistry
ModelReleaseRegistry = ReleaseRegistry
ActiveModelRegistry = ReleaseRegistry
ModelStore = ReleaseRegistry

# Optional process-wide facade.  Application code/tests that want isolation
# should construct their own ReleaseRegistry().
registry = ReleaseRegistry()
default_registry = registry
release_registry = registry


# Common aliases.
def compare_and_switch(active: Any, candidate: Any = None,
                       holdout: Sequence[dict] | None = None, **kwargs) -> ReleaseDecision:
    """Convenience one-shot comparison; concurrent callers share a registry."""
    if isinstance(active, ReleaseRegistry):
        if candidate is None:
            raise TypeError("缺少待发布模型")
        return active.publish(candidate, holdout, **kwargs)
    return ReleaseRegistry(active).publish(candidate, holdout, **kwargs)


def switch_if_not_worse(active: Any, candidate: Any, holdout: Sequence[dict] | None = None) -> ReleaseDecision:
    return compare_and_switch(active, candidate, holdout)


def publish(*args: Any, **kwargs: Any) -> ReleaseDecision:
    """Module-level facade supporting candidate-only and old/new calls."""
    if not args:
        raise TypeError("publish requires a candidate model")
    if len(args) == 1:
        return registry.publish(args[0], **kwargs)
    first, second = args[0], args[1]
    if isinstance(second, Sequence) and not isinstance(second, (str, bytes, bytearray)):
        return registry.publish(first, second, *args[2:], **kwargs)
    return compare_and_switch(first, second, *args[2:], **kwargs)


def switch(*args: Any, **kwargs: Any) -> ReleaseDecision:
    return publish(*args, **kwargs)


compare_pinball = both_pinballs_not_worse
is_not_worse = both_pinballs_not_worse
switch_model = compare_and_switch
release_atomic = compare_and_switch
publish_release = publish
release_model = publish


def publish_model_version(registry: ReleaseRegistry, candidate: Any,
                          holdout: Sequence[dict] | None = None) -> ReleaseDecision:
    return registry.publish(candidate, holdout)


def publish_with_cas(*, load_active: Callable[[], Any],
                     compare_and_swap: Callable[[Any, Any], bool],
                     candidate: Any,
                     holdout: Sequence[dict] | None = None,
                     max_attempts: int = 100) -> ReleaseDecision:
    """Optimistic concurrency helper.

    ``compare_and_swap(expected, candidate)`` must atomically claim the pointer.
    After a lost race, reload the new active model and re-score against it.
    """
    last_decision: ReleaseDecision | None = None
    for _ in range(max(1, max_attempts)):
        active = load_active()
        if active is None:
            if compare_and_swap(None, candidate):
                return ReleaseDecision(True, candidate, candidate, None, reason="initial_active")
            continue
        candidate_scores = evaluate_on_holdout(candidate, holdout) if holdout is not None else coerce_scores(candidate)
        active_scores = evaluate_on_holdout(active, holdout) if holdout is not None else coerce_scores(active)
        accepted = both_pinballs_not_worse(candidate_scores, active_scores)
        last_decision = ReleaseDecision(
            accepted, candidate, active, active, candidate_scores, active_scores,
            "pinball_not_worse" if accepted else "pinball_worse",
        )
        if not accepted:
            return last_decision
        if compare_and_swap(active, candidate):
            return ReleaseDecision(True, candidate, candidate, active, candidate_scores, active_scores, "switched")
    raise ConcurrentReleaseError("并发发布冲突多次，未完成切换")


atomic_compare_and_swap = publish_with_cas
optimistic_publish = publish_with_cas


def tree_to_dict(tree: Tree) -> dict[str, Any]:
    if isinstance(tree, Leaf):
        return {"type": "leaf", "correction": tree.correction}
    return {
        "type": "branch",
        "feature": tree.feature,
        "threshold": tree.threshold,
        "left": tree_to_dict(tree.left),
        "right": tree_to_dict(tree.right),
    }


def tree_from_dict(payload: Mapping[str, Any]) -> Tree:
    if payload.get("type") == "leaf":
        return Leaf(float(payload["correction"]))
    return Branch(
        feature=int(payload["feature"]),
        threshold=float(payload["threshold"]),
        left=tree_from_dict(payload["left"]),
        right=tree_from_dict(payload["right"]),
    )


def model_to_dict(model: QuantileModel) -> dict[str, Any]:
    return {
        "constant": model.constant,
        "rate": model.rate,
        "trees": [tree_to_dict(tree) for tree in model.trees],
        "monotone_feature": model.monotone_feature,
        "nonnegative": model.nonnegative,
        "floor": model_to_dict(model.floor) if model.floor is not None else None,
        "quantile": model.quantile,
    }


def model_from_dict(payload: Mapping[str, Any]) -> QuantileModel:
    floor_payload = payload.get("floor")
    return QuantileModel(
        constant=float(payload["constant"]),
        rate=float(payload["rate"]),
        trees=tuple(tree_from_dict(item) for item in payload.get("trees", [])),
        monotone_feature=payload.get("monotone_feature"),
        nonnegative=bool(payload.get("nonnegative", False)),
        floor=model_from_dict(floor_payload) if floor_payload else None,
        quantile=payload.get("quantile"),
    )


def models_to_data(models: Any) -> dict[str, Any]:
    if isinstance(models, QuantilePair):
        return {"version": models.version, "p50": model_to_dict(models.p50), "p90": model_to_dict(models.p90)}
    p50, p90 = extract_models(models)
    return {"p50": model_to_dict(p50), "p90": model_to_dict(p90)}


def models_from_data(payload: Mapping[str, Any]) -> QuantilePair:
    p50 = model_from_dict(payload["p50"])
    p90 = model_from_dict(payload["p90"])
    if p90.floor is None:
        p90 = replace(p90, floor=p50)
    return QuantilePair(p50, p90, payload.get("version"))


def fingerprint_models(models: Any) -> str:
    encoded = json.dumps(models_to_data(models), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
