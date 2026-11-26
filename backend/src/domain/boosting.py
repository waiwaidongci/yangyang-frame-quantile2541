"""Gradient-boosted quantiles with an exact pinball line search per terminal leaf."""

from collections.abc import Sequence
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import floor

from .settings import FEATURES
from .trees import Branch, Leaf, Tree, Vector, fit_tree

QUANTILES = (50, 90)
FRAME_FEATURE = FEATURES.index("frames")
MIN_FRAMES = 1
MAX_FRAMES = 1800


def shot_features(row: dict) -> Vector:
    return tuple(row[field] for field in FEATURES)


def empirical_quantile(values: Sequence[float], percentile: int) -> float:
    """Return the smallest value whose empirical CDF reaches the given percentile."""
    ordered = sorted(values)
    rank = (len(ordered) * percentile + 99) // 100
    return ordered[rank - 1]


@dataclass(frozen=True)
class QuantileModel:
    constant: float
    rate: float
    trees: tuple[Tree, ...]
    # Fitted models set this to the ledger's frame feature.  The dataclass
    # default preserves the raw sum for low-level tree/oracle helpers.
    monotone_feature: int | None = None
    nonnegative: bool = False
    floor: "QuantileModel | None" = None
    quantile: int | None = None

    def raw_predict(self, features: Vector) -> float:
        """Evaluate the fitted boosted sum without certification projections."""
        prediction = self.constant
        for tree in self.trees:
            prediction += self.rate * tree.predict(features)
        return prediction

    def _frame_candidates(self, features: Vector) -> set[float | int]:
        """Return starts of every frame interval visible in this tree ensemble."""
        current = features[self.monotone_feature]
        candidates = {current}
        if self.monotone_feature == FRAME_FEATURE and current >= MIN_FRAMES:
            candidates.add(MIN_FRAMES)

        def visit(node: Tree) -> None:
            if not isinstance(node, Branch):
                return
            if node.feature == self.monotone_feature:
                # Predicate is frame <= threshold, so floor(threshold) + 1 is
                # the first integer in the right-hand interval.
                start = floor(node.threshold) + 1
                low = MIN_FRAMES if node.feature == FRAME_FEATURE else -float("inf")
                if low <= start <= current:
                    candidates.add(start)
            visit(node.left)
            visit(node.right)

        for tree in self.trees:
            visit(tree)
        return candidates

    def unary_predict(self, features: Vector) -> float:
        """Certify monotonicity/non-negativity, but ignore a quantile floor."""
        if self.monotone_feature is None:
            prediction = self.raw_predict(features)
        elif self.monotone_feature == FRAME_FEATURE and MIN_FRAMES <= features[FRAME_FEATURE] <= MAX_FRAMES:
            current = features[FRAME_FEATURE]
            starts, prefix_max = frame_profile(self, tuple(features[1:]))
            index = bisect_right(starts, current) - 1
            prediction = prefix_max[max(0, index)]
        else:
            feature_index = self.monotone_feature
            current = features[feature_index]
            prediction = self.raw_predict(features)
            for value in self._frame_candidates(features):
                trial = list(features)
                trial[feature_index] = value
                prediction = max(prediction, self.raw_predict(tuple(trial)))
        return max(0.0, prediction) if self.nonnegative else prediction

    def predict(self, features: Vector) -> float:
        prediction = self.unary_predict(features)
        if self.floor is not None:
            prediction = max(prediction, self.floor.predict(features))
        return prediction


@lru_cache(maxsize=4096)
def frame_profile(model: "QuantileModel", other_features: tuple) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Prefix maxima for every frame interval at one fixed non-frame coordinate."""
    starts = {float(MIN_FRAMES)}

    def visit(node: Tree) -> None:
        if not isinstance(node, Branch):
            return
        if node.feature == FRAME_FEATURE:
            start = floor(node.threshold) + 1
            if MIN_FRAMES < start <= MAX_FRAMES:
                starts.add(float(start))
        visit(node.left)
        visit(node.right)

    for tree in model.trees:
        visit(tree)

    ordered_starts = tuple(sorted(starts))
    prefix_values = []
    best = float("-inf")
    for start in ordered_starts:
        value = model.raw_predict((int(start), *other_features))
        best = max(best, value)
        prefix_values.append(best)
    return ordered_starts, tuple(prefix_values)


@dataclass(frozen=True)
class QuantilePair:
    """One immutable P50/P90 release; scheduling must use this pair as a unit."""

    p50: QuantileModel
    p90: QuantileModel
    version: str | None = None

    @property
    def models(self) -> tuple[QuantileModel, QuantileModel]:
        return self.p50, self.p90

    def predict_pair(self, features: Vector) -> tuple[float, float]:
        p50 = max(0.0, self.p50.predict(features))
        p90 = max(0.0, self.p90.predict(features), p50)
        return p50, p90

    def predict(self, features: Vector) -> tuple[float, float]:
        return self.predict_pair(features)


def fit_quantile(features: Sequence[Vector], actual: Sequence[float], percentile: int,
                 settings: dict, *, lower_bound: QuantileModel | None = None,
                 monotone_frames: bool = True, auto_lower_bound: bool = True) -> QuantileModel:
    if lower_bound is None and auto_lower_bound and percentile > 50:
        # Calls using the historical two-statement pattern still receive an
        # ordered pair directly: this deterministic P50 is identical to the
        # separately fitted P50 for the same data/settings.
        lower_bound = fit_quantile(
            features, actual, 50, settings,
            monotone_frames=monotone_frames, auto_lower_bound=False,
        )
    constant = empirical_quantile(actual, percentile)
    rate = settings["learning_rate"]
    alpha = percentile / 100
    fitted = [constant] * len(actual)
    trees = []
    for _ in range(settings["estimators"]):
        residuals = [observed - estimate for observed, estimate in zip(actual, fitted)]
        # Zero is a valid subgradient at an exact hit.
        gradients = [alpha if residual > 0 else alpha - 1 if residual < 0 else 0.0 for residual in residuals]

        def correction(members: list[int]) -> float:
            return empirical_quantile([residuals[index] for index in members], percentile)

        tree = fit_tree(features, gradients, correction, settings["max_depth"], settings["min_leaf"])
        fitted = [estimate + rate * tree.predict(vector) for estimate, vector in zip(fitted, features)]
        trees.append(tree)
    return QuantileModel(
        constant,
        rate,
        tuple(trees),
        monotone_feature=FRAME_FEATURE if monotone_frames else None,
        nonnegative=True,
        floor=lower_bound,
        quantile=percentile,
    )


def fit_model_pair(features: Sequence[Vector], actual: Sequence[float],
                   settings: dict, *, version: str | None = None) -> QuantilePair:
    """Fit ordered quantiles and bind P90 to the P50 floor by construction."""
    p50 = fit_quantile(features, actual, 50, settings)
    p90 = fit_quantile(features, actual, 90, settings, lower_bound=p50)
    return QuantilePair(p50, p90, version)


def predict_shot(row: dict, models: tuple[QuantileModel, QuantileModel] | QuantilePair) -> dict:
    vector = shot_features(row)
    if isinstance(models, QuantilePair):
        model50, model90 = models.models
    else:
        model50, model90 = models
    # ``predict`` includes the built-in P50 floor when a fitted ordered pair is
    # used.  The final max remains a safety certificate for hand-built pairs.
    direct_p50 = max(0.0, model50.predict(vector))
    direct_p90 = max(0.0, model90.predict(vector))
    crossing_adjusted = direct_p90 < direct_p50
    p50 = round(direct_p50, 6)
    p90 = round(max(direct_p90, direct_p50), 6)
    if p90 < p50:  # defensive against decimal rounding at the sixth place
        p90 = p50
    return {
        **row,
        "p50_minutes": p50,
        "p90_minutes": p90,
        "crossing_adjusted": crossing_adjusted,
        "model_version": getattr(models, "version", None),
    }


def train_models(features: Sequence[Vector], actual: Sequence[float], settings: dict,
                 *, version: str | None = None) -> QuantilePair:
    return fit_model_pair(features, actual, settings, version=version)


fit_models = train_models
build_models = train_models


def predict(row: dict, models: tuple[QuantileModel, QuantileModel] | QuantilePair) -> dict:
    """Convenience alias for call sites that name the prediction step directly."""
    return predict_shot(row, models)


def __getattr__(name):
    # Keep release-governance names reachable from the model module without a
    # circular top-level import (versions imports boosting).
    if name in {"ReleaseRegistry", "ModelRegistry", "ModelVersion", "fit_model_pair"}:
        if name == "fit_model_pair":
            return fit_model_pair
        from . import versions
        return getattr(versions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
