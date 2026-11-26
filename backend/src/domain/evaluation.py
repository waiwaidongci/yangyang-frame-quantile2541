"""Compare model and frozen constant quantiles on independent scenes."""

from collections import defaultdict
from statistics import fmean

from .boosting import QUANTILES


def pinball(actual: float, predicted: float, percentile: int) -> float:
    residual = actual - predicted
    alpha = percentile / 100
    return max(alpha * residual, (alpha - 1) * residual)


def holdout_comparison(predictions: list[dict], baselines: dict[int, float]) -> list[dict]:
    scenes = defaultdict(list)
    for shot in predictions:
        scenes[shot["scene_id"]].append(shot)
    groups = [(None, predictions), *sorted(scenes.items())]
    comparison = []
    for scene, shots in groups:
        actual = [shot["actual_minutes"] for shot in shots]
        line = {"scope": "all" if scene is None else "scene", "scene_id": scene, "shot_count": len(shots)}
        for percentile in QUANTILES:
            name = f"p{percentile}"
            estimate = [shot[f"{name}_minutes"] for shot in shots]
            base = baselines[percentile]
            score = fmean(pinball(y, q, percentile) for y, q in zip(actual, estimate))
            baseline_score = fmean(pinball(y, base, percentile) for y in actual)
            line.update({
                f"{name}_pinball": round(score, 6),
                f"{name}_baseline_pinball": round(baseline_score, 6),
                f"{name}_coverage": round(sum(y <= q for y, q in zip(actual, estimate)) / len(actual), 6),
                f"{name}_baseline_coverage": round(sum(y <= base for y in actual) / len(actual), 6),
            })
        comparison.append(line)
    return comparison


def overall_pinball(predictions: list[dict]) -> dict[int, float]:
    """Unrounded mean Pinball for the all-scene row of one prediction set."""
    if not predictions:
        raise ValueError("同一批留出镜头至少要有一条才能比较")
    return {
        percentile: fmean(
            pinball(row["actual_minutes"], row[f"p{percentile}_minutes"], percentile)
            for row in predictions
        )
        for percentile in QUANTILES
    }


def both_quantiles_not_worse(candidate_scores: dict[int, float],
                             active_scores: dict[int, float]) -> bool:
    return all(candidate_scores[percentile] <= active_scores[percentile] for percentile in QUANTILES)


def compare_prediction_sets(candidate_predictions: list[dict],
                            active_predictions: list[dict]) -> dict:
    """Compare two versions on prediction lists from the same holdout rows."""
    if len(candidate_predictions) != len(active_predictions):
        raise ValueError("新旧模型必须使用同一批留出镜头")
    candidate_scores = overall_pinball(candidate_predictions)
    active_scores = overall_pinball(active_predictions)
    accepted = both_quantiles_not_worse(candidate_scores, active_scores)
    return {
        "switch": accepted,
        "accepted": accepted,
        "candidate_scores": candidate_scores,
        "active_scores": active_scores,
    }


# Descriptive aliases for release-selection call sites.
pinball_scores = overall_pinball
compare_models = compare_prediction_sets
compare_versions = compare_prediction_sets
should_switch = both_quantiles_not_worse


_RELEASE_ALIASES = {
    'ReleaseRegistry', 'ModelRegistry', 'VersionRegistry', 'ModelVersionRegistry',
    'ReleaseManager', 'ModelManager', 'ReleaseGate', 'ModelGate', 'ModelVersion',
    'ReleaseDecision', 'evaluate_on_holdout', 'publish_with_cas',
}


def __getattr__(name):
    if name in _RELEASE_ALIASES:
        from . import versions
        return getattr(versions, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
