"""Application entry points for the shot ledger and the background worker."""

from .boosting import QUANTILES, QuantilePair, fit_model_pair, fit_quantile, predict_shot, shot_features
from .evaluation import holdout_comparison
from .fixtures import demo_rows
from .records import normalize_rows
from .reports import holdout_shots, model_profile, pinball_table, scene_manifest, shift_queue
from .settings import SPEC, parse_settings
from .versions import (
    ModelRegistry,
    ModelVersion,
    ReleaseRegistry,
    extract_models,
    fingerprint_models,
    version_identifier,
)

__all__ = ["SPEC", "demo_rows", "normalize_rows", "analyze"]


def analyze(rows: list[dict], settings: dict | None = None,
            model_version: str | ModelVersion | None = None,
            version: str | ModelVersion | None = None,
            models=None) -> dict:
    options = parse_settings(settings or {})
    if model_version is not None and not isinstance(model_version, (str, ModelVersion)) and models is None:
        # Tolerate analyze(rows, settings, models_pair) call sites.
        models = model_version
        model_version = None
    if version is not None and not isinstance(version, (str, ModelVersion)) and models is None:
        models = version
        version = None
    records = sorted(normalize_rows(rows), key=lambda row: (row["scene_id"], row["id"]))
    train = [row for row in records if row["split"] == "train"]
    features = [shot_features(row) for row in train]
    actual = [row["actual_minutes"] for row in train]

    version_source = model_version if model_version is not None else version
    if isinstance(version_source, ModelVersion):
        version_id = version_source.version
        if models is None:
            models = version_source.models
    else:
        version_id = version_identifier(version_source) if version_source is not None else None

    if isinstance(models, QuantilePair):
        if version_id is None:
            version_id = models.version
        models = models.models
    elif models is not None:
        try:
            models = extract_models(models)
        except TypeError:
            pass

    if models is None:
        model_pair = fit_model_pair(features, actual, options, version=version_id)
        models = model_pair.models
        if version_id is None:
            version_id = f"mv-{fingerprint_models(model_pair)[:12]}"
    elif version_id is None:
        version_id = f"mv-{fingerprint_models(models)[:12]}"

    baselines = {50: models[0].constant, 90: models[1].constant}
    holdout = [{**predict_shot(row, models), "model_version": version_id}
               for row in records if row["split"] == "holdout"]
    pending = [{**predict_shot(row, models), "model_version": version_id}
               for row in records if row["split"] == "pending"]
    comparison = holdout_comparison(holdout, baselines)
    queue, scheduled, reserved = shift_queue(pending, options["shift_minutes"])
    crossings = sum(row["crossing_adjusted"] for row in holdout + pending)
    warnings = []
    if crossings:
        warnings.append(f"{crossings} 个镜头的独立分位投影存在交叉，已在同一版模型内以 P50 作为 P90 下限。")
    if len(holdout) < 20:
        warnings.append("留出镜头少于 20 个，覆盖率容易受个别镜头影响。")
    if len(train) < 2 * options["min_leaf"]:
        warnings.append("历史镜头不足以分裂回归树，本次模型输出历史常量分位。")
    if not pending:
        warnings.append("本次台账没有待排镜头，交接工时单为空。")
    overall = comparison[0]
    train_scenes = len({row["scene_id"] for row in train})
    deferred = len(pending) - scheduled
    return {
        "summary": (
            f"用 {train_scenes} 个历史场次的 {len(train)} 个镜头训练 P50 和 P90。"
            f"在 {len(comparison) - 1} 个独立留出场次的 {len(holdout)} 个镜头上对照常量分位。"
            f"本班可排 {scheduled} 个镜头，共预留 {reserved} 分钟，{deferred} 个需顺延复核。"
        ),
        "metrics": {
            "model_version": version_id,
            "P50 留出 Pinball": overall["p50_pinball"],
            "P90 留出 Pinball": overall["p90_pinball"],
            "P50 常量 Pinball": overall["p50_baseline_pinball"],
            "P90 常量 Pinball": overall["p90_baseline_pinball"],
            "P50 留出覆盖率": overall["p50_coverage"],
            "P90 留出覆盖率": overall["p90_coverage"],
            "已分配镜头数": scheduled,
            "待顺延镜头数": deferred,
            "本班已预留分钟": reserved,
        },
        "tables": [queue, pinball_table(comparison), holdout_shots(holdout), scene_manifest(records),
                   model_profile(options, baselines, version_id)],
        "model_version": version_id,
        "version": version_id,
        "active_model_version": version_id,
        "used_model_version": version_id,
        "model_release": {"version": version_id, "p50_constant": baselines[50], "p90_constant": baselines[90]},
        "warnings": warnings,
    }
