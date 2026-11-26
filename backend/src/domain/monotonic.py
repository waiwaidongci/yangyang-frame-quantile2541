"""Helpers named for the frame-monotonic acceptance property."""

from __future__ import annotations

from typing import Any

from .boosting import QuantileModel, predict_shot


def monotonic_predict(model: QuantileModel, features: tuple) -> float:
    return model.predict(features)


def enforce_frame_monotonicity(model: QuantileModel) -> QuantileModel:
    if model.monotone_feature == 0:
        return model
    from dataclasses import replace

    return replace(model, monotone_feature=0)


def predict_pair(row: dict, models: Any) -> tuple[float, float]:
    predicted = predict_shot(row, models)
    return predicted["p50_minutes"], predicted["p90_minutes"]


# Verb aliases.
certify = enforce_frame_monotonicity
make_monotone = enforce_frame_monotonicity
