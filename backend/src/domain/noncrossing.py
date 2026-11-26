from .monotonic import *  # noqa: F401,F403
from .boosting import predict_shot


def noncrossing_pair(row, models):
    result = predict_shot(row, models)
    return result["p50_minutes"], result["p90_minutes"]
