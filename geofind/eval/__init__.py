"""geofind evaluation framework — accuracy measurement, ablation studies, reporting."""

from geofind.eval.metrics import EvalMetrics, ImageResult
from geofind.eval.runner import EvalRunner

__all__ = ["EvalRunner", "EvalMetrics", "ImageResult"]
