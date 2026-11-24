"""Evaluation: the window evaluator, its metrics, the per-window grounding unit
(`run_cell`) and the end-of-run table."""

from vi3.eval.evaluator import Evaluator
from vi3.eval.metrics import calculate_metrics

__all__ = ["Evaluator", "calculate_metrics"]
