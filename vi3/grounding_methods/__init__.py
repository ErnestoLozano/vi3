"""Grounding methods: three ways of grounding a frozen 3D foundation model in
metric scale with the IMU init and targets of `base.Grounder` — closed-form
alignment (+ gyro rotations), test-time refinement, and pose conditioning."""

from vi3.grounding_methods.base import Grounder
from vi3.grounding_methods.closed_form import ClosedFormGrounder
from vi3.grounding_methods.conditioning import ConditioningGrounder
from vi3.grounding_methods.ttr import TTRGrounder

__all__ = [
    "ClosedFormGrounder",
    "ConditioningGrounder",
    "Grounder",
    "TTRGrounder",
]
