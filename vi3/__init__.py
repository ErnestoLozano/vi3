"""vi3 — metric grounding of pretrained 3D foundation models with an IMU.

A GT-free visual-inertial init per window, then closed-form alignment
(``vi3.grounding_methods.ClosedFormGrounder``), test-time refinement
(``TTRGrounder``) or pose conditioning (``ConditioningGrounder``) of a frozen
foundation model (``vi3.models``).
"""

__version__ = "0.1.0"
