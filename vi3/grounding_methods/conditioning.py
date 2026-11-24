"""Pose conditioning — the IMU trajectory as the foundation model's pose input.

For foundation models that accept camera poses as an input (Pi3X, DA3, Any4D), the IMU
target trajectory of the shared init (``base.Grounder``) is fed in metric scale
as the pose prior of a single forward pass; the model's own outputs (poses and
depth) are then read as usual. No weights are updated and α stays 1. Suited to
models that accept poses: the metric scale reaches their depth through the
model itself, even when their pose and depth scales differ.

An adapter supports conditioning by exposing a `cond_poses` attribute
([N, 4, 4] camera→world, OpenCV axes, frame-0 relative; None = images only) and
optionally `cond_metric` / `cond_intrinsics` (see `vi3/models/README.md`). The
IMU poses are fed pose-only (no intrinsics prior), as metric.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vi3.data.preprocessing import prepare_batch
from vi3.grounding_methods.base import Grounder
from vi3.models.pi3_adapter import set_forward_intrinsics
from vi3.transforms import _M_NP


def pose_input_adapter(model):
    """The module in the `.model` chain that takes poses (declares `cond_poses`),
    or None when the model has no pose input."""
    m = model
    while m is not None:
        if "cond_poses" in vars(m):
            return m
        inner = getattr(m, "model", None)
        m = None if inner is m else inner
    return None


def target_poses_to_c2w_opencv(pos_list, quat_list):
    """[N, 4, 4] camera→world in OpenCV camera axes from frame-0-relative target
    poses (TartanAir camera axes: t, quat xyzw): R_ocv = Mᵀ R M, t_ocv = Mᵀ t with
    M the OpenCV→TartanAir change of basis."""
    Ts = []
    for t_tar, q in zip(pos_list, quat_list):
        T = np.eye(4)
        T[:3, :3] = _M_NP.T @ Rotation.from_quat(np.asarray(q)).as_matrix() @ _M_NP
        T[:3, 3] = _M_NP.T @ np.asarray(t_tar)
        Ts.append(T)
    return torch.tensor(np.stack(Ts), dtype=torch.float32)


class ConditioningGrounder(Grounder):
    """The IMU trajectory fed as the foundation model's (metric) pose input."""

    @torch.no_grad()
    def train(self, frame_indices, config):
        """Init + one conditioned forward per evaluation; no optimisation."""
        self._new_window(config)
        model = self.reset_model().eval()
        adapter = pose_input_adapter(model)
        if adapter is None:
            raise ValueError(
                f"{type(self.model_base).__name__} takes no pose input: its adapter "
                f"has no `cond_poses` attribute (see vi3/models/README.md).")
        prep = prepare_batch(self.dataset, frame_indices, self.device)
        images, samples = prep["images"], prep["samples"]
        K = torch.stack([s["K"].to(self.device) for s in samples])
        set_forward_intrinsics(model, K, (images.shape[-2], images.shape[-1]))
        # The init runs on the unconditioned model (`model_base`), as for CFA / TTR.
        self.cached_targets = self._build_targets(samples, config)

        pos = [p.detach().cpu().numpy() for p in self.cached_targets["p"]]
        quat = [q.detach().cpu().numpy() for q in self.cached_targets["q"]]
        adapter.cond_poses = target_poses_to_c2w_opencv(pos, quat).to(self.device)
        if "cond_metric" in vars(adapter):
            adapter.cond_metric = True
        if "cond_intrinsics" in vars(adapter):
            adapter.cond_intrinsics = False

        history = {}
        self._post_step(0, {}, history)
        return model, history
