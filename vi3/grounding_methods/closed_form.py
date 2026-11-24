"""Closed-form IMU alignment (CFA) — metric grounding without refinement.

Same IMU init and inertial trajectory as the other grounders (``base.Grounder``),
but the foundation model's weights are never updated. Per window the frozen
model's camera centres are fitted to the IMU-integrated trajectory with a Sim(3)
(Umeyama) alignment, and its scale is applied — to translations and depth,
through the α wrapper of ``base.Grounder``. With ``gyro=True`` (CFA+gyro, the
default method) each frame's rotation is also taken from the gyro-integrated IMU
rotation, keeping the predicted camera centre; ``gyro=False`` (``cfa``) applies
the scale alone.

A fast, general-purpose way to ground a model: metric pose and depth in ~1 s per
window, with the model's own geometry kept as it is.
"""

import torch
import torch.nn as nn

from vggt.utils.rotation import mat_to_quat

from vi3.data.preprocessing import prepare_batch
from vi3.models.pi3_adapter import set_forward_intrinsics
from vi3.grounding_methods.base import Grounder, _ScaleWrappedModel
from vi3.transforms import pose_enc_to_target_pose_torch, target_pose_to_opencv_extrinsic
from vi3.utils import procrustes_alignment


class GyroRotationModel(nn.Module):
    """Replaces each frame's predicted rotation with the IMU (gyro-integrated)
    rotation, keeping the predicted camera centre; depth is untouched."""

    def __init__(self, model, q_imu):
        super().__init__()
        self.model = model
        self.q_imu = q_imu                       # [N, 4] frame-0-relative c2w, xyzw

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        pe = out["pose_enc"].clone()
        for i in range(pe.shape[1]):
            c = pose_enc_to_target_pose_torch(pe[0, i].float())[:3]
            E = target_pose_to_opencv_extrinsic(c, self.q_imu[i].to(c))
            pe[0, i, :3] = E[:, 3].to(pe.dtype)
            pe[0, i, 3:7] = mat_to_quat(E[:, :3]).to(pe.dtype)
        out["pose_enc"] = pe
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class ClosedFormGrounder(Grounder):
    """CFA: the IMU-derived Umeyama scale applied to a frozen foundation model.

    Config knobs (beyond the IMU-init ones of ``Grounder``):
      * ``gyro`` (default True): also take each frame's rotation from the gyro.
    """

    @torch.no_grad()
    def train(self, frame_indices, config):
        """Init + closed-form fit; no optimisation steps. Returns the grounded
        model and a history carrying the init diagnostics and α."""
        self._new_window(config)
        # Same copy flags as TTR (depth head trainable when there is one): nothing
        # is trained, but which heads require grad can select a different kernel
        # path in some models (Pi3).
        has_depth_head = self._get_depth_head(self.model_base) is not None
        model = self.reset_model(unfreeze_depth_head=has_depth_head).eval()
        prep = prepare_batch(self.dataset, frame_indices, self.device)
        images, samples = prep["images"], prep["samples"]
        K = torch.stack([s["K"].to(self.device) for s in samples])
        set_forward_intrinsics(model, K, (images.shape[-2], images.shape[-1]))
        self.cached_targets = self._build_targets(samples, config)

        # Raw (α = 1) camera centres of the frozen model vs the IMU trajectory.
        inner = model.model if isinstance(model, _ScaleWrappedModel) else model
        out = inner(images.unsqueeze(0))
        p_pred = torch.stack([pose_enc_to_target_pose_torch(p)
                              for p in out["pose_enc"][0].float()])[:, :3].double()
        p_imu = torch.stack(self.cached_targets["p"]).to(self.device).double()
        _, s, _, _ = procrustes_alignment(p_pred.T, p_imu.T)
        self.alpha.fill_(float(s) if float(s) > 0 else 1.0)

        history = {}
        self._post_step(0, {}, history)
        if config.get("gyro", True):
            model = GyroRotationModel(
                model, torch.stack(self.cached_targets["q"]).to(self.device))
        return model, history
