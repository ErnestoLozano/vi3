# `_forward_head_oop` is adapted from Pi3 (https://github.com/yyfz/Pi3, pi3/models/pi3x.py, `Pi3X.forward_head`),
# modified to apply the metric scale out of place:
#   Copyright (c) 2025, the authors. All rights reserved.
#   Redistributed under Pi3's BSD-style license (third_party/pi3/LICENSE), whose conditions and disclaimer apply.

"""Pi3X foundation model adapter — re-expresses Pi3X's output in VGGT's interface.

Pi3X (`pi3.models.pi3x.Pi3X`, HF `yyfz233/Pi3X`) is Pi3 plus a learned
*metric/scale token*: a single scalar `metric = exp(head(decoder(token)))`
that multiplies the whole reconstruction (camera translations, points,
local points) to make it metric — see pi3x.py:435-441. Otherwise its output
format matches Pi3, so this adapter subclasses the Pi3 adapter and only
overrides the forward to (a) call Pi3X's forward, optionally with pose
conditioning, and (b) run the identical Pi3->VGGT I/O translation.

Scale handling for VI3 (TTR):
  We surface Pi3X's reconstruction *with its metric token applied* — so the
  frozen `base` result IS the scale-token feed-forward number. VI3's α
  (the grounders) then multiplies on top, i.e. it *corrects* the token
  via the IMU. α is initialised from the IMU-derived scale, so it starts
  from a data-driven estimate regardless of the token's error. This keeps
  exactly ONE scale degree of freedom that TTR drives (α), rather than
  fighting Pi3X's frozen scalar. The metric branch (metric_token /
  metric_decoder / metric_head) stays frozen.

Pose conditioning (the `cond` grounding method): set `self.cond_poses = [N,4,4]`
cam->world and forward feeds them to Pi3X with an all-True `mask_add_pose`. Left
as None for the normal images-only path.

Head surrogates (`camera_head`, `aggregator.global_blocks/frame_blocks`,
`dense_head`) are inherited from `_Pi3VGGTAdapter` and point at Pi3X's
identically-named submodules, so Grounder.reset_model / TTRGrounder._make_optimizer
walk Pi3X with the same code paths as VGGT/Pi3.
"""
import torch
import torch.nn.functional as F

from vi3.models.foundation_model import outputs_from_poses
from vi3.models.pi3_adapter import _Pi3VGGTAdapter


def _forward_head_oop(self, hidden, pos, B, N, H, W, patch_h, patch_w):
    """Byte-identical to Pi3X.forward_head EXCEPT the metric multiply on
    camera_poses (vendored line :438) is rebuilt OUT OF PLACE, so autograd
    doesn't see an in-place write on a graph-tracked tensor during TTR.

    Bound onto the Pi3X instance in the adapter (vendored file untouched).
    `self` is the Pi3X foundation model.
    """
    from pi3.utils.geometry import homogenize_points
    hw = patch_h * patch_w + self.patch_start_idx

    ret_point = self.point_decoder(hidden, xpos=pos)
    ret_camera = self.camera_decoder(hidden, xpos=pos)
    pos_hw = pos.reshape(B, N * hw, -1)
    ret_metric = self.metric_decoder(
        self.metric_token.repeat(B, 1, 1), hidden.reshape(B, N * hw, -1),
        xpos=pos_hw[:, 0:1], ypos=pos_hw)
    ret_conf = self.conf_decoder(hidden, xpos=pos)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        point_feat = ret_point[:, self.patch_start_idx:].float()
        xy, z = self._chunked_conv_head(self.point_head, point_feat, patch_h, patch_w)
        del point_feat
        xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
        z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
        z = torch.exp(z.clamp(max=15.0))
        local_points = torch.cat([xy * z, z], dim=-1)
        rays = F.normalize(torch.cat([xy, torch.ones_like(z)], dim=-1), dim=-1)
        camera_poses = self.camera_head(
            ret_camera[:, self.patch_start_idx:].float(),
            patch_h, patch_w).reshape(B, N, 4, 4)
        metric = self.metric_head(ret_metric.float()).reshape(B).exp()
        conf_feat = ret_conf[:, self.patch_start_idx:].float()
        conf = self._chunked_conv_head(self.conf_head, conf_feat, patch_h, patch_w)[0]
        del conf_feat
        conf = conf.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

        points = torch.einsum(
            "bnij, bnhwj -> bnhwi", camera_poses,
            homogenize_points(local_points))[..., :3] * metric.view(B, 1, 1, 1, 1)

        # --- vendored line :438 made OUT OF PLACE ---
        # original: camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric
        R_cp = camera_poses[..., :3, :3]                       # [B,N,3,3]
        t_cp = camera_poses[..., :3, 3:4] * metric.view(B, 1, 1, 1)  # [B,N,3,1]
        bottom = camera_poses[..., 3:4, :]                     # [B,N,1,4]
        camera_poses = torch.cat(
            [torch.cat([R_cp, t_cp], dim=-1), bottom], dim=-2)  # [B,N,4,4]

        local_points = local_points * metric.view(B, 1, 1, 1, 1)

    return dict(points=points, local_points=local_points, rays=rays,
                conf=conf, camera_poses=camera_poses, metric=metric)


class _Pi3XVGGTAdapter(_Pi3VGGTAdapter):
    """Pi3X wrapped to the VGGT-style {pose_enc, depth, ...} dict."""

    def __init__(self, pi3x):
        # Parent wires camera_head / aggregator / dense_head surrogates from
        # the foundation model's submodules (Pi3X shares Pi3's names) and sets up
        # cached_K / cached_image_hw. We store the foundation model at `self.pi3` too.
        super().__init__(pi3x)
        # Pose conditioning (the `cond` grounding method). None => images-only.
        self.cond_poses = None

        # ---- in-place autograd fix (no vendored-code edit) ----
        # Pi3X.forward_head applies its metric token to camera_poses IN PLACE:
        #   camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric   (:438)
        # and `points` (:435) reads camera_poses BEFORE that write, so points'
        # backward needs the un-mutated tensor. Under no_grad (base eval) this
        # is harmless; during TTR (camera_head trainable) it breaks backward
        # ("variable ... modified by an inplace operation"). Same cure as
        # _ScaleWrappedModel — rebuild OUT OF PLACE instead of mutating — applied
        # at the exact site by binding a patched forward_head onto the foundation model
        # instance (vendored file untouched). Everything else is byte-identical
        # to Pi3X.forward_head; only line :438 becomes an out-of-place cat.
        pi3x.forward_head = _forward_head_oop.__get__(pi3x, type(pi3x))

    def forward(self, images: torch.Tensor):
        if self.cached_K is None or self.cached_image_hw is None:
            raise RuntimeError(
                "_Pi3XVGGTAdapter.forward called without cached_K / "
                "cached_image_hw. Plumb them in from the grounder/evaluator."
            )
        kw = {}
        if self.cond_poses is not None:
            poses = self.cond_poses.to(images.device).unsqueeze(0)  # [1,N,4,4]
            N = poses.shape[1]
            mask = torch.ones((1, N), dtype=torch.bool, device=images.device)
            kw = dict(poses=poses, mask_add_pose=mask)

        # Pi3X internally runs some heads under a no-autocast block; wrap the
        # call in bf16 autocast to match how the model is meant to run. Grads
        # still flow for unfrozen params.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(images.device.type == "cuda")):
            out = self.pi3(images, **kw)

        # Same I/O translation as _Pi3VGGTAdapter.
        return outputs_from_poses(
            out["camera_poses"], out["local_points"][..., 2:3], self.cached_K,
            self.cached_image_hw, world_points=out["points"],
            depth_conf=out["conf"])
