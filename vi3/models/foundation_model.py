"""The foundation model contract — what a 3D foundation model must provide to vi3.

vi3 talks to every foundation model through one forward signature and one output dict:

    forward(images) -> {"pose_enc", "depth", "depth_conf"?, "world_points"?}

  images      [B, N, 3, H, W] in [0, 1] (B = 1), as loaded by the `ModelSpec`.
  pose_enc    [B, N, 9] = [t(3), q(4), fov_h, fov_w]: world→camera OpenCV
              extrinsic (translation, quaternion xyzw) with frame 0 as the
              world (frame 0 = identity), plus the field of view in radians.
  depth       [B, N, H, W, 1] z-depth of each frame.
  depth_conf  [B, N, H, W] or [..., 1] (optional; visualization only).
  world_points [B, N, H, W, 3] in the frame-0 world (optional; derived from
              depth + pose when absent).

Most models predict camera→world poses, depth and (or) intrinsics in their own
world frame; `outputs_from_poses` does the frame-0 rebase and the conversion to
this dict, so an adapter is usually a dozen lines (see `vi3/models/README.md`).

That is all the closed-form alignment (CFA, the default method) needs. TTR
(test-time refinement) additionally unfreezes parts of the model by attribute
name: `camera_head` (always), `aggregator.global_blocks` / `aggregator.frame_blocks`
(the last `unfreeze_n` blocks; `frame_blocks` may be an empty list) and
`depth_head` or `dense_head` (with the default loss). Expose those as aliases
of the model's own submodules.
"""

import torch
import torch.nn as nn

from vggt.utils.rotation import mat_to_quat


class FoundationModel(nn.Module):
    """Base class for adapters. Before every forward, the grounders set
    `cached_K` ([N, 3, 3], the dataset intrinsics at the loaded resolution) and
    `cached_image_hw` ((H, W)) through `set_forward_intrinsics`, for models that
    take intrinsics as input or need them for the FoV channels."""

    def __init__(self):
        super().__init__()
        self.cached_K = None
        self.cached_image_hw = None

    def forward(self, images):
        raise NotImplementedError


def _se3_inverse(T):
    """Batched SE(3) inverse. T: [..., 4, 4]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_inv = R.transpose(-2, -1)
    out = torch.zeros_like(T)
    out[..., :3, :3] = R_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", R_inv, t)
    out[..., 3, 3] = 1.0
    return out


def _fov_from_K(K, image_hw):
    """[fov_h, fov_w] from intrinsics. K: [..., 3, 3]."""
    H, W = image_hw
    fov_h = 2.0 * torch.atan2(torch.full_like(K[..., 1, 1], H / 2.0), K[..., 1, 1])
    fov_w = 2.0 * torch.atan2(torch.full_like(K[..., 0, 0], W / 2.0), K[..., 0, 0])
    return fov_h, fov_w


def outputs_from_poses(cam_to_world, depth, K, image_hw, *,
                       world_points=None, depth_conf=None):
    """Build the foundation model output dict from a model's raw predictions.

    Args:
      cam_to_world  [B, N, 4, 4] camera→world poses, OpenCV camera axes, in any
                    world frame (rebased here so frame 0 is the world).
      depth         [B, N, H, W] or [B, N, H, W, 1] z-depth.
      K             [B, N, 3, 3] or [N, 3, 3] intrinsics for the FoV channels —
                    the model's own prediction, or the adapter's `cached_K`.
      image_hw      (H, W) of the input images.
      world_points  optional [B, N, H, W, 3] in the same world as cam_to_world.
      depth_conf    optional confidence map.
    """
    B, N = cam_to_world.shape[:2]
    T_inv0 = _se3_inverse(cam_to_world[:, 0])                          # [B, 4, 4]
    T_rebased = torch.einsum("bij,bnjk->bnik", T_inv0, cam_to_world)   # cam→cam0
    T_wc = _se3_inverse(T_rebased)                                     # world→cam
    t_wc = T_wc[..., :3, 3]
    q_wc = mat_to_quat(T_wc[..., :3, :3].reshape(B * N, 3, 3)).reshape(B, N, 4)
    K = K.to(device=t_wc.device, dtype=t_wc.dtype)
    if K.dim() == 3:
        K = K.unsqueeze(0).expand(B, -1, -1, -1)
    fov_h, fov_w = _fov_from_K(K, image_hw)
    out = {
        "pose_enc": torch.cat([t_wc, q_wc, fov_h.unsqueeze(-1),
                               fov_w.unsqueeze(-1)], dim=-1),          # [B, N, 9]
        "depth": depth if depth.dim() == 5 else depth.unsqueeze(-1),
    }
    if depth_conf is not None:
        out["depth_conf"] = depth_conf
    if world_points is not None:
        pts_h = torch.cat([world_points, torch.ones_like(world_points[..., :1])], dim=-1)
        out["world_points"] = torch.einsum("bij,bnhwj->bnhwi", T_inv0, pts_h)[..., :3]
    return out
