"""Coordinate conventions and frame conversions.

VGGT's `pose_enc` is the world->camera **extrinsic** in OpenCV camera axes
(x-right, y-down, z-forward). VGGT's "world" is just camera-0 in those
axes — there is no global world, only a frame attached to the first input
camera. So converting VGGT's output for comparison with dataset GT is
purely a **camera-axes change of basis** plus an SE(3) inversion; no
world-frame information is involved.

The grounders and target poses live in TartanAir's camera convention
(x-forward, y-right, z-down). The change of basis is one proper rotation:

    OpenCV (right, down, forward) -> TartanAir (forward, right, down)
    M = [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]    (det = +1, 120° around (1,1,1)/sqrt(3))

For one VGGT pose_enc the conversion is:
    1. R_cw = R_wc^T,  t_cw = -R_wc^T @ t_wc          (invert extrinsic)
    2. R_new = M @ R_cw @ M^T,  t_new = M @ t_cw      (cam axes basis change)

The world frame is never touched: both VGGT's implicit world (cam-0 OpenCV)
and the output's world (cam-0 TartanAir axes) are local camera frames.
`pose_enc_to_target_pose` thus converts a VGGT pose_enc into the shape and
convention of the IMU target poses the grounders compare against.

Gravity direction for IMU integration is dataset-specific (NED z-down
vs ENU z-up) and lives on the Dataset class as `GRAVITY_WORLD`, *not*
here — the world frame of the GT poses is whatever the dataset says it
is and the grounder reads gravity from there.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.rotation import mat_to_quat, quat_to_mat


# Numpy and torch forms of the OpenCV-cam -> TartanAir-cam basis change.
# Both are det=+1 proper rotations.
_M_NP = np.array([[0., 0., 1.],
                  [1., 0., 0.],
                  [0., 1., 0.]])


def M_change_of_basis(device=None, dtype=None):
    """Proper rotation OpenCV-cam -> TartanAir-cam (det = +1)."""
    return torch.tensor([[0., 0., 1.],
                         [1., 0., 0.],
                         [0., 1., 0.]], device=device, dtype=dtype)


def vec_to_se3(v, device=None):
    """Build a 4x4 SE(3) matrix from a 7-vec [tx, ty, tz, qx, qy, qz, qw].

    v: torch.Tensor of shape [7]. Uses scipy for the quaternion->matrix step
    (this is meant for one-off conversions on the CPU; for the differentiable
    path use `quat_to_mat` from `vggt.utils.rotation`).
    """
    device = device if device is not None else v.device
    t, q = v[:3], v[3:]
    rot_mat = R.from_quat(q.detach().cpu().numpy()).as_matrix()
    mat = torch.eye(4, device=device, dtype=v.dtype)
    mat[:3, :3] = torch.from_numpy(rot_mat).to(device=device, dtype=v.dtype)
    mat[:3, 3] = t.to(device=device, dtype=v.dtype)
    return mat


def pose_enc_to_target_pose(vec):
    """Convert one VGGT pose_enc (world->cam extrinsic, OpenCV axes) to a
    camera->world pose in TartanAir camera convention. Numpy path; used for
    evaluation and plotting.

    vec: torch.Tensor of shape [>=7] — first 7 entries are [t(3), q(4)].
    Returns: torch.Tensor of shape [7] on the same device/dtype as `vec`.
    """
    t_wc = vec[:3].detach().cpu().numpy().astype(np.float64)
    q_wc = vec[3:7].detach().cpu().numpy().astype(np.float64)

    R_wc = R.from_quat(q_wc).as_matrix()
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc

    R_new = _M_NP @ R_cw @ _M_NP.T
    t_new = _M_NP @ t_cw
    q_new = R.from_matrix(R_new).as_quat()

    t_out = torch.from_numpy(t_new).to(vec.device).to(vec.dtype)
    q_out = torch.from_numpy(q_new).to(vec.device).to(vec.dtype)
    return torch.cat([t_out, q_out])


def pose_enc_to_target_pose_torch(vec):
    """Differentiable equivalent of `pose_enc_to_target_pose`.

    vec: torch.Tensor of shape [>=7] — first 7 entries are [t(3), q(4)].
    Returns: torch.Tensor of shape [7] on the same device/dtype as `vec`.

    Uses VGGT's `quat_to_mat` / `mat_to_quat`. The latter applies
    `standardize_quaternion` (forces w >= 0); this is sign-invariant under our
    downstream losses (geodesic distance uses |dot|; Hamilton-product losses
    handle either sign).
    """
    t_wc = vec[:3]
    q_wc = vec[3:7]
    q_wc = q_wc / (torch.norm(q_wc) + 1e-12)
    R_wc = quat_to_mat(q_wc)

    R_cw = R_wc.t()
    t_cw = -R_wc.t() @ t_wc

    M = M_change_of_basis(device=vec.device, dtype=vec.dtype)
    R_new = M @ R_cw @ M.t()
    t_new = M @ t_cw

    q_new = mat_to_quat(R_new)
    q_new = q_new / (torch.norm(q_new) + 1e-12)
    return torch.cat([t_new, q_new])


def target_pose_to_opencv_extrinsic(p, q):
    """Inverse of `pose_enc_to_target_pose_torch` at the SE(3)-matrix level.

    Takes a frame-0-relative target pose expressed as (camera->world) in
    TartanAir camera axes — the convention produced by `_build_imu_targets`
    (Grounder) — and returns a [3, 4] world->cam extrinsic in VGGT's
    OpenCV camera convention. That extrinsic is directly compatible with
    `vggt.utils.geometry.project_world_points_to_cam`.

    The world frame is also implicitly "camera 0" in both conventions, so
    the change of basis applies to both world and cam sides of the SE(3).

    p: [3] target translation
    q: [4] target quaternion [xyzw]
    Returns: [3, 4] tensor (R | t) — world->cam in OpenCV axes
    """
    # Cam-to-world rotation/translation in TartanAir basis.
    q_unit = q / (torch.norm(q) + 1e-12)
    R_cw = quat_to_mat(q_unit)

    # Invert to world-to-cam (still TartanAir basis).
    R_wc = R_cw.t()
    t_wc = -R_wc @ p

    # Change basis to OpenCV: M^T @ R @ M for rotations, M^T @ t for translations.
    M = M_change_of_basis(device=p.device, dtype=p.dtype)
    R_wc_open = M.t() @ R_wc @ M
    t_wc_open = M.t() @ t_wc

    return torch.cat([R_wc_open, t_wc_open.unsqueeze(-1)], dim=-1)


def derive_world_points_from_depth(depth, pose_enc, K, image_hw):
    """Unproject VGGT's predicted depth into the cam-0 world (OpenCV) frame.

    Used instead of a model's point-map output: VGGT-Ω has no point-map head,
    and for VGGT the VGGT authors recommend this path over the point map
    (consistent with the depth head by construction).

    Args:
        depth:    [B, S, H, W, 1] or [B, S, H, W] — predicted depth.
        pose_enc: [B, S, 9] — VGGT pose_enc (OpenCV world->cam extrinsic
                  with [t(3), q(4), fov_h, fov_w]).
        K:        [B, S, 3, 3] or [S, 3, 3] or [3, 3] — per-frame
                  intrinsics matching the same (H, W).
        image_hw: (H, W) — image dims used to interpret pose_enc's FoV.

    Returns:
        world_points: [B, S, H, W, 3] in VGGT's cam-0 OpenCV world frame.
        Self-consistent with `pose_enc` and `depth` by construction.

    Math:
        extr (world->cam OpenCV) = pose_encoding_to_extri_intri(pose_enc)
        R_cw = R_wc^T,  t_cw = -R_cw @ t_wc                  (invert)
        For each pixel (u, v):
            ray_cam  = K^-1 @ [u, v, 1]^T                    (unit ray in cam)
            pt_cam   = depth[u, v] * ray_cam                 (point in cam)
            pt_world = R_cw @ pt_cam + t_cw                  (cam -> world)
    """
    H, W = image_hw

    if depth.dim() == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]                                # [B, S, H, W]
    assert depth.dim() == 4, f"depth must be [B,S,H,W,(1)], got {depth.shape}"

    B, S, _, _ = depth.shape
    device, dtype = depth.device, depth.dtype

    # 1. Extrinsics (world->cam, OpenCV) from pose_enc, then invert.
    extr_wc, _ = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=image_hw, build_intrinsics=False,
    )                                                         # [B, S, 3, 4]
    R_wc = extr_wc[..., :3, :3]                              # [B, S, 3, 3]
    t_wc = extr_wc[..., :3, 3]                                # [B, S, 3]
    R_cw = R_wc.transpose(-1, -2)                             # [B, S, 3, 3]
    t_cw = -torch.einsum("bsij,bsj->bsi", R_cw, t_wc)         # [B, S, 3]

    # 2. Broadcast K to [B, S, 3, 3] and invert. K is typically a stack of
    #    per-frame intrinsics; allow lower-dim inputs (per-sequence or shared).
    K = K.to(device=device, dtype=dtype)
    if K.dim() == 2:
        K = K.expand(B, S, 3, 3)
    elif K.dim() == 3:
        # Assume [S, 3, 3]; broadcast to [B, S, 3, 3].
        K = K.unsqueeze(0).expand(B, S, 3, 3)
    K_inv = torch.linalg.inv(K)                               # [B, S, 3, 3]

    # 3. Build per-pixel homogeneous coords [u, v, 1] once, shape [H*W, 3].
    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack([us, vs, torch.ones_like(us)], dim=-1)  # [H, W, 3]
    pixels = pixels.reshape(-1, 3)                                # [H*W, 3]

    # 4. Unproject: ray_cam[..., p, :] = K_inv @ pixels[p, :]
    #    rays: [B, S, H*W, 3]
    rays = torch.einsum("bsij,pj->bspi", K_inv, pixels)          # [B, S, H*W, 3]

    # 5. Scale by depth (broadcast over xyz):
    #    pt_cam[..., p, :] = depth[..., p] * rays[..., p, :]
    d = depth.reshape(B, S, -1).unsqueeze(-1)                    # [B, S, H*W, 1]
    pts_cam = d * rays                                            # [B, S, H*W, 3]

    # 6. Cam -> world: pt_world = R_cw @ pt_cam + t_cw  (per frame).
    pts_world = torch.einsum("bsij,bspj->bspi", R_cw, pts_cam) \
                 + t_cw.unsqueeze(-2)                             # [B, S, H*W, 3]

    return pts_world.reshape(B, S, H, W, 3)
