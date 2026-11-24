"""Loss terms of test-time refinement (``vi3.grounding_methods.ttr``): relative
pose deltas, the frame-0 origin anchor, focal length and cross-view depth
consistency."""

import torch
import torch.nn.functional as F

from vggt.utils.geometry import project_world_points_to_cam

from vi3.utils import quat_inv, quat_mul, rotation_geodesic_loss


def pose_delta_loss(pred_ned_poses, p_targets, q_targets):
    """L1 on translation deltas + geodesic on rotation deltas, averaged over pairs.

    Supervises *relative* motion between consecutive frames, not absolute pose
    — this is invariant to the world-frame origin and matches the structure of
    the IMU-preintegrated targets.

    pred_ned_poses: list of length N of 7-vec tensors
                     (post-`pose_enc_to_target_pose_torch`).
    p_targets:     list of length N of 3-vec target translation tensors.
    q_targets:     list of length N of 4-vec target quaternion tensors [xyzw].
    Returns:       (t_loss, r_loss) — both scalar tensors, averaged over pairs.
    """
    N = len(pred_ned_poses)
    assert N == len(p_targets) == len(q_targets) and N >= 2

    device = pred_ned_poses[0].device
    t_loss = torch.tensor(0.0, device=device)
    r_loss = torch.tensor(0.0, device=device)

    for i in range(1, N):
        pose_prev = pred_ned_poses[i - 1]
        pose_curr = pred_ned_poses[i]

        # Translation delta
        t_loss = t_loss + F.l1_loss(
            pose_curr[:3] - pose_prev[:3],
            p_targets[i] - p_targets[i - 1],
        )

        # Rotation delta: q_rel = q_prev^-1 * q_curr
        q_p_inv = quat_inv(pose_prev[3:7].clone())
        q_rel_pred = quat_mul(q_p_inv, pose_curr[3:7])

        q_g_inv = quat_inv(q_targets[i - 1].clone())
        q_rel_target = quat_mul(q_g_inv, q_targets[i])

        r_loss = r_loss + rotation_geodesic_loss(q_rel_pred, q_rel_target)

    n_pairs = N - 1
    return t_loss / n_pairs, r_loss / n_pairs


def origin_anchor_loss(pred_ned_pose_0):
    """L1 between frame-0 prediction position and the origin.

    Frame 0 is the anchor of the relative coordinate system; both grounders pin
    its predicted NED translation to (0,0,0) so the rest of the trajectory is
    interpreted in frame-0-local coordinates.
    """
    p0_t = pred_ned_pose_0[:3]
    return F.l1_loss(p0_t, torch.zeros(3, device=p0_t.device, dtype=p0_t.dtype))


def focal_length_loss(pred_pose_enc, target_K, image_hw):
    """L1 supervision on the FoV channels of VGGT's pose_enc.

    Mirrors VGGT's training-time `loss_FL` component (vggt/training/loss.py).
    The FoV channels of pose_enc (last 2 dims) are derived from the focal
    length as `fov_h = 2*atan((H/2)/fy)`, `fov_w = 2*atan((W/2)/fx)`. We
    convert `target_K` the same way and L1-compare against the prediction.

    pred_pose_enc: [N, 9]  — VGGT's pose_enc (T, quat, fov_h, fov_w)
    target_K:      [N, 3, 3] or [3, 3] (broadcast)
    image_hw:      (H, W)

    Returns: scalar torch.Tensor (0-dim).
    """
    H, W = image_hw
    fy = target_K[..., 1, 1]
    fx = target_K[..., 0, 0]
    target_fov_h = 2.0 * torch.atan((H / 2.0) / fy)
    target_fov_w = 2.0 * torch.atan((W / 2.0) / fx)
    target_fov = torch.stack([target_fov_h, target_fov_w], dim=-1)   # [..., 2]
    return F.l1_loss(pred_pose_enc[..., 7:], target_fov)


def _project_target_to_source(target_wp_flat, extr_s, intr_s, H, W):
    """Project a target frame's world points into a source camera.

    Numerically robust: clamps Z>1e-3 before division so points behind / on
    the source's principal plane produce finite pixel coords with zero
    gradient (clamp's gradient is 0 in the clamped region). The validity
    mask then drops them from the loss.

    Args:
      target_wp_flat: [H*W, 3] world coords (target's per-pixel points)
      extr_s:         [1, 3, 4] source's world->cam extrinsic
      intr_s:         [1, 3, 3] source's intrinsic
      H, W:           image dims

    Returns:
      grid:   [1, H, W, 2] sampling grid in normalized [-1, 1] coords
      valid:  [H, W] bool — projected pixel is in-frame AND z_raw > 1e-3
      z_raw:  [H, W] target's depth in the source's camera frame
              (un-clamped; useful for depth-consistency loss)
    """
    _, cam = project_world_points_to_cam(
        target_wp_flat, extr_s, intr_s, only_points_cam=True,
    )
    # cam: [1, 3, H*W]
    z_raw = cam[0, 2]                                       # [H*W]
    valid_z = z_raw > 1e-3                                   # in front of source

    z_safe = z_raw.clamp(min=1e-3)
    xy_ndc = cam[:, :2, :] / z_safe.unsqueeze(0).unsqueeze(0)            # [1, 2, H*W]
    xy_h   = torch.cat([xy_ndc, torch.ones_like(xy_ndc[:, :1, :])], dim=1)
    pix    = torch.bmm(intr_s, xy_h)[:, :2, :]                           # [1, 2, H*W]

    u = pix[0, 0].reshape(H, W)
    v = pix[0, 1].reshape(H, W)
    grid_u = 2.0 * u / max(W - 1, 1) - 1.0
    grid_v = 2.0 * v / max(H - 1, 1) - 1.0
    valid_uv = (grid_u.abs() <= 1.0) & (grid_v.abs() <= 1.0)
    valid = valid_z.reshape(H, W) & valid_uv                              # [H, W] bool
    grid  = torch.stack([grid_u, grid_v], dim=-1).unsqueeze(0)            # [1, H, W, 2]
    return grid, valid, z_raw.reshape(H, W)


def _aggregate_per_target(per_source_errs, H, W, device, *, min_projection):
    """Aggregate per-source error maps into one per-target scalar.

    per_source_errs: list of (err [H,W], valid [H,W]) tuples
    Returns: scalar tensor (mean over valid pixels of the chosen reduction).
    """
    if not per_source_errs:
        return torch.tensor(0.0, device=device)

    if min_projection:
        err_stack   = torch.stack([e for e, _ in per_source_errs], dim=0)
        valid_stack = torch.stack([v for _, v in per_source_errs], dim=0)
        err_masked  = err_stack.masked_fill(~valid_stack, float("inf"))
        min_err, _  = err_masked.min(dim=0)                              # [H, W]
        any_valid   = valid_stack.any(dim=0)
        min_err     = torch.where(any_valid, min_err, torch.zeros_like(min_err))
        denom       = any_valid.float().sum().clamp(min=1.0)
        return min_err.sum() / denom
    err_sum = torch.zeros(H, W, device=device)
    count   = torch.zeros(H, W, device=device)
    for err, valid in per_source_errs:
        err_sum = err_sum + err * valid.float()
        count   = count + valid.float()
    mean_err  = err_sum / count.clamp(min=1.0)
    any_valid = count > 0
    denom     = any_valid.float().sum().clamp(min=1.0)
    return (mean_err * any_valid.float()).sum() / denom



def depth_consistency_loss(depths, world_points, extrinsics, intrinsics,
                           *, min_projection=True, padding_mode="zeros"):
    """Cross-view depth consistency loss (absolute, metric-coupled).

    For every ordered pair (target=t, source=s) with s != t:
      1. Project target's world points into source's camera frame.
      2. The Z-component of those projected points = the depth the source
         camera SHOULD see at each projected pixel ("expected_depth").
      3. Bilinear-sample source's predicted depth at the projected pixels
         ("sampled_depth").
      4. err_st = |expected_depth - sampled_depth|.

    Absolute (not scale-invariant) by design. When this loss is combined
    with the metric IMU pose-supervision term (TTRGrounder), the pose
    supervision pulls the foundation model's pose_enc toward metric units; the absolute
    depth-consistency loss then couples VGGT's depth predictions into the
    same metric scale. A scale-invariant `|Δz| / z` variant would discard
    that information and let depth and pose drift on different scales.

    No GT depth is consumed anywhere; this is fully self-supervised on
    VGGT's own predicted depths and pose.

    Shapes:
      depths:        [N, H, W]      (VGGT's predicted depth per frame)
      world_points:  [N, H, W, 3]   (cam-0 world frame)
      extrinsics:    [N, 3, 4]      world->cam (OpenCV)
      intrinsics:    [N, 3, 3]

    Returns: scalar torch.Tensor (0-dim).
    """
    N, H, W = depths.shape
    device = depths.device

    depths       = depths.float()
    world_points = world_points.float()
    extrinsics   = extrinsics.float()
    intrinsics   = intrinsics.float()

    per_target_losses = []
    for t in range(N):
        target_wp = world_points[t].reshape(-1, 3)             # [H*W, 3]

        per_source_errs = []
        for s in range(N):
            if s == t:
                continue
            grid, valid, expected = _project_target_to_source(
                target_wp, extrinsics[s:s + 1], intrinsics[s:s + 1], H, W,
            )
            # Sample source's depth at the projected target pixels
            sampled = F.grid_sample(
                depths[s:s + 1].unsqueeze(1), grid,           # [1, 1, H, W]
                mode="bilinear", padding_mode=padding_mode, align_corners=True,
            )[0, 0]                                            # [H, W]

            err = (expected - sampled).abs()                  # [H, W]
            per_source_errs.append((err, valid))

        per_target_losses.append(_aggregate_per_target(
            per_source_errs, H, W, device, min_projection=min_projection,
        ))

    if not per_target_losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(per_target_losses).mean()
