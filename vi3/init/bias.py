"""Closed-form gyro-bias initializer (GT-free).

`estimate_gyro_bias_window` recovers a constant gyro bias (rad/s) for the
window from the IMU gyro stream and the per-frame relative rotations, so the
grounder starts from a near-correct bias instead of zero (bias=0 corrupts the
gyro-integrated rotation target). The relative rotations come from the frozen
foundation model's feed-forward poses (`predicted_poses`), which keeps the estimate
GT-free — the whole point of the `vel` path. Passing GT poses instead makes it
near-exact (calibration-grade) and is available for validation.

Math (arithmetic-mean approximation, per consecutive frame pair i→j):
    ω̄            = (1/L) Σ_k ω̃_k                (per-sample mean gyro)
    Δt           = mean sampling interval
    R_pred_step  = Exp(ω̄ · Δt)                  (per-sample step, bg = 0)
    R_gt_step    = Exp(Log(R_ij) / L)           (per-sample step from R_ij)
    bg           = -(1/Δt) · Log(R_pred_step^T · R_gt_step)
Averaged over all consecutive pairs in the window.
"""

import torch
import pypose as pp

from vggt.utils.rotation import quat_to_mat


def estimate_gyro_bias_pair(gyro_samples, dts, R_ij):
    """One frame-pair gyro-bias estimate.

    Args:
        gyro_samples: [L, 3] raw gyro samples in the integration window between
            two consecutive frames. Pre-bias (no subtraction).
        dts: [L] per-sample dt; mean(dts) gives Δt.
        R_ij: [3, 3] relative rotation, frame_j ← frame_i, in the same axes as
            the gyro (cam axes in our pipeline).

    Returns:
        bg: [3] tensor — closed-form gyro-bias estimate from this pair.
    """
    L = gyro_samples.shape[0]
    omega_bar = gyro_samples.mean(dim=0)                        # [3]
    dt = dts.mean()                                            # scalar, per-sample

    # Per-sample predicted rotation step (bg = 0).
    R_pred = pp.so3((omega_bar * dt).unsqueeze(0)).Exp()       # SO3, [1] batch

    # Per-sample rotation step from R_ij = Exp(Log(R_ij) / L).
    # Re-orthonormalize R_ij first: it comes from the foundation model's predicted poses,
    # whose bf16 forward + SE3 rebasing can leave it slightly non-orthogonal
    # (enough to trip pypose's strict mat2SO3 validation on some foundation models).
    # Nearest-rotation via SVD (det-fix to stay a proper rotation, not a
    # reflection). A no-op for already-orthogonal inputs (e.g. Pi3X).
    U, _, Vh = torch.linalg.svd(R_ij.double())
    R_ortho = U @ Vh
    if torch.det(R_ortho) < 0:                                 # reflection -> flip
        U = U.clone(); U[:, -1] = -U[:, -1]
        R_ortho = U @ Vh
    R_ij_so3 = pp.mat2SO3(R_ortho.to(R_ij.dtype).unsqueeze(0))  # SO3, [1] batch
    log_R_ij = R_ij_so3.Log().squeeze(0)                       # [3] so(3) tangent
    R_gt_step = pp.so3((log_R_ij / L).unsqueeze(0)).Exp()      # SO3, [1] batch

    # Per-sample SO(3) residual: predicted^T · target-step.
    R_residual = R_pred.Inv() @ R_gt_step                     # SO3
    log_residual = R_residual.Log().squeeze(0)                # [3] so(3)

    return -log_residual / dt                                  # [3] rad/s


def estimate_gyro_bias_window(samples, device, predicted_poses=None):
    """Average gyro-bias estimate across all consecutive frame pairs.

    Args:
        samples: list of N dicts; each has `imu_gyro` [L_i, 3] and `imu_dts`
            [L_i]. `samples[i]["gt_pose"]` is read for R_ij only when
            `predicted_poses` is None (the GT-source path).
        device: where to assemble the result.
        predicted_poses: optional list of N [7] tensors in the same
            `[tx, ty, tz, qx, qy, qz, qw]` convention as `gt_pose` (cam→world in
            cam axes). When supplied, R_ij is derived from these instead of GT
            poses, making the estimator GT-free — pass the un-adapted model's
            pose_enc converted via `pose_enc_to_target_pose_torch`.

            Model R_ij carry the foundation model's rotation error, so the
            estimate is a warm start for the bias rather than an exact value.

    Returns:
        bg_hat: [3] tensor on `device`. Averages pairs (0,1), (1,2), …; frame
            0's IMU window (the warmup) is included via pair (0,1).
    """
    estimates = []
    for j in range(1, len(samples)):
        i = j - 1
        gyro_j = samples[j]["imu_gyro"].to(device)             # [L, 3]
        dts_j  = samples[j]["imu_dts"].to(device)              # [L]
        if gyro_j.shape[0] == 0 or float(dts_j.sum()) <= 0:
            continue

        # R_ij: rotation from frame_i's axes onto frame_j's axes. Both poses are
        # cam→world in cam axes, so R_ij = R_i^T @ R_j.
        if predicted_poses is None:
            q_i = samples[i]["gt_pose"][3:7].to(device)        # xyzw
            q_j = samples[j]["gt_pose"][3:7].to(device)
        else:
            q_i = predicted_poses[i][3:7].to(device)
            q_j = predicted_poses[j][3:7].to(device)
        R_i = quat_to_mat(q_i / (torch.norm(q_i) + 1e-12))     # [3, 3]
        R_j = quat_to_mat(q_j / (torch.norm(q_j) + 1e-12))     # [3, 3]
        R_ij = R_i.transpose(-1, -2) @ R_j                     # [3, 3]

        estimates.append(estimate_gyro_bias_pair(gyro_j, dts_j, R_ij))

    if not estimates:
        return torch.zeros(3, device=device)
    return torch.stack(estimates, dim=0).mean(dim=0)
