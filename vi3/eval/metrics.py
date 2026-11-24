"""Per-frame and aggregated pose / depth / scale metrics."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from vi3.transforms import pose_enc_to_target_pose


# Valid-depth filter: drop padded pixels (== 0) and TartanAir's sky/infinity
# sentinel (~16384). MAX_VALID_DEPTH chosen well above realistic indoor depths
# but well below the sentinel; matches typical KITTI / NYU practice.
MIN_VALID_DEPTH = 0.01    # m
MAX_VALID_DEPTH = 80.0    # m

# Below this GT per-frame translation, trajectory scale (||t_pred||/||t_gt||) is
# ill-posed (noise/noise): the frame is excluded from the scale metric (NaN,
# nan-aware aggregation).
SCALE_MIN_GT_MOTION = 0.01    # m


def depth_metrics_full(pred_depth, gt_depth_raw):
    """Per-frame depth metrics on valid pixels; all NaN when GT depth is
    unavailable or sparse (< 100 valid pixels).

        depth_absrel     — scale-aligned AbsRel, mean(|s·pred − gt| / gt), with s
                           the median-of-ratios scale (shape only).
        depth_d1         — scale-aligned δ<1.25: fraction of pixels with
                           max(s·pred/gt, gt/s·pred) < 1.25.
        depth_absrel_abs — metric AbsRel, mean(|pred − gt| / gt), no alignment.
        depth_d1_abs     — metric δ<1.25, no alignment: an up-to-scale model
                           scores ~0 until it is given metric scale.
    """
    nan = float("nan")
    keys = ("depth_absrel", "depth_d1", "depth_absrel_abs", "depth_d1_abs")
    if gt_depth_raw is None:
        return dict.fromkeys(keys, nan)
    pred = pred_depth.squeeze().float()
    gt   = gt_depth_raw.squeeze().float().to(pred.device)
    valid = (gt > MIN_VALID_DEPTH) & (gt < MAX_VALID_DEPTH)
    if int(valid.sum()) < 100:
        return dict.fromkeys(keys, nan)
    p_v = pred[valid]
    g_v = gt[valid]
    s = (g_v / (p_v + 1e-12)).median().item()
    sp = s * p_v
    absrel = float(((sp - g_v).abs() / (g_v + 1e-12)).mean().item())
    ratio  = torch.maximum(sp / (g_v + 1e-12), g_v / (sp + 1e-12))
    d1     = float((ratio < 1.25).float().mean().item())
    absrel_abs = float(((p_v - g_v).abs() / (g_v + 1e-12)).mean().item())
    ratio_abs  = torch.maximum(p_v / (g_v + 1e-12), g_v / (p_v + 1e-12))
    d1_abs     = float((ratio_abs < 1.25).float().mean().item())
    return dict(depth_absrel=absrel, depth_d1=d1,
                depth_absrel_abs=absrel_abs, depth_d1_abs=d1_abs)


def calculate_metrics(pred_pose_raw, pred_depth, gt_pose_vec, gt_depth_raw):
    """Per-frame pose + depth + scale metrics.

    pred_pose_raw: torch.Tensor [>=7] — VGGT pose_enc (extrinsic, OpenCV).
    pred_depth:    torch.Tensor [H, W] — predicted depth (units / scale free).
    gt_pose_vec:   torch.Tensor [7] — frame-0-relative target [t(3), q(4)].
    gt_depth_raw:  torch.Tensor [H, W] or None — GT metric depth. If None
                    (datasets without depth GT, e.g. EuRoC), the depth
                    metric is reported as NaN.

    Returns the translation / rotation error, the signed per-axis values (pred
    and GT, for the 3D view's plots), the pose scale and the depth metrics of
    `depth_metrics_full`.
    """
    pred_target_pose = pose_enc_to_target_pose(pred_pose_raw)

    # Translation
    t_p = pred_target_pose[:3].cpu().numpy()
    t_g = gt_pose_vec[:3].cpu().numpy()
    t_diff = t_p - t_g

    # Rotation (geodesic + per-axis Euler signed error)
    q_p = pred_target_pose[3:].cpu().numpy()
    q_g = gt_pose_vec[3:].cpu().numpy()
    dot = np.clip(np.abs(np.sum(q_p * q_g)), -1.0, 1.0)
    geodesic_err = np.degrees(2 * np.arccos(dot))

    r_p = R.from_quat(q_p).as_euler("xyz", degrees=True)
    r_g = R.from_quat(q_g).as_euler("xyz", degrees=True)

    # Pose scale (predicted translation magnitude vs GT); NaN when the GT frame
    # has barely moved, so the frame drops out of the nan-aware aggregation.
    dist_gt = np.linalg.norm(t_g)
    dist_pred = np.linalg.norm(t_p)
    scale = dist_pred / (dist_gt + 1e-8) if dist_gt > SCALE_MIN_GT_MOTION else float("nan")

    dm = depth_metrics_full(pred_depth, gt_depth_raw)

    return {
        "p_total": np.linalg.norm(t_diff),
        "p_x": t_p[0], "p_x_gt": t_g[0],
        "p_y": t_p[1], "p_y_gt": t_g[1],
        "p_z": t_p[2], "p_z_gt": t_g[2],
        "q_total": geodesic_err,
        "q_roll":  r_p[0], "q_roll_gt":  r_g[0],
        "q_pitch": r_p[1], "q_pitch_gt": r_g[1],
        "q_yaw":   r_p[2], "q_yaw_gt":   r_g[2],
        "scale": scale,
        **dm,
    }


def window_alignment_metrics(pred_poses, gt_poses):
    """Scale-invariant trajectory metrics over one window.

    pred_poses, gt_poses: [N, 7] arrays of camera->world [t(3), q(4) xyzw], both
    frame-0-relative. Returns:
      sim3_trans_e — RMSE of camera centres after a Sim(3) (Umeyama) fit of the
                     prediction to GT (metres; shape only, scale removed),
      rpe_rot_e    — mean consecutive-pair relative-rotation error (deg),
      rpe_dir_e    — mean consecutive-pair relative-translation direction error
                     (deg; scale-free).
    Window-level quantities are NaN when GT barely moves (alignment undefined).
    """
    from vi3.utils import procrustes_alignment

    nan = float("nan")
    out = dict(sim3_trans_e=nan, rpe_rot_e=nan, rpe_dir_e=nan)
    P = np.asarray(pred_poses, dtype=np.float64)
    G = np.asarray(gt_poses, dtype=np.float64)
    Rp = R.from_quat(P[:, 3:7])
    Rg = R.from_quat(G[:, 3:7])

    # Relative pose errors between consecutive frames (scale-free).
    rot_e, dir_e = [], []
    for i in range(len(P) - 1):
        dRp = Rp[i].inv() * Rp[i + 1]
        dRg = Rg[i].inv() * Rg[i + 1]
        rot_e.append(np.degrees((dRg.inv() * dRp).magnitude()))
        tp = Rp[i].inv().apply(P[i + 1, :3] - P[i, :3])
        tg = Rg[i].inv().apply(G[i + 1, :3] - G[i, :3])
        if np.linalg.norm(tg) > SCALE_MIN_GT_MOTION and np.linalg.norm(tp) > 1e-9:
            c = np.dot(tp, tg) / (np.linalg.norm(tp) * np.linalg.norm(tg))
            dir_e.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    if rot_e:
        out["rpe_rot_e"] = float(np.mean(rot_e))
    if dir_e:
        out["rpe_dir_e"] = float(np.mean(dir_e))

    # Sim(3) alignment of camera centres (needs GT motion to be defined).
    spread = np.linalg.norm(G[:, :3] - G[:, :3].mean(0), axis=1).max()
    if spread > SCALE_MIN_GT_MOTION:
        src = torch.from_numpy(P[:, :3].T.copy())
        tgt = torch.from_numpy(G[:, :3].T.copy())
        aligned = procrustes_alignment(src, tgt)[0]
        res = (aligned - tgt).numpy()
        out["sim3_trans_e"] = float(np.sqrt(np.mean(np.sum(res ** 2, axis=0))))
    return out


def aggregate_sequence_metrics(all_frame_metrics):
    """Aggregate per-frame metrics over a window: translation RMSE, mean
    rotation error, and nan-aware means of the per-axis values, scale and depth
    (scale is NaN on near-stationary frames, depth where GT depth is missing)."""
    summary = {}
    for k in ["p_x", "p_y", "p_z", "q_roll", "q_pitch", "q_yaw"]:
        summary[k] = np.mean([f[k] for f in all_frame_metrics])
        summary[f"{k}_gt"] = np.mean([f[f"{k}_gt"] for f in all_frame_metrics])
    for k in ["scale", "depth_absrel", "depth_d1", "depth_absrel_abs", "depth_d1_abs"]:
        vals = [f[k] for f in all_frame_metrics]
        summary[k] = float(np.nanmean(vals)) if not all(
            isinstance(v, float) and np.isnan(v) for v in vals) else float("nan")
    p_errors = np.array([f["p_total"] for f in all_frame_metrics])
    summary["p_total"] = float(np.sqrt(np.mean(p_errors ** 2)))
    q_errors = np.array([f["q_total"] for f in all_frame_metrics])
    summary["q_total"] = float(np.mean(q_errors))
    return summary
