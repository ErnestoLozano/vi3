"""Rerun 3D + 2D + scalar visualization of one window (`scripts/run.py --viz`).

One `RerunSession` per run. Callers feed in:
    * dataset samples for the window (GT pose, GT depth, K, image)
    * the model outputs per method (pose_enc, depth, depth_conf)
    * per-frame metric dicts per method (signed-axis errors)

All 3D entities live in a single world frame: the TartanAir cam-0 world
frame (FRD: x-forward, y-right, z-down). Conversions:
    * VGGT pose_enc -> camera pose in TartanAir cam-0:  `pose_enc_to_target_pose`
    * VGGT world_points (OpenCV cam-0) -> TartanAir cam-0: `wp @ M.T`

Lazy `import rerun as rr` inside `__init__` so the rest of the codebase
works without the optional `rerun-sdk` dependency.
"""

import numpy as np
import torch

from vi3.transforms import M_change_of_basis, pose_enc_to_target_pose


# ---------------------------------------------------------------------------
# Small helpers

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _img_chw_to_hwc_u8(image_chw):
    """[3, H, W] float in [0, 1] -> [H, W, 3] uint8."""
    img = _to_np(image_chw)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _vggt_world_points_to_tartanair(wp_opencv):
    """[N, H, W, 3] OpenCV cam-0 -> [N, H, W, 3] TartanAir cam-0.

    Basis change: x_t = M @ x_o, applied per point (broadcasting via @M.T).
    """
    wp = _to_np(wp_opencv)
    M = _to_np(M_change_of_basis())
    return wp @ M.T


BASELINE_RED = (220, 50, 50)      # the frozen 3D foundation model
GROUNDED_GREEN = (50, 210, 90)    # any grounding method
IMU_BLUE = (70, 130, 240)         # the IMU target trajectory


def gt_depth_colors(sample, image_hw, subsample):
    """Source-image colors of the GT-depth points of `_gt_depth_to_pointcloud`
    (same subsample and validity mask, 0 < depth < 80), aligned 1:1."""
    H, W = image_hw
    dmap = _to_np(sample["gt_depth"]).reshape(H, W)
    img_hwc = _img_chw_to_hwc_u8(sample["image"])                 # [H, W, 3] u8
    yy, xx = np.meshgrid(np.arange(0, H, subsample), np.arange(0, W, subsample),
                         indexing="ij")
    z = dmap[yy, xx]
    return img_hwc[yy, xx][(z > 1e-3) & (z < 80.0)]


def _gt_depth_to_pointcloud(depth_hw, K_3x3, image_hw, p_cw_tartan, q_cw_tartan,
                            subsample=4, max_depth=80.0):
    """Unproject a GT depth map into the cam-0 world frame (TartanAir conv).

    GT depth is metric and in the same camera convention as the dataset
    (TartanAir FRD: x-forward = depth axis). Pixels with depth outside
    (0, max_depth) are dropped (sky sentinel removal).
    """
    H, W = image_hw
    depth = _to_np(depth_hw).reshape(H, W)
    K = _to_np(K_3x3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Subsample on the image grid for tractable point counts.
    ys = np.arange(0, H, subsample)
    xs = np.arange(0, W, subsample)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    z = depth[yy, xx]
    valid = (z > 1e-3) & (z < max_depth)

    # Pinhole in OpenCV convention: x_right = (u - cx) * z / fx ; y_down = (v - cy) * z / fy ;
    # but TartanAir camera axes are (forward, right, down), so depth is along
    # the cam-x axis. Project as (x_forward = z_depth, y_right = (u-cx)*z/fx,
    # z_down = (v-cy)*z/fy).
    u_right = (xx - cx) * z / fx
    v_down  = (yy - cy) * z / fy

    cam_pts = np.stack([z, u_right, v_down], axis=-1)  # TartanAir FRD
    cam_pts = cam_pts[valid]                            # [P, 3]

    # cam->world via the camera pose (camera-to-world in TartanAir basis).
    p = _to_np(p_cw_tartan).reshape(3)
    q = _to_np(q_cw_tartan).reshape(4)
    R_cw = _quat_xyzw_to_mat(q)
    world_pts = cam_pts @ R_cw.T + p
    return world_pts


def _quat_xyzw_to_mat(q):
    """[xyzw] -> 3x3 rotation. Avoids the scipy dependency for hot paths."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ])


# ---------------------------------------------------------------------------
# RerunSession

class RerunSession:
    """One session = one Rerun recording of a run."""

    def __init__(self, app_name="vi3", spawn=True, recording_id=None):
        try:
            import rerun as rr
        except ImportError as e:
            raise ImportError(
                "rerun-sdk is required for visualization. Install with "
                "`pip install rerun-sdk` (>=0.20)."
            ) from e
        self._rr = rr
        rr.init(app_name, recording_id=recording_id, spawn=spawn)

        # World-root axis convention: FRD (forward-right-down) — matches
        # TartanAir's camera convention so cam-to-world poses are read
        # directly without further axis swapping.
        rr.log("/", rr.ViewCoordinates.FRD, static=True)

    # -----------------------------------------------------------------
    # Static GT scene
    # -----------------------------------------------------------------
    def log_gt_static(self, samples, image_hw, *, log_depth_pointcloud=True,
                       depth_subsample=4, depth_radii=0.02, depth_color="rgb",
                       image_plane_distance=0.4, axis_length=0.05):
        """Log GT camera frusta, RGB images, and (optionally) GT-depth point
        clouds for every context frame, in the cam-0 world.

        Uses `s["gt_pose"]` already in TartanAir cam->world coords; we anchor
        to frame 0 with `poses_relative_to_frame0` so the world origin
        coincides with the predictions' origin.

        depth_color: "rgb" (default) colors the GT-depth cloud by the source
                     image — the photo-realistic metric reference; "solid" uses
                     the GT color.
        """
        from vi3.data.preprocessing import poses_relative_to_frame0

        rr = self._rr
        H, W = image_hw
        device = samples[0]["gt_pose"].device
        gt_poses = [s["gt_pose"] for s in samples]
        p_rel, q_rel = poses_relative_to_frame0(gt_poses, device=device)

        centers = []
        for i, s in enumerate(samples):
            base = f"/scene/gt/cam_{i:03d}"
            p_i = _to_np(p_rel[i])
            q_i = _to_np(q_rel[i])
            K   = _to_np(s["K"])
            centers.append(p_i)

            rr.log(base,
                   rr.Transform3D(
                       translation=p_i,
                       quaternion=np.asarray(q_i, dtype=np.float32),   # xyzw
                       axis_length=axis_length,
                   ),
                   static=True)
            # The full-RGB image quad on the GT frusta is the main thing that
            # occludes the point cloud in a multi-frame window. Gate both the
            # Pinhole plane and its texture on image_plane_distance>0.
            if image_plane_distance > 0:
                rr.log(f"{base}/image",
                       rr.Pinhole(image_from_camera=K, resolution=(W, H),
                                  camera_xyz=rr.ViewCoordinates.FRD,
                                  image_plane_distance=image_plane_distance),
                       static=True)
                rr.log(f"{base}/image",
                       rr.Image(_img_chw_to_hwc_u8(s["image"])).compress(jpeg_quality=85),
                       static=True)

            if log_depth_pointcloud and "gt_depth" in s:
                pts = _gt_depth_to_pointcloud(
                    s["gt_depth"], s["K"], image_hw,
                    p_rel[i], q_rel[i], subsample=depth_subsample,
                )
                cols = (gt_depth_colors(s, image_hw, depth_subsample)
                        if depth_color == "rgb" else self._COLORS["gt"])
                rr.log(f"/scene/gt/points/cam_{i:03d}",
                       rr.Points3D(pts, radii=float(depth_radii), colors=cols),
                       static=True)

        self._log_trajectory("gt", centers, radii=0.015)

    # -----------------------------------------------------------------
    # Per-method 3D + 2D
    # -----------------------------------------------------------------
    # GT in its real colors (white trajectory), the frozen 3DFM in red, every
    # grounding method in green, the IMU trajectory in blue.
    _COLORS = {
        "gt":            (245, 245, 245),
        "baseline":      BASELINE_RED,
        "imu_target":    IMU_BLUE,
        "imu_dense":     (90, 150, 255),
    }

    def _color_for(self, label):
        return self._COLORS.get(label, GROUNDED_GREEN)

    def _log_trajectory(self, label, centers, *, radii=0.01):
        """Polyline through `centers` (list/array of 3-vec) colored by method.

        Logged once, static, under /scene/{label}/trajectory.
        """
        centers = np.asarray([_to_np(c).reshape(3) for c in centers])
        if len(centers) < 2:
            return
        color = self._color_for(label)
        self._rr.log(
            f"/scene/{label}/trajectory",
            self._rr.LineStrips3D([centers], radii=radii, colors=[color]),
            static=True,
        )

    def log_method_3d(self, label, pred_pose_enc, world_points, samples, image_hw,
                       *, point_subsample=4, cloud_color="solid", point_radii=0.01,
                       image_plane_distance=0.4, axis_length=0.05):
        """Log predicted camera frusta + dense world-points for one method.

        pred_pose_enc: [N, 9] VGGT pose_enc (OpenCV extrinsics).
        world_points:  [N, H, W, 3] VGGT world_points (OpenCV cam-0).
        cloud_color:   "solid" (default) colors the dense points by the method
                       color (baseline red, grounding methods green); "rgb" by
                       the source image. The trajectory polyline always uses
                       the method color.
        point_radii:   per-point sphere radius (m). Use a smaller value for
                       solid reference shells (GT/baseline) so they read crisp
                       against a dense RGB cloud.
        """
        rr = self._rr
        H, W = image_hw
        color = self._color_for(label)

        # Camera frusta — convert each frame to TartanAir cam->world.
        pred_pose_enc = pred_pose_enc.detach().float().cpu()
        centers = []
        for i in range(pred_pose_enc.shape[0]):
            cam_pose = _to_np(pose_enc_to_target_pose(pred_pose_enc[i]))
            p_i, q_i = cam_pose[:3], cam_pose[3:7]
            centers.append(p_i)
            base = f"/scene/{label}/cam_{i:03d}"
            rr.log(base,
                   rr.Transform3D(
                       translation=p_i,
                       quaternion=np.asarray(q_i, dtype=np.float32),   # xyzw
                       axis_length=axis_length,
                   ),
                   static=True)
            # Frustum / image plane. `image_plane_distance<=0` skips the Pinhole
            # entirely so nothing is drawn in front of the point cloud (the cloud
            # is the hero for scale teasers); >0 draws a small frustum at that
            # distance in scene metres.
            if image_plane_distance > 0:
                K = _to_np(samples[i]["K"])
                rr.log(f"{base}/image",
                       rr.Pinhole(image_from_camera=K, resolution=(W, H),
                                  camera_xyz=rr.ViewCoordinates.FRD,
                                  image_plane_distance=image_plane_distance),
                       static=True)

        self._log_trajectory(label, centers)

        # Dense world points (OpenCV cam-0 -> TartanAir cam-0).
        wp = _vggt_world_points_to_tartanair(world_points)  # [N, H, W, 3]
        s = int(point_subsample)
        wp_sub = wp[:, ::s, ::s, :].reshape(-1, 3)

        # Cloud color: per-pixel source RGB ("rgb") or a solid method color
        # ("solid", e.g. baseline in red so it reads as one object).
        if cloud_color == "solid":
            cols = np.tile(np.asarray(color, dtype=np.uint8), (wp_sub.shape[0], 1))
        else:
            imgs = np.stack(
                [_img_chw_to_hwc_u8(samples[i]["image"]) for i in range(len(samples))],
                axis=0,
            )                                                       # [N, H, W, 3] uint8
            cols = imgs[:, ::s, ::s, :].reshape(-1, 3)

        rr.log(f"/scene/{label}/points",
               rr.Points3D(wp_sub, radii=float(point_radii), colors=cols),
               static=True)

    def log_method_2d(self, label, depth, depth_conf, image_hw):
        """Log VGGT's predicted depth + confidence maps as 2D images per frame.

        Path layout: `/images/cam_{i}/{label}/depth` and `.../depth_conf`.
        Grouping by camera (not by method) lets one space-view per camera
        show every method's predictions side by side.

        Both go through `DepthImage` so Rerun applies a consistent colormap.

        depth:      [N, H, W] or [N, H, W, 1]
        depth_conf: [N, H, W]
        """
        rr = self._rr
        depth = _to_np(depth)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth_conf = _to_np(depth_conf)

        N = depth.shape[0]
        for i in range(N):
            rr.log(f"/images/cam_{i:03d}/{label}/depth",
                   rr.DepthImage(depth[i].astype(np.float32), meter=1.0),
                   static=True)
            rr.log(f"/images/cam_{i:03d}/{label}/depth_conf",
                   rr.DepthImage(depth_conf[i].astype(np.float32), meter=1.0),
                   static=True)

    # -----------------------------------------------------------------
    # Scalars: per-frame errors
    # -----------------------------------------------------------------
    def log_per_frame_errors(self, label, per_frame_metrics):
        """Log signed per-axis position / rotation errors indexed by `frame`.

        Path is `/scalars/frame_err/{key}/{label}` — one plot per error key,
        methods overlaid.

        per_frame_metrics: list of dicts produced by `calculate_metrics`.
        """
        rr = self._rr
        keys_signed = [
            ("p_x", "p_x_gt"), ("p_y", "p_y_gt"), ("p_z", "p_z_gt"),
            ("q_roll", "q_roll_gt"), ("q_pitch", "q_pitch_gt"), ("q_yaw", "q_yaw_gt"),
        ]
        keys_scalar = ["p_total", "q_total", "depth_absrel", "scale"]

        for i, m in enumerate(per_frame_metrics):
            rr.set_time("frame", sequence=i)
            for pred_k, gt_k in keys_signed:
                rr.log(f"/scalars/frame_err/{pred_k}/{label}",
                       rr.Scalars(float(m[pred_k]) - float(m[gt_k])))
            for k in keys_scalar:
                v = m[k]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                rr.log(f"/scalars/frame_err/{k}/{label}", rr.Scalars(float(v)))

    # -----------------------------------------------------------------
    # IMU-only: the integrated target trajectory
    # -----------------------------------------------------------------
    def log_imu_trajectory(self, p_imu, q_imu, samples, image_hw,
                            label="imu_target"):
        """Log the IMU-integrated frusta (keyframes) + trajectory polyline."""
        rr = self._rr
        H, W = image_hw
        centers = []
        for i in range(len(p_imu)):
            base = f"/scene/{label}/cam_{i:03d}"
            p_np = _to_np(p_imu[i]).reshape(3)
            centers.append(p_np)
            rr.log(base,
                   rr.Transform3D(
                       translation=p_np,
                       quaternion=np.asarray(_to_np(q_imu[i]), dtype=np.float32),   # xyzw
                   ),
                   static=True)
            K = _to_np(samples[i]["K"])
            rr.log(f"{base}/image",
                   rr.Pinhole(image_from_camera=K, resolution=(W, H),
                              camera_xyz=rr.ViewCoordinates.FRD,
                              image_plane_distance=0.25),
                   static=True)

        self._log_trajectory(label, centers, radii=0.012)

    def log_imu_dense_path(self, samples, grounder, label="imu_dense"):
        """Reconstruct and log the per-IMU-sample integrated path.

        Reuses the IMU integrator with the grounder's current biases. Composes
        each segment's intermediate `pre_paths` into the cam-0 world via the
        same `v_i*t + 0.5*g*t^2 + R_i*delta_p` rule the grounder uses for
        keyframes, but at every IMU sample inside each inter-frame window.

        Use this after `train()` so the dense path reflects the
        init-estimated (fixed) biases.
        """
        import pypose as pp
        from vggt.utils.rotation import quat_to_mat

        from vi3.init.preintegration import IMUPreIntegrator

        device = samples[0]["gt_pose"].device
        # Read gravity from the dataset's world-frame convention (NED for
        # TartanAir, ENU for EuRoC, ...) — see `grounder.dataset.GRAVITY_WORLD`.
        gravity = torch.tensor(grounder.dataset.GRAVITY_WORLD,
                               device=device, dtype=torch.float32)

        init = grounder.init                     # the window's IMUInit
        ba, bg = init.ba, init.bg
        pre = IMUPreIntegrator(acc_density=init.acc_noise_density,
                               gyro_density=init.gyro_noise_density)

        q0_world = samples[0]["gt_pose"][3:7].to(device)
        R0_world = quat_to_mat(q0_world / (torch.norm(q0_world) + 1e-12))
        v0_world = (samples[1]["gt_vel"][0].to(device)
                    if len(samples) > 1 else torch.zeros(3, device=device))
        v_i = R0_world.t() @ v0_world
        g_rel = R0_world.t() @ gravity

        R_i = pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))
        p_i = torch.zeros(3, device=device)
        dense = [p_i.detach().cpu().numpy()]

        with torch.no_grad():
            for s in range(1, len(samples)):
                acc  = samples[s]["imu_acc"].to(device)
                gyro = samples[s]["imu_gyro"].to(device)
                dts  = samples[s]["imu_dts"].to(device)
                out  = pre(acc, gyro, dts, ba, bg)

                # Per-IMU-sample dense path: at intra-segment time t,
                # p(t) = p_i + v_i*t + 0.5*g_rel*t^2 + R_i * pre_paths[k]
                t_acc = torch.tensor(0.0, device=device)
                for k, dp_local in enumerate(out["pre_paths"]):
                    t_acc = t_acc + dts[k]
                    p_t = p_i + v_i * t_acc + 0.5 * g_rel * (t_acc ** 2) + R_i * dp_local
                    dense.append(p_t.detach().cpu().numpy())

                # Update segment-end state for the next window.
                dt = dts.sum()
                p_i = p_i + v_i * dt + 0.5 * g_rel * dt * dt + R_i * out["delta_p"]
                v_i = v_i + g_rel * dt + R_i * out["delta_v"]
                R_i = R_i * out["delta_r"].squeeze(0)

        self._log_trajectory(label, dense, radii=0.005)

    # -----------------------------------------------------------------
    # Blueprint: define the default viewer layout in one place.
    # -----------------------------------------------------------------
    def setup_blueprint(self, n_cams, methods):
        """Send a blueprint that lays out the viewer sensibly.

        Layout (left to right):
          * Left: 3D scene with everything overlaid.
          * Middle: vertical stack of time-series views — one per scalar
            metric, methods overlaid as series.
          * Right: tabs, one per camera, each showing a grid of that
            camera's 2D images (depth / conf) for every method.

        n_cams:           number of context frames (so we know how many tabs).
        methods:          list of method labels actually logged (e.g.
                           ["baseline", "cfa_gyro"]).
        """
        rr = self._rr
        import rerun.blueprint as rrb

        # --- Left: 3D scene ---
        scene = rrb.Spatial3DView(name="3D scene", origin="/scene")

        # --- Middle: scalar plots, one per metric ---
        # Per-frame error plots (per-frame, methods overlaid).
        frame_keys = ["p_total", "q_total", "depth_absrel", "scale",
                       "p_x", "p_y", "p_z", "q_roll", "q_pitch", "q_yaw"]
        frame_views = [
            rrb.TimeSeriesView(
                name=f"frame_err/{k}",
                origin=f"/scalars/frame_err/{k}",
                time_ranges=rrb.VisibleTimeRange(
                    "frame",
                    start=rrb.TimeRangeBoundary.infinite(),
                    end=rrb.TimeRangeBoundary.infinite(),
                ),
            ) for k in frame_keys
        ]

        scalars = rrb.Grid(name="Per-frame error", *frame_views)

        # --- Right: one tab per camera, each a grid of the camera's images ---
        cam_tabs_children = []
        for i in range(n_cams):
            cam_origin = f"/images/cam_{i:03d}"
            panes = []
            for m in methods:
                panes.append(rrb.Spatial2DView(name=f"{m}/depth",
                                                origin=f"{cam_origin}/{m}/depth"))
                panes.append(rrb.Spatial2DView(name=f"{m}/conf",
                                                origin=f"{cam_origin}/{m}/depth_conf"))
            cam_tabs_children.append(rrb.Grid(name=f"cam_{i:03d}", *panes))
        cam_tabs = rrb.Tabs(*cam_tabs_children, name="Cameras")

        bp = rrb.Blueprint(
            rrb.Horizontal(scene, scalars, cam_tabs, column_shares=[2, 2, 2]),
            collapse_panels=True,
        )
        rr.send_blueprint(bp)
