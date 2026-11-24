"""TartanAir dataset wrapper.

Pose convention: camera->world, where the camera frame is (x-forward, y-right,
z-down) and the world frame is NED. Returned sample dicts follow the contract
documented in `vi3.data.base.RefinementDataset`.
"""

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from vi3.data.base import (RefinementDataset, compute_K_preprocessed,
                             predict_post_hw)
from vi3.models import build_model_spec


class TartanAirRefinementDataset(RefinementDataset):
    """TartanAir trajectory with synchronized IMU + GT depth. The IMU is
    synthetic: noise- and bias-free by construction."""

    # Continuous-time noise densities (Allan-variance form) of a commercial MEMS
    # unit, for the preintegrator's covariance (density² · dt per segment).
    # Units: m/s²/√Hz, rad/s/√Hz.
    IMU_ACC_NOISE_DENSITY  = 0.04     # VRW (acc random walk density)
    IMU_GYRO_NOISE_DENSITY = 0.0035   # ARW (gyro random walk density)

    # Native TartanAir camera (lcam_front) — used to derive the post-
    # preprocessing intrinsics matrix `self.K`. Images on disk are 640×640
    # square with a 90° horizontal FOV and a centered principal point.
    NATIVE_W   = 640
    NATIVE_H   = 640
    NATIVE_FX  = 320.0   # 90° horizontal FOV: fx = W/2 / tan(45°) = W/2
    NATIVE_FY  = 320.0   # square pixels
    NATIVE_CX  = 320.0
    NATIVE_CY  = 320.0

    # World-frame convention. NED: x-north, y-east, z-down. Gravity points
    # along world +z (i.e. down). Read by `Grounder` to orient gravity
    # relative to the camera at frame 0.
    WORLD_FRAME   = "ned"
    GRAVITY_WORLD = (0.0, 0.0, 9.81)

    # Bias is not measured GT (it is zero by construction), so hold it at 0:
    # BIAS_IS_GT = False makes the grounder keep bg = ba = 0, the true value.
    BIAS_IS_GT = False

    def __init__(self, base_path, env="House", difficulty="Data_hard", traj="P000",
                 cam="lcam_front", mode="pad", model_spec=None, frame_stride=1):
        """
        model_spec: vi3.models.ModelSpec instance controlling preprocessing
            (loader fn, target size, patch size). Defaults to the VGGT spec.
        frame_stride: keep every k-th RGB frame so consecutive dataset indices
            are k frames apart in time -> ~k× the per-window baseline (more
            parallax/motion per window). IMU preintegration is UNAFFECTED and
            correct: `get_imu_segment(idx)` masks IMU by TIMESTAMP between
            cam_times[idx-1] and cam_times[idx] (the previous *kept* frame), so
            it accumulates all inertial samples across the skipped frames (as in
            the EuRoC and UZH-FPV loaders).
        """
        super().__init__(mode=mode)
        if model_spec is None:
            model_spec = build_model_spec("vggt")
        self.model_spec = model_spec
        # Image preprocessing follows the model spec (e.g. VGGT-Ω's "max_size").
        self._loader_mode = model_spec.preproc_mode

        self.frame_stride = max(1, int(frame_stride))
        self.difficulty = "Data_easy" if difficulty.lower() == "easy" else "Data_hard"
        self.env, self.traj = env, traj

        # Directories
        self.img_dir   = os.path.join(base_path, env, self.difficulty, traj, f"image_{cam}")
        self.depth_dir = os.path.join(base_path, env, self.difficulty, traj, f"depth_{cam}")
        self.imu_dir   = os.path.join(base_path, env, self.difficulty, traj, "imu")
        for d in (self.img_dir, self.depth_dir, self.imu_dir):
            if not os.path.exists(d):
                raise FileNotFoundError(f"Directory not found: {d}")

        # Poses
        self.pose_file = os.path.join(base_path, env, self.difficulty, traj, f"pose_{cam}.txt")
        self.poses = np.loadtxt(self.pose_file)

        # Images & depth
        self.img_names   = sorted(f for f in os.listdir(self.img_dir)   if f.endswith(".png"))
        self.depth_names = sorted(f for f in os.listdir(self.depth_dir) if f.endswith(".png"))

        # IMU
        self.acc              = np.load(os.path.join(self.imu_dir, "acc.npy"))
        self.gyro             = np.load(os.path.join(self.imu_dir, "gyro.npy"))
        self.imu_times        = np.load(os.path.join(self.imu_dir, "imu_time.npy"))
        self.cam_times        = np.load(os.path.join(self.imu_dir, "cam_time.npy"))
        self.acc_nograv_world = np.load(os.path.join(self.imu_dir, "acc_nograv.npy"))
        self.vel_world        = np.load(os.path.join(self.imu_dir, "vel_global.npy"))
        self.pos_world        = np.load(os.path.join(self.imu_dir, "pos_global.npy"))
        # Stored ori is Euler XYZ; convert to xyzw quat once.
        self.ori_world = torch.from_numpy(
            R.from_euler("xyz", np.load(os.path.join(self.imu_dir, "ori_global.npy")),
                         degrees=False).as_quat()
        ).numpy()

        # frame_stride: subsample the PER-FRAME arrays together (img_names,
        # depth_names, poses, cam_times) so consecutive dataset indices are
        # `frame_stride` original frames apart. The IMU arrays (acc/gyro/
        # imu_times/vel/pos/ori) are NOT subsampled — `get_imu_segment` masks
        # them by timestamp between the kept frames' cam_times, so it correctly
        # accumulates IMU across the skipped frames.
        if self.frame_stride > 1:
            k = self.frame_stride
            self.img_names   = self.img_names[::k]
            self.depth_names = self.depth_names[::k]
            self.poses       = self.poses[::k]
            self.cam_times   = self.cam_times[::k]

        # Intrinsics K for the preprocessed image — eagerly computed by
        # predicting the model's output shape from native dims + spec.
        self._post_hw = predict_post_hw(
            native_h=self.NATIVE_H, native_w=self.NATIVE_W,
            target_size=self.model_spec.target_size,
            patch_size=self.model_spec.patch_size,
            mode=self.model_spec.preproc_mode,
        )
        self.K = self._compute_K_preprocessed(post_hw=self._post_hw)

    def __len__(self):
        return len(self.img_names)

    # ------------------------------------------------------------------
    # Intrinsics derivation
    # ------------------------------------------------------------------
    def _compute_K_preprocessed(self, post_hw=None):
        """Return the [3, 3] intrinsics matrix for the preprocessed image.

        For fixed-output models (VGGT) we know `post_hw = (target_size,
        target_size)` analytically. For variable-output models (Omega) the
        caller passes the actually-realized `post_hw` from the first
        sample. Result is a constant across samples (camera is fixed).
        """
        if post_hw is None:
            ts = self.model_spec.target_size
            post_hw = (ts, ts)
        post_h, post_w = post_hw
        return compute_K_preprocessed(
            native_fx=self.NATIVE_FX, native_fy=self.NATIVE_FY,
            native_cx=self.NATIVE_CX, native_cy=self.NATIVE_CY,
            native_w=self.NATIVE_W, native_h=self.NATIVE_H,
            post_h=post_h, post_w=post_w,
            patch_size=self.model_spec.patch_size,
            target_size=self.model_spec.target_size,
            mode=self.model_spec.preproc_mode,
        )

    # ------------------------------------------------------------------
    # IMU segmentation
    # ------------------------------------------------------------------
    def get_imu_segment(self, idx):
        """Return IMU data for the integration window preceding camera frame `idx`.

        For idx > 0 the window is (cam_times[idx-1], cam_times[idx]].
        For idx == 0 the window is [imu_times[0], cam_times[0]] (warmup).

        Guarantees sum(imu_dts) == cam_times[idx] - t_start exactly. dt is in
        a backward-hold convention: dts[i] = time the i-th sample is active.
        """
        t_end = float(self.cam_times[idx])

        if idx == 0:
            # All IMU data up to and including the first camera frame. Clamp so
            # t_start <= t_end even when the IMU stream starts after cam_times[0].
            t_start = min(float(self.imu_times[0]), t_end)
            mask = self.imu_times <= t_end
        else:
            t_start = float(self.cam_times[idx - 1])
            # Exclusive lower bound — boundary sample belongs to the prior segment.
            mask = (self.imu_times > t_start) & (self.imu_times <= t_end)

        actual_imu_times = self.imu_times[mask]

        if len(actual_imu_times) == 0:
            # No IMU samples in this window — use the nearest one so shapes stay
            # consistent. Rare; happens on synthetic data with degenerate timings.
            nearest = int(np.clip(
                np.searchsorted(self.imu_times, (t_start + t_end) * 0.5),
                0, len(self.imu_times) - 1,
            ))
            raw_acc  = torch.from_numpy(self.acc[[nearest]]).float()
            raw_gyro = torch.from_numpy(self.gyro[[nearest]]).float()
            dts      = torch.tensor([t_end - t_start], dtype=torch.float32)
            gt_mask  = np.zeros(len(self.imu_times), dtype=bool)
            gt_mask[nearest] = True
        else:
            # np.diff with prepend = [t0-t_start, t1-t0, ..., tN-1-tN-2]. Sum =
            # tN-1 - t_start. Extend dts[-1] by (t_end - actual_imu_times[-1])
            # so the segment exactly covers the camera interval.
            dts = np.diff(actual_imu_times, prepend=t_start).copy()
            dts[-1] += t_end - actual_imu_times[-1]

            raw_acc  = torch.from_numpy(self.acc[mask]).float()
            raw_gyro = torch.from_numpy(self.gyro[mask]).float()
            dts      = torch.from_numpy(dts).float()
            gt_mask  = mask

        acc, gyro = raw_acc, raw_gyro
        L = acc.shape[0]
        b_a = torch.zeros(L, 3)          # bias-free by construction
        b_g = torch.zeros(L, 3)

        return {
            "imu_acc":       acc,
            "imu_gyro":      gyro,
            "imu_dts":       dts,
            "gt_acc_nograv": torch.from_numpy(self.acc_nograv_world[gt_mask]).float(),
            "gt_vel":        torch.from_numpy(self.vel_world[gt_mask]).float(),
            "gt_pos":        torch.from_numpy(self.pos_world[gt_mask]).float(),
            "gt_ori":        torch.from_numpy(self.ori_world[gt_mask]).float(),
            "gt_bias_acc":   b_a,
            "gt_bias_gyro":  b_g,
        }

    def __getitem__(self, idx):
        img_path   = os.path.join(self.img_dir,   self.img_names[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_names[idx])

        # Model-aware image preprocessing. Squeeze drops the batch dim
        # added by the loader (it expects a list of paths).
        image = self.model_spec.load_fn(
            [img_path], mode=self._loader_mode,
        ).squeeze(0)

        # GT depth: resize to match the realized image shape. For fixed-
        # output models (VGGT) this delegates to the base class. For
        # variable-output models (Omega) we resize-only via nearest
        # interpolation to the predicted post-preprocess shape (no padding).
        gt_depth_raw = self.load_tartanair_depth(depth_path)
        if self.model_spec.fixed_output_hw:
            gt_depth = self.preprocess_depth(gt_depth_raw)
        else:
            import torch.nn.functional as _F
            d = torch.from_numpy(gt_depth_raw).unsqueeze(0).unsqueeze(0)
            d = _F.interpolate(d, size=self._post_hw, mode="nearest")
            gt_depth = d.squeeze(0).squeeze(0)
        imu_data = self.get_imu_segment(idx)

        return {
            "image":         image,
            "K":             self.K,
            "gt_pose":       torch.from_numpy(self.poses[idx]).float(),
            "gt_depth":      gt_depth,
            "imu_acc":       imu_data["imu_acc"],
            "imu_gyro":      imu_data["imu_gyro"],
            "imu_dts":       imu_data["imu_dts"],
            "gt_acc_nograv": imu_data["gt_acc_nograv"],
            "gt_vel":        imu_data["gt_vel"],
            "gt_pos":        imu_data["gt_pos"],
            "gt_ori":        imu_data["gt_ori"],
            "gt_bias_acc":   imu_data["gt_bias_acc"],
            "gt_bias_gyro":  imu_data["gt_bias_gyro"],
        }
