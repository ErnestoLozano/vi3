"""EuRoC MAV dataset wrapper (ETH ASL).

Same sample-dict contract as `TartanAirRefinementDataset`, with these
EuRoC specifics:

  * `gt_depth` is populated per-frame from one of two physical sources,
    cached on disk under `mav0/depth_cache_<H>x<W>/<ts>.npy`:
      - **Leica MS50** pointcloud (Vicon Room, V1_*/V2_*): `pointcloud0/data.ply`
        projected through the cam's GT pose (mm accuracy where covered).
      - **Stereo SGBM** (Machine Hall, MH_*): cam0+cam1 stereo-rectified,
        matched with cv2.StereoSGBM, re-projected (holes in textureless
        regions).
    If neither source is present, `gt_depth` is omitted and the depth metric
    is NaN.
  * `gt_bias_acc` / `gt_bias_gyro` are per-IMU-sample `[M, 3]` tensors sliced
    from the state-GT CSV's `b_a_*` / `b_w_*` columns (time-varying true biases).

Poses are normalized so the **camera axes** match TartanAir's convention
(x-forward, y-right, z-down); the **world frame** is left in EuRoC's native
ENU. We compute `T_W_cam0 = T_W_B @ T_BS_cam0` (extrinsics from sensor.yaml)
and apply a constant change-of-basis on the camera side only:
    R_W_cam_tartan = R_W_cam_euroc @ M_cam.T   (M_cam = vi3.transforms.M_change_of_basis)
Velocities are world-frame ENU quantities, returned untouched.
"""

import os

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from vi3.data._euroc_extract import ensure_extracted
from vi3.data.base import (RefinementDataset, compute_K_preprocessed,
                             predict_post_hw)
from vi3.models import build_model_spec


# ---------------------------------------------------------------------------
# Sequence catalog — short label -> (category_dir, group)
# ---------------------------------------------------------------------------

EUROC_SEQUENCES = {
    "MH_01_easy":      ("machine_hall", "MH"),
    "MH_02_easy":      ("machine_hall", "MH"),
    "MH_03_medium":    ("machine_hall", "MH"),
    "MH_04_difficult": ("machine_hall", "MH"),
    "MH_05_difficult": ("machine_hall", "MH"),
    "V1_01_easy":      ("vicon_room1",  "V1"),
    "V1_02_medium":    ("vicon_room1",  "V1"),
    "V1_03_difficult": ("vicon_room1",  "V1"),
    "V2_01_easy":      ("vicon_room2",  "V2"),
    "V2_02_medium":    ("vicon_room2",  "V2"),
    "V2_03_difficult": ("vicon_room2",  "V2"),
}


# Change-of-basis for the **camera axes only**:
#   EuRoC cam (x-right, y-down, z-forward) -> TartanAir cam (x-forward, y-right, z-down)
# Same matrix as `vi3.transforms.M_change_of_basis`. Applied on the camera
# side of the SE(3) only; world side stays in EuRoC's native ENU.
_M_CAM = np.array([[0., 0., 1.],
                   [1., 0., 0.],
                   [0., 1., 0.]], dtype=np.float64)


class EuRoCRefinementDataset(RefinementDataset):
    """One EuRoC sequence, in TartanAir's sample-dict shape (sans gt_depth).

    Sample keys:
        image, K, gt_pose, imu_acc, imu_gyro, imu_dts,
        gt_acc_nograv, gt_vel, gt_pos, gt_ori,
        gt_bias_acc [M, 3], gt_bias_gyro [M, 3]   (EuRoC: time-varying)
    """

    NATIVE_W = 752    # cam0 native (VI-Sensor MT9M034)
    NATIVE_H = 480

    # World-frame convention. ENU: x-east, y-north, z-up. Gravity points
    # along world -z (i.e. down = -z). Read by `Grounder` to orient gravity
    # relative to the camera at frame 0.
    WORLD_FRAME   = "enu"
    GRAVITY_WORLD = (0.0, 0.0, -9.81)

    # Real MEMS IMU (ADIS16448) with time-varying bg/ba drift and noise.
    # EuRoC ships per-sample GT biases from the dataset's state estimate
    # (Vicon/Leica-anchored), loaded directly from the GT CSV (cols 11:17).
    # These are true GT, not derived by us.
    BIAS_IS_GT = True

    # Continuous-time IMU noise densities. EuRoC's VI-Sensor uses an
    # ADIS16448 IMU; the dataset paper (Burri et al. 2016) publishes
    # σ_a = 2.0e-3 m/s²/√Hz, σ_g = 1.7e-4 rad/s/√Hz. Read by `IMUInit`
    # for the preintegrator's covariance-propagation Q matrix. Units:
    # m/s²/√Hz, rad/s/√Hz.
    IMU_ACC_NOISE_DENSITY  = 2.0e-3
    IMU_GYRO_NOISE_DENSITY = 1.7e-4

    # Stationary-prefix skip: drop initial frames whose body position is
    # still within `DEFAULT_MOTION_THRESH_M` of the first GT-window frame.
    # EuRoC sequences open with several seconds of stationary MAV on the
    # floor; standard VIO benchmarks drop that prefix.
    DEFAULT_MOTION_THRESH_M = 0.10

    def __init__(self, base_path, sequence="MH_01_easy",
                 mode="pad", cam="cam0",
                 model_spec=None, frame_stride=1):
        """
        base_path:              EUROC_ROOT (contains machine_hall/, vicon_room1/, vicon_room2/)
        sequence:               one of `EUROC_SEQUENCES.keys()`
        cam:                    "cam0" only (stereo deferred)
        model_spec:             vi3.models.ModelSpec controlling preprocessing
                                + K math. Defaults to the VGGT spec.
        """
        super().__init__(mode=mode)
        if model_spec is None:
            model_spec = build_model_spec("vggt")
        self.model_spec = model_spec
        self._loader_mode = model_spec.preproc_mode
        if sequence not in EUROC_SEQUENCES:
            raise ValueError(
                f"Unknown EuRoC sequence {sequence!r}. "
                f"Known: {sorted(EUROC_SEQUENCES.keys())}"
            )
        if cam != "cam0":
            raise NotImplementedError("Only cam0 is supported in v1.")

        self.sequence = sequence
        category, group = EUROC_SEQUENCES[sequence]
        self.group = group

        seq_dir = os.path.join(base_path, category, sequence)
        mav0    = ensure_extracted(seq_dir)
        self.mav0_dir = mav0
        self.cam_dir  = os.path.join(mav0, cam, "data")
        if not os.path.isdir(self.cam_dir):
            raise FileNotFoundError(f"Camera data dir not found: {self.cam_dir}")

        # --- Camera intrinsics, distortion, body→cam extrinsic ---
        cam_yaml = _load_yaml(os.path.join(mav0, cam, "sensor.yaml"))
        fu, fv, cu, cv = cam_yaml["intrinsics"]
        self.NATIVE_FX, self.NATIVE_FY = float(fu), float(fv)
        self.NATIVE_CX, self.NATIVE_CY = float(cu), float(cv)
        self._dist_coeffs = np.array(cam_yaml["distortion_coefficients"],
                                     dtype=np.float64)
        self._native_K = np.array([
            [self.NATIVE_FX, 0,              self.NATIVE_CX],
            [0,              self.NATIVE_FY, self.NATIVE_CY],
            [0,              0,              1],
        ], dtype=np.float64)
        # Precompute undistortion maps — image size is fixed for the sequence.
        self._undist_map1, self._undist_map2 = cv2.initUndistortRectifyMap(
            self._native_K, self._dist_coeffs, None, self._native_K,
            (self.NATIVE_W, self.NATIVE_H), cv2.CV_32FC1,
        )

        T_BS_cam0 = _parse_T(cam_yaml["T_BS"])
        # ASL convention: `T_BS` is the sensor-to-body extrinsic (p_body =
        # T_BS @ p_sensor). So body→sensor (cam) rotation is its transpose.
        R_body_to_cam0_euroc = T_BS_cam0[:3, :3].T

        # Composition that rotates an IMU body-frame vector all the way into
        # the TartanAir cam frame our `gt_pose` lives in:
        #   v_body_FLU -> v_cam_RDF (EuRoC) -> v_cam_tartanair
        R_body_to_cam_tartan = _M_CAM @ R_body_to_cam0_euroc

        # --- CSV streams (timestamps in nanoseconds) ---
        cam_ts_ns = np.loadtxt(os.path.join(mav0, cam, "data.csv"),
                                delimiter=",", usecols=(0,), dtype=np.int64)
        imu_csv = np.loadtxt(os.path.join(mav0, "imu0", "data.csv"),
                              delimiter=",", skiprows=1)
        imu_ts_ns = imu_csv[:, 0].astype(np.int64)
        gyro_body = imu_csv[:, 1:4].astype(np.float64)        # rad/s, body FLU
        acc_body  = imu_csv[:, 4:7].astype(np.float64)        # m/s², body FLU (gravity in)

        # Rotate IMU samples into the cam (TartanAir-axes) frame so they
        # share the frame `pose_delta_loss` and `R0_world.T @ gravity`
        # operate in. This matches TartanAir's implicit body=cam assumption.
        acc  = acc_body  @ R_body_to_cam_tartan.T   # (N, 3) @ (3, 3) ≡ R @ v per row
        gyro = gyro_body @ R_body_to_cam_tartan.T

        gt_csv = np.loadtxt(
            os.path.join(mav0, "state_groundtruth_estimate0", "data.csv"),
            delimiter=",", skiprows=1,
        )
        gt_ts_ns     = gt_csv[:, 0].astype(np.int64)
        gt_pos_W     = gt_csv[:, 1:4].astype(np.float64)
        gt_quat_wxyz = gt_csv[:, 4:8].astype(np.float64)
        gt_vel_W     = gt_csv[:, 8:11].astype(np.float64)
        gt_bias_gyro_body = gt_csv[:, 11:14].astype(np.float64)    # rad/s, body FLU
        gt_bias_acc_body  = gt_csv[:, 14:17].astype(np.float64)    # m/s², body FLU
        # Same rotation as the raw IMU samples: bias is in the same frame as acc/gyro.
        gt_bias_gyro = gt_bias_gyro_body @ R_body_to_cam_tartan.T
        gt_bias_acc  = gt_bias_acc_body  @ R_body_to_cam_tartan.T

        # Float seconds relative to the earliest stamp across all streams.
        t0_ns = min(int(cam_ts_ns.min()), int(imu_ts_ns.min()), int(gt_ts_ns.min()))
        cam_times = ((cam_ts_ns - t0_ns) * 1e-9).astype(np.float64)
        imu_times = ((imu_ts_ns - t0_ns) * 1e-9).astype(np.float64)
        gt_times  = ((gt_ts_ns  - t0_ns) * 1e-9).astype(np.float64)

        # --- Clip camera frames to the GT-covered window ---
        # State-GT typically starts after cam0/imu0. We keep only frames
        # inside [gt_times[0], gt_times[-1]] so every frame has GT pose
        # + GT velocity + GT bias available without extrapolation.
        keep = (cam_times >= gt_times[0]) & (cam_times <= gt_times[-1])
        cam_times = cam_times[keep]
        cam_ts_ns = cam_ts_ns[keep]
        if len(cam_times) == 0:
            raise RuntimeError(f"No camera frames inside GT window for {sequence!r}")

        # --- Skip the stationary calibration prefix ---
        # Drop frames before the body has moved DEFAULT_MOTION_THRESH_M from the
        # first GT-window frame. EuRoC sequences open with several seconds of
        # sub-mm motion; the frozen foundation model produces degenerate poses at zero
        # parallax, so evaluating those frames would just measure "predict zero
        # motion".
        ref_pos = gt_pos_W[0]
        cam_gt_idx = np.clip(
            np.searchsorted(gt_times, cam_times), 0, len(gt_times) - 1,
        )
        cam_body_pos = gt_pos_W[cam_gt_idx]
        dist = np.linalg.norm(cam_body_pos - ref_pos, axis=1)
        moving = dist > self.DEFAULT_MOTION_THRESH_M
        if not moving.any():
            raise RuntimeError(
                f"No camera frame in {sequence!r} reached the "
                f"{self.DEFAULT_MOTION_THRESH_M:.3f} m motion threshold; "
                f"sequence appears entirely stationary."
            )
        first_moving = int(np.argmax(moving))
        self.warmup_skip_frames = first_moving
        cam_times = cam_times[first_moving:]
        cam_ts_ns = cam_ts_ns[first_moving:]

        # --- Convert all GT poses: cam axes only, world stays ENU ---
        # T_W_B (body in world, EuRoC frames) -> T_W_cam0_euroc -> swap cam axes.
        # We do *not* touch the world side — it remains in EuRoC's native ENU
        # convention, matched by `GRAVITY_WORLD = (0, 0, -9.81)`.
        N_gt = len(gt_times)
        T_W_cam = np.empty((N_gt, 4, 4), dtype=np.float64)
        # Also keep the un-axis-swapped (OpenCV-cam) pose for projecting
        # the Leica scan into image coordinates downstream — projection
        # math uses K (OpenCV convention) so the cam side must be OpenCV.
        T_W_cam_opencv = np.empty((N_gt, 4, 4), dtype=np.float64)
        for i in range(N_gt):
            q_xyzw = np.array([
                gt_quat_wxyz[i, 1], gt_quat_wxyz[i, 2],
                gt_quat_wxyz[i, 3], gt_quat_wxyz[i, 0],
            ])
            R_W_B = R.from_quat(q_xyzw).as_matrix()
            T_W_B = np.eye(4)
            T_W_B[:3, :3] = R_W_B
            T_W_B[:3, 3]  = gt_pos_W[i]
            T_W_cam0_euroc = T_W_B @ T_BS_cam0
            T_W_cam_opencv[i] = T_W_cam0_euroc
            T_W_cam[i] = _swap_cam_axes_se3(T_W_cam0_euroc, _M_CAM)

        # Velocities and positions are world-frame (ENU) — keep them as is.
        # Biases live in the body / sensor frame; they're already in EuRoC's
        # IMU body frame and are returned unchanged.

        self._gt_times      = gt_times
        self._gt_T_W_cam    = T_W_cam
        self._gt_T_W_cam_opencv = T_W_cam_opencv
        self._gt_vel_W      = gt_vel_W           # ENU world-frame velocity
        self._gt_bias_acc   = gt_bias_acc        # [N_gt, 3]
        self._gt_bias_gyro  = gt_bias_gyro       # [N_gt, 3]
        self.cam_times      = cam_times
        self.cam_ts_ns      = cam_ts_ns
        self.imu_times      = imu_times
        self._imu_acc       = acc
        self._imu_gyro      = gyro

        # frame_stride: keep every k-th CAM frame so consecutive dataset indices
        # are k original frames apart (more motion per window). Only the per-frame
        # cam arrays are subsampled; GT poses are looked up by TIMESTAMP
        # (`searchsorted(_gt_times, cam_times[idx])` in __getitem__), and IMU is
        # masked by timestamp between kept frames' cam_times in get_imu_segment,
        # so both correctly span the skipped frames. `img_names` is rebuilt below
        # from the (already-subsampled) cam_ts_ns. Mirrors tartanair.py.
        self.frame_stride = max(1, int(frame_stride))
        if self.frame_stride > 1:
            k = self.frame_stride
            self.cam_times = self.cam_times[::k]
            self.cam_ts_ns = self.cam_ts_ns[::k]

        # Intrinsics for the preprocessed image, computed eagerly from
        # the predicted post-preprocess shape.
        self._post_hw = predict_post_hw(
            native_h=self.NATIVE_H, native_w=self.NATIVE_W,
            target_size=self.model_spec.target_size,
            patch_size=self.model_spec.patch_size,
            mode=self.model_spec.preproc_mode,
        )
        self.K = self._compute_K_preprocessed(post_hw=self._post_hw)

        # Filenames in cam0/data/ are <ts_ns>.png. Build from the (already
        # stride-subsampled) self.cam_ts_ns so img_names[idx] matches the frame
        # __getitem__ actually loads.
        self.img_names = [f"{ts}.png" for ts in self.cam_ts_ns]

        # ---- gt_depth source detection (Leica > stereo > none) -------
        # Stored as paths / metadata; the actual depth maps are produced
        # lazily per-frame and cached on disk.
        self._T_BS_cam0       = T_BS_cam0     # needed for stereo extrinsic
        self._depth_source    = None          # "leica" | "stereo" | None
        self._leica_pcd       = None          # lazy: [N, 3] numpy
        self._leica_pcd_path  = None
        self._stereo_state    = None          # lazy: dict of rect maps + Q
        ply_path = os.path.join(mav0, "pointcloud0", "data.ply")
        cam1_dir = os.path.join(mav0, "cam1", "data")
        if os.path.exists(ply_path):
            self._depth_source   = "leica"
            self._leica_pcd_path = ply_path
        elif os.path.isdir(cam1_dir):
            self._depth_source   = "stereo"
            self._stereo_cam1_yaml_path = os.path.join(mav0, "cam1", "sensor.yaml")
            self._stereo_cam1_dir = cam1_dir
        # else: no depth source — gt_depth will be omitted from samples.

        # Per-sequence depth cache dir (keyed by post-preprocess resolution
        # so different model_specs don't clobber each other).
        self._depth_cache_dir = os.path.join(
            mav0, f"depth_cache_{self._post_hw[0]}x{self._post_hw[1]}",
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.cam_times)

    # ------------------------------------------------------------------
    # Intrinsics derivation — delegates to the free helper for consistency
    # with TartanAir, and so per-model variants (VGGT/Omega) are handled
    # in one place.
    # ------------------------------------------------------------------
    def _compute_K_preprocessed(self, post_hw=None):
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
    # GT depth — Leica (Vicon Room) or stereo (Machine Hall).
    # ------------------------------------------------------------------
    def _load_leica_pcd(self):
        """Lazy-load the Leica MS50 ASCII PLY → [N, 3] float32 world points."""
        if self._leica_pcd is not None:
            return self._leica_pcd
        from plyfile import PlyData
        pcd = PlyData.read(self._leica_pcd_path)["vertex"]
        self._leica_pcd = np.stack(
            [pcd["x"], pcd["y"], pcd["z"]], axis=-1
        ).astype(np.float32)
        return self._leica_pcd

    def _project_leica_to_depth(self, T_W_cam_opencv):
        """Project the Leica scan into a [post_h, post_w] depth map at the
        preprocessed-image resolution. Uses self.K (preprocessed OpenCV
        intrinsics) and inverts T_W_cam to bring world points into cam."""
        H, W = self._post_hw
        K = self.K.numpy().astype(np.float64)
        pcd = self._load_leica_pcd().astype(np.float64)               # [N, 3]
        T_cam_W = np.linalg.inv(T_W_cam_opencv)
        P_cam = pcd @ T_cam_W[:3, :3].T + T_cam_W[:3, 3]              # [N, 3]
        front = P_cam[:, 2] > 0.05
        P_cam = P_cam[front]
        proj_uv = (K[:3, :3] @ P_cam.T).T                              # [N', 3]
        u = proj_uv[:, 0] / proj_uv[:, 2]
        v = proj_uv[:, 1] / proj_uv[:, 2]
        z = P_cam[:, 2]
        in_b = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_i = u[in_b].astype(np.int32)
        v_i = v[in_b].astype(np.int32)
        z_i = z[in_b].astype(np.float32)
        # Z-buffer reduction (multiple points per pixel → keep closest).
        depth = np.full((H, W), np.inf, dtype=np.float32)
        np.minimum.at(depth, (v_i, u_i), z_i)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def _setup_stereo(self):
        """One-shot rectify setup for cam0+cam1 (used by Machine Hall)."""
        if self._stereo_state is not None:
            return self._stereo_state
        cam1_yaml = _load_yaml(self._stereo_cam1_yaml_path)
        c1_fu, c1_fv, c1_cu, c1_cv = cam1_yaml["intrinsics"]
        K1 = np.array([[c1_fu, 0, c1_cu], [0, c1_fv, c1_cv], [0, 0, 1]],
                      dtype=np.float64)
        D1 = np.array(cam1_yaml["distortion_coefficients"], dtype=np.float64)
        T_BS_cam1 = _parse_T(cam1_yaml["T_BS"])
        # T_cam0_cam1 = inv(T_BS_cam0) @ T_BS_cam1: cam1 in cam0's frame.
        T_c0_c1 = np.linalg.inv(self._T_BS_cam0) @ T_BS_cam1
        R_c0_c1, t_c0_c1 = T_c0_c1[:3, :3], T_c0_c1[:3, 3]
        size = (self.NATIVE_W, self.NATIVE_H)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self._native_K, self._dist_coeffs, K1, D1, size,
            R_c0_c1, t_c0_c1, alpha=0,
        )
        rmap0 = cv2.initUndistortRectifyMap(
            self._native_K, self._dist_coeffs, R1, P1, size, cv2.CV_32FC1,
        )
        rmap1 = cv2.initUndistortRectifyMap(
            K1, D1, R2, P2, size, cv2.CV_32FC1,
        )
        # StereoSGBM — 128 disparities, block 5; works well on 752×480 VGA.
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=5,
            P1=8 * 3 * 5 * 5, P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        # Disparity → depth via the direct formula depth = fx_rect * baseline / disp.
        fx_rect  = float(P1[0, 0])
        baseline = float(abs(t_c0_c1[0]))   # ≈ 0.11 m on the VI-Sensor
        self._stereo_state = dict(
            rmap0=rmap0, rmap1=rmap1, sgbm=sgbm,
            fx_rect=fx_rect, baseline=baseline,
        )
        return self._stereo_state

    def _project_stereo_to_depth(self, idx):
        """Per-frame SGBM stereo → depth in the rectified-cam0 frame, then
        resampled to `self._post_hw` to match the preprocessed image grid.
        Approximation: stays in rectified frame (small rotation R1 vs the
        un-rectified cam0 K). Sufficient for aggregate depth metrics."""
        ts_ns = int(self.cam_ts_ns[idx])
        img1_path = os.path.join(self._stereo_cam1_dir, f"{ts_ns}.png")
        if not os.path.exists(img1_path):
            return None
        img0 = cv2.imread(os.path.join(self.cam_dir, f"{ts_ns}.png"),
                            cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            return None
        st = self._setup_stereo()
        r0 = cv2.remap(img0, st["rmap0"][0], st["rmap0"][1],
                        interpolation=cv2.INTER_LINEAR)
        r1 = cv2.remap(img1, st["rmap1"][0], st["rmap1"][1],
                        interpolation=cv2.INTER_LINEAR)
        disp = st["sgbm"].compute(r0, r1).astype(np.float32) / 16.0
        # Direct formula: depth = fx_rect * baseline / disparity.
        # SGBM marks invalid pixels with disp = -1 (after /16); zero-out
        # the result there. Also clip absurd values from textureless
        # regions (SGBM produces matches with disp ≈ minDisparity → huge depth).
        depth_rect = np.zeros_like(disp)
        valid = disp > 0
        depth_rect[valid] = (st["fx_rect"] * st["baseline"]) / disp[valid]
        depth_rect[~np.isfinite(depth_rect)] = 0.0
        depth_rect[depth_rect > 50.0] = 0.0
        # Resize to preprocessed resolution. For fixed-output models (VGGT/Pi3)
        # `preprocess_depth` does the resize + pad. For variable-output
        # models (Omega) we resize-only via nearest interpolation to the
        # predicted post-preprocess shape (no padding). Same branch as
        # TartanAirRefinementDataset uses for its raw depth maps.
        if self.model_spec.fixed_output_hw:
            return self.preprocess_depth(depth_rect).numpy()
        import torch.nn.functional as _F
        d = torch.from_numpy(depth_rect).unsqueeze(0).unsqueeze(0)
        d = _F.interpolate(d, size=self._post_hw, mode="nearest")
        return d.squeeze(0).squeeze(0).numpy()

    def _get_gt_depth(self, idx):
        """Return a [post_h, post_w] float32 depth tensor, or None.

        Lazy + on-disk caching: per-frame .npy under `_depth_cache_dir`
        keyed by camera timestamp. First miss triggers projection;
        subsequent calls (and other dataloader workers) just read.
        """
        if self._depth_source is None:
            return None
        ts_ns = int(self.cam_ts_ns[idx])
        cache_path = os.path.join(self._depth_cache_dir, f"{ts_ns}.npy")
        if os.path.exists(cache_path):
            depth = np.load(cache_path)
        else:
            if self._depth_source == "leica":
                t = float(self.cam_times[idx])
                gi = int(np.clip(np.searchsorted(self._gt_times, t), 0,
                                  len(self._gt_times) - 1))
                depth = self._project_leica_to_depth(
                    self._gt_T_W_cam_opencv[gi]
                )
            elif self._depth_source == "stereo":
                depth = self._project_stereo_to_depth(idx)
            else:
                return None
            if depth is None:
                return None
            os.makedirs(self._depth_cache_dir, exist_ok=True)
            np.save(cache_path, depth.astype(np.float32))
        return torch.from_numpy(depth).float()

    # ------------------------------------------------------------------
    # IMU + GT segmentation
    # ------------------------------------------------------------------
    def get_imu_segment(self, idx):
        """Return the per-frame IMU + state-GT window. IMU samples are
        unmodified (real sensor noise, no synthetic corruption).
        """
        t_end = float(self.cam_times[idx])
        if idx == 0:
            t_start = min(float(self.imu_times[0]), t_end)
            mask = self.imu_times <= t_end
        else:
            t_start = float(self.cam_times[idx - 1])
            mask = (self.imu_times > t_start) & (self.imu_times <= t_end)

        actual_t = self.imu_times[mask]
        if len(actual_t) == 0:
            nearest = int(np.clip(
                np.searchsorted(self.imu_times, (t_start + t_end) * 0.5),
                0, len(self.imu_times) - 1,
            ))
            acc  = torch.from_numpy(self._imu_acc[[nearest]]).float()
            gyro = torch.from_numpy(self._imu_gyro[[nearest]]).float()
            dts  = torch.tensor([t_end - t_start], dtype=torch.float32)
            imu_idxs = np.array([nearest])
        else:
            dts_np = np.diff(actual_t, prepend=t_start).copy()
            dts_np[-1] += t_end - actual_t[-1]
            acc  = torch.from_numpy(self._imu_acc[mask]).float()
            gyro = torch.from_numpy(self._imu_gyro[mask]).float()
            dts  = torch.from_numpy(dts_np).float()
            imu_idxs = np.flatnonzero(mask)

        # Per-IMU-sample GT lookup: nearest GT row by timestamp (200 Hz / 200 Hz).
        imu_t_window = self.imu_times[imu_idxs]
        gt_idxs = np.clip(
            np.searchsorted(self._gt_times, imu_t_window),
            0, len(self._gt_times) - 1,
        )
        gt_vel = self._gt_vel_W[gt_idxs]                            # [M, 3] ENU
        gt_pos = self._gt_T_W_cam[gt_idxs, :3, 3]                  # [M, 3]
        gt_ori = np.empty((len(gt_idxs), 4), dtype=np.float32)
        for k, gi in enumerate(gt_idxs):
            gt_ori[k] = R.from_matrix(self._gt_T_W_cam[gi, :3, :3]) \
                          .as_quat().astype(np.float32)
        # Time-varying real biases from the GT estimator.
        gt_b_a = self._gt_bias_acc[gt_idxs]                         # [M, 3]
        gt_b_g = self._gt_bias_gyro[gt_idxs]                        # [M, 3]

        return {
            "imu_acc":       acc,
            "imu_gyro":      gyro,
            "imu_dts":       dts,
            # gravity-subtracted world-frame accel: not computed from EuRoC
            # (TartanAir provides this as a debug aux; visualization only).
            "gt_acc_nograv": torch.zeros_like(acc),
            "gt_vel":        torch.from_numpy(gt_vel.astype(np.float32)),
            "gt_pos":        torch.from_numpy(gt_pos.astype(np.float32)),
            "gt_ori":        torch.from_numpy(gt_ori),
            "gt_bias_acc":   torch.from_numpy(gt_b_a.astype(np.float32)),   # [M, 3]
            "gt_bias_gyro":  torch.from_numpy(gt_b_g.astype(np.float32)),   # [M, 3]
        }

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        ts_ns = int(self.cam_ts_ns[idx])
        img_path = os.path.join(self.cam_dir, f"{ts_ns}.png")

        # Undistort + grayscale->RGB, then write to a per-(sequence, frame)
        # /tmp cache and call the existing preprocessing helper on the
        # cached path. Cheaper than re-porting resize+pad math here.
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_und = cv2.remap(img_gray, self._undist_map1, self._undist_map2,
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_und, cv2.COLOR_GRAY2RGB)

        tmp_path = _tmp_image_path(self.sequence, ts_ns)
        if not os.path.exists(tmp_path):
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            cv2.imwrite(tmp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        image = self.model_spec.load_fn(
            [tmp_path], mode=self._loader_mode,
        ).squeeze(0)

        # GT pose at this camera frame: nearest GT row by timestamp.
        t = float(self.cam_times[idx])
        gi = int(np.clip(np.searchsorted(self._gt_times, t), 0,
                          len(self._gt_times) - 1))
        T = self._gt_T_W_cam[gi]
        pose_pos  = T[:3, 3].astype(np.float32)
        pose_quat = R.from_matrix(T[:3, :3]).as_quat().astype(np.float32)
        gt_pose = torch.from_numpy(np.concatenate([pose_pos, pose_quat]))

        imu = self.get_imu_segment(idx)

        out = {
            "image":         image,
            "K":             self.K,
            "gt_pose":       gt_pose,
            "imu_acc":       imu["imu_acc"],
            "imu_gyro":      imu["imu_gyro"],
            "imu_dts":       imu["imu_dts"],
            "gt_acc_nograv": imu["gt_acc_nograv"],
            "gt_vel":        imu["gt_vel"],
            "gt_pos":        imu["gt_pos"],
            "gt_ori":        imu["gt_ori"],
            "gt_bias_acc":   imu["gt_bias_acc"],
            "gt_bias_gyro":  imu["gt_bias_gyro"],
        }
        # Lazy + disk-cached GT depth from Leica scan (Vicon Room) or
        # stereo SGBM (Machine Hall). Returns None when no source exists.
        gt_depth = self._get_gt_depth(idx)
        if gt_depth is not None:
            out["gt_depth"] = gt_depth
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_T(t_bs_dict):
    """Parse the EuRoC `T_BS` (cols, rows, data) dict into a 4x4 numpy array."""
    rows, cols = int(t_bs_dict["rows"]), int(t_bs_dict["cols"])
    data = np.asarray(t_bs_dict["data"], dtype=np.float64).reshape(rows, cols)
    if data.shape != (4, 4):
        raise ValueError(f"Expected 4x4 T_BS, got {data.shape}")
    return data


def _swap_cam_axes_se3(T, M_cam):
    """Apply a camera-axes change-of-basis to an SE(3) `T_W_cam` (cam→world).

    Only the cam side of the SE(3) is touched — the world side stays put.
    For a cam→world matrix `T` with rotation `R_W_cam` and translation
    `t_W_cam_in_W`, swapping cam axes via `R_cam_new = R_cam_old @ M_cam.T`
    gives the new SE(3) operating on cam-new vectors and still producing
    world-frame outputs in the original (un-touched) world basis:

        R_W_camnew = R_W_camold @ M_cam.T
        t_W_camnew = t_W_camold                (translation in world; no swap)
    """
    T_new = np.eye(4)
    T_new[:3, :3] = T[:3, :3] @ M_cam.T
    T_new[:3, 3]  = T[:3, 3]
    return T_new


def _tmp_image_path(sequence, ts_ns):
    """Per-(sequence, frame) cache path for the undistorted RGB-3 image.

    Lives under /tmp so it's wiped on reboot; deterministic so repeated
    reads of the same frame skip `cv2.remap`.
    """
    return os.path.join("/tmp/euroc_undist_cache", sequence, f"{ts_ns}.png")
