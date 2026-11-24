"""UZH-FPV (Snapdragon mono left) dataset wrapper (RPG/UZH).

A high-speed FPV-drone dataset that complements EuRoC: aggressive aerial
motion (forward + 45° tilt sequences, 5-10 m/s) under real IMU noise. We use
the Snapdragon mono-left camera (640×480 @ 30 Hz, fisheye/equidistant).

Sample contract is identical to `EuRoCRefinementDataset`:
    image, K, gt_pose, imu_acc, imu_gyro, imu_dts,
    gt_acc_nograv, gt_vel, gt_pos, gt_ori,
    gt_bias_acc [M, 3], gt_bias_gyro [M, 3],
    gt_depth [post_h, post_w]                  (Snapdragon stereo SGBM)

UZH-FPV specifics:

  * **Pose-only ground truth** — `groundtruth.txt` ships `(t, tx, ty, tz,
    qx, qy, qz, qw)`; no velocity, no biases. We derive them offline:
      - `gt_vel` via central differences on the dense GT position stream.
      - `gt_bias_*` via IMU − GT-pose-derivative (differentiate GT rotation
        for true ω, double-differentiate position for true accel, rotate to
        body and remove gravity, subtract from raw IMU, smooth ~1 s).
        Cached on disk.
  * **Per-family calibration** — one Kalibr calibration per family
    (indoor/outdoor × forward/45), under `<UZHFPV_ROOT>/calib/<family>/`:
    pinhole + `equidistant` distortion, with `T_cam_imu` / `T_cn_cnm1`.
  * **Snapdragon stereo SGBM depth** — cam0+cam1 rectified with
    `cv2.fisheye.*` (baseline ≈ 80 mm), matched with StereoSGBM.

World-frame convention: the published GT lives in a roughly z-up VIO-init
frame, so `GRAVITY_WORLD = (0, 0, -9.81)`, sanity-checked on load.

Race sequences (`race_1..3`) ship no `groundtruth.txt` and use a different
camera; they are out of scope.
"""

import os

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from vi3.data.base import (
    RefinementDataset, compute_K_preprocessed, predict_post_hw,
    _central_diff_position, _second_central_diff_position,
    _central_diff_rotation, _linear_interp_3d, _moving_average,
)
from vi3.models import build_model_spec


# ---------------------------------------------------------------------------
# Sequence catalog + per-family calibration mapping
# ---------------------------------------------------------------------------

# Short label -> (family_dir, group_tag).
# `family_dir` selects the calibration; `group_tag` is for suite subsets.
UZHFPV_SEQUENCES = {
    # indoor_forward
    "indoor_forward_3":  ("indoor_forward",  "forward"),
    "indoor_forward_5":  ("indoor_forward",  "forward"),
    "indoor_forward_6":  ("indoor_forward",  "forward"),
    "indoor_forward_7":  ("indoor_forward",  "forward"),
    "indoor_forward_9":  ("indoor_forward",  "forward"),
    "indoor_forward_10": ("indoor_forward",  "forward"),
    # indoor_45
    "indoor_45_2":   ("indoor_45",     "45"),
    "indoor_45_4":   ("indoor_45",     "45"),
    "indoor_45_9":   ("indoor_45",     "45"),
    "indoor_45_12":  ("indoor_45",     "45"),
    "indoor_45_13":  ("indoor_45",     "45"),
    "indoor_45_14":  ("indoor_45",     "45"),
    # outdoor_forward
    "outdoor_forward_1": ("outdoor_forward", "outdoor_forward"),
    "outdoor_forward_3": ("outdoor_forward", "outdoor_forward"),
    "outdoor_forward_5": ("outdoor_forward", "outdoor_forward"),
    # outdoor_45
    "outdoor_45_1": ("outdoor_45",    "outdoor_45"),
}


# Change-of-basis for the **camera axes only**:
#   OpenCV cam (x-right, y-down, z-forward) -> TartanAir cam (x-forward, y-right, z-down).
# Same matrix as `vi3.transforms.M_change_of_basis` / EuRoC's `_M_CAM`.
_M_CAM = np.array([[0., 0., 1.],
                   [1., 0., 0.],
                   [0., 1., 0.]], dtype=np.float64)


class UZHFPVRefinementDataset(RefinementDataset):
    """One UZH-FPV Snapdragon-left sequence in TartanAir's sample-dict shape."""

    # Snapdragon mono cam0 native resolution.
    NATIVE_W = 640
    NATIVE_H = 480

    # GT world frame: arbitrary VIO-init frame, roughly z-up. Gravity points
    # along world -z.
    WORLD_FRAME   = "world"
    GRAVITY_WORLD = (0.0, 0.0, -9.81)

    # Real Snapdragon MEMS IMU with time-varying bg/ba drift and noise
    # (per `imu.yaml`).
    # Biases are DERIVED by us (`_compute_or_load_biases`: central-difference
    # GT poses for omega, twice-difference positions for accel, minus gravity,
    # then smoothed) — pseudo-GT, not shipped. So not usable as a GT-bias init.
    BIAS_IS_GT = False

    # Continuous-time IMU noise densities from the Snapdragon
    # `imu-..._imu.yaml`:
    #   accelerometer_noise_density: 0.1     (m/s²/√Hz)
    #   gyroscope_noise_density:     0.05    (rad/s/√Hz)
    # These are the values Kalibr fit; consumed by `Grounder` for the
    # preintegrator covariance.
    IMU_ACC_NOISE_DENSITY  = 0.1
    IMU_GYRO_NOISE_DENSITY = 0.05

    # Moving-average window for bias smoothing. ~1 s at 500 Hz IMU.
    BIAS_SMOOTHING_WINDOW = 501  # odd, so the centered window is symmetric

    # Stationary-prefix skip: drop initial frames before the drone has moved
    # this far from its start position. FPV-drone sequences open with the
    # operator holding the drone idle before release; matches EuRoC's
    # pre-takeoff calibration window.
    DEFAULT_MOTION_THRESH_M = 0.10

    def __init__(self, base_path, sequence="indoor_forward_3",
                 mode="pad", cam="cam0",
                 model_spec=None, frame_stride=1):
        """
        base_path:              UZHFPV_ROOT (the extracted-data dir; see the
                                data README for the expected layout).
        sequence:               one of `UZHFPV_SEQUENCES.keys()`.
        cam:                    "cam0" only.
        model_spec:             vi3.models.ModelSpec; defaults to VGGT spec.
        frame_stride:           keep every k-th RGB frame (subsample the
                                per-frame cam arrays); the IMU is timestamp-
                                masked in `get_imu_segment`, so a strided pair's
                                segment accumulates across skipped frames, and
                                GT is timestamp-looked-up (unaffected). Mirrors
                                the EuRoC/TartanAir stride support.
        """
        super().__init__(mode=mode)
        if model_spec is None:
            model_spec = build_model_spec("vggt")
        self.frame_stride = max(1, int(frame_stride))
        self.model_spec = model_spec
        self._loader_mode = model_spec.preproc_mode
        if sequence not in UZHFPV_SEQUENCES:
            raise ValueError(
                f"Unknown UZH-FPV sequence {sequence!r}. "
                f"Known: {sorted(UZHFPV_SEQUENCES)}"
            )
        if cam != "cam0":
            raise NotImplementedError("Only cam0 (Snapdragon left) supported.")

        self.sequence = sequence
        family, group = UZHFPV_SEQUENCES[sequence]
        self.family = family
        self.group = group

        # Directory layout from the extractor:
        #   <base>/calib/<family>/camchain-imucam-..._imu.yaml
        #   <base>/<seq>/snapdragon/{groundtruth.txt, imu.txt,
        #                           left_images.txt, right_images.txt, img/}
        self.snap_dir = os.path.join(base_path, sequence, "snapdragon")
        self.cam_dir  = os.path.join(self.snap_dir, "img")
        self.calib_dir = os.path.join(base_path, "calib", family)
        if not os.path.isdir(self.snap_dir):
            raise FileNotFoundError(f"Snapdragon dir not found: {self.snap_dir}")
        if not os.path.isdir(self.calib_dir):
            raise FileNotFoundError(
                f"Calibration dir not found: {self.calib_dir}. "
                f"See the data README for the expected UZH-FPV layout."
            )

        # ----------------------------------------------------------------
        # 1. Calibration: cam0 intrinsics + distortion + T_cam_imu (cam1
        #    is loaded lazily inside `_setup_stereo`).
        # ----------------------------------------------------------------
        calib = _load_imucam_chain(self.calib_dir)
        cam0_cal = calib["cam0"]
        fx, fy, cx, cy = cam0_cal["intrinsics"]
        if cam0_cal["distortion_model"] != "equidistant":
            raise NotImplementedError(
                f"cam0 distortion model {cam0_cal['distortion_model']!r} "
                f"unsupported; UZH-FPV is equidistant (fisheye)."
            )
        if tuple(cam0_cal["resolution"]) != (self.NATIVE_W, self.NATIVE_H):
            raise RuntimeError(
                f"cam0 resolution mismatch: yaml has {cam0_cal['resolution']}, "
                f"expected ({self.NATIVE_W}, {self.NATIVE_H})."
            )
        self.NATIVE_FX, self.NATIVE_FY = float(fx), float(fy)
        self.NATIVE_CX, self.NATIVE_CY = float(cx), float(cy)
        self._dist_coeffs = np.array(cam0_cal["distortion_coeffs"],
                                      dtype=np.float64)  # k1..k4 (4 values)
        self._native_K = np.array([
            [self.NATIVE_FX, 0,              self.NATIVE_CX],
            [0,              self.NATIVE_FY, self.NATIVE_CY],
            [0,              0,              1],
        ], dtype=np.float64)
        # Fisheye undistortion → rectified pinhole at the SAME intrinsics
        # (the standard "undistort but keep K" trick — Kalibr fits this).
        self._undist_map1, self._undist_map2 = cv2.fisheye.initUndistortRectifyMap(
            self._native_K, self._dist_coeffs.reshape(4, 1),
            np.eye(3),
            self._native_K,
            (self.NATIVE_W, self.NATIVE_H), cv2.CV_32FC1,
        )

        # T_cam_imu (Kalibr convention: cam ← imu). Inverse is imu ← cam.
        # Body frame is FLU. Camera frame is OpenCV RDF. We want the
        # rotation that takes a *body* vector into the TartanAir cam
        # frame (FRD) — that's the same composition pattern as EuRoC.
        T_cam0_imu = np.array(cam0_cal["T_cam_imu"], dtype=np.float64)
        # IMU body → cam0 (OpenCV RDF) rotation.
        R_body_to_cam0 = T_cam0_imu[:3, :3]
        # IMU body → cam (TartanAir FRD) rotation.
        R_body_to_cam_tartan = _M_CAM @ R_body_to_cam0
        self._R_body_to_cam_tartan = R_body_to_cam_tartan

        # ----------------------------------------------------------------
        # 2. CSV-style streams.
        # ----------------------------------------------------------------
        # imu.txt: # timestamp ang_vel(3) lin_acc(3)
        imu_raw = np.loadtxt(os.path.join(self.snap_dir, "imu.txt"),
                              comments="#", dtype=np.float64)
        # First column is sample index (an integer); keep cols 1-7.
        if imu_raw.shape[1] == 8:   # has index column at [:, 0]
            imu_ts = imu_raw[:, 1]
            gyro_body = imu_raw[:, 2:5]   # rad/s, body FLU
            acc_body  = imu_raw[:, 5:8]   # m/s², body FLU (gravity in)
        else:
            imu_ts = imu_raw[:, 0]
            gyro_body = imu_raw[:, 1:4]
            acc_body  = imu_raw[:, 4:7]

        # Rotate IMU samples into TartanAir cam axes.
        acc  = acc_body  @ R_body_to_cam_tartan.T
        gyro = gyro_body @ R_body_to_cam_tartan.T

        # Cam-IMU temporal calibration (Kalibr `timeshift_cam_imu`, per-family):
        # t_imu = t_cam + timeshift. Our IMU↔frame pairing masks `imu_ts` against
        # `cam_times`, so bring IMU stamps into the camera clock by subtracting the
        # timeshift (~8-17 ms; without it the per-frame IMU window is
        # mis-registered against the camera poses).
        ts_cam_imu = float(cam0_cal.get("timeshift_cam_imu", 0.0))
        imu_ts = imu_ts - ts_cam_imu

        # left_images.txt: # id timestamp image_name
        left_idx, cam_times, img_names = _load_image_list(
            os.path.join(self.snap_dir, "left_images.txt"),
        )

        # groundtruth.txt: # timestamp tx ty tz qx qy qz qw
        gt = np.loadtxt(os.path.join(self.snap_dir, "groundtruth.txt"),
                         comments="#", dtype=np.float64)
        gt_times    = gt[:, 0]
        gt_pos_W    = gt[:, 1:4]
        gt_quat_xyzw = gt[:, 4:8]

        # Use first timestamp seen anywhere as t0 → all streams float-seconds
        # relative to that. (UZH-FPV is already in seconds, but rebasing to 0
        # keeps numerics tight and matches our EuRoC normalization.)
        t0 = float(min(cam_times.min(), imu_ts.min(), gt_times.min()))
        cam_times = cam_times - t0
        imu_ts    = imu_ts - t0
        gt_times  = gt_times - t0

        # ----------------------------------------------------------------
        # 3. Build per-GT-row rotation matrices + camera-frame poses.
        # ----------------------------------------------------------------
        N_gt = len(gt_times)
        gt_R_W_body = R.from_quat(gt_quat_xyzw).as_matrix()      # [N_gt, 3, 3]
        # cam→world rotation, TartanAir cam axes (ASL convention):
        #   R_W_cam = R_W_body @ R_body_to_cam_tartan.T
        R_W_cam_tartan = np.einsum('nij,jk->nik',
                                    gt_R_W_body, R_body_to_cam_tartan.T)
        # Translation in world is the body position; the body-vs-cam0 lever-arm
        # (~3 mm) is left uncorrected, within the pose-evaluation noise floor.
        T_W_cam = np.empty((N_gt, 4, 4), dtype=np.float64)
        T_W_cam[:, :3, :3] = R_W_cam_tartan
        T_W_cam[:, :3, 3]  = gt_pos_W
        T_W_cam[:, 3, :3]  = 0.0
        T_W_cam[:, 3, 3]   = 1.0

        # Un-axis-swapped (OpenCV cam) version of the same poses.
        R_W_cam_opencv = np.einsum('nij,jk->nik',
                                    gt_R_W_body, np.linalg.inv(R_body_to_cam0))
        T_W_cam_opencv = np.empty_like(T_W_cam)
        T_W_cam_opencv[:, :3, :3] = R_W_cam_opencv
        T_W_cam_opencv[:, :3, 3]  = gt_pos_W
        T_W_cam_opencv[:, 3, :3]  = 0.0
        T_W_cam_opencv[:, 3, 3]   = 1.0

        # ----------------------------------------------------------------
        # 4. Derive GT velocity by central differences.
        # ----------------------------------------------------------------
        gt_vel_W = _central_diff_position(gt_pos_W, gt_times)     # [N_gt, 3]

        # ----------------------------------------------------------------
        # 5. Clip cam frames to the GT-covered window.
        # ----------------------------------------------------------------
        keep = (cam_times >= gt_times[0]) & (cam_times <= gt_times[-1])
        cam_times = cam_times[keep]
        img_names = [n for n, k in zip(img_names, keep) if k]
        if len(cam_times) == 0:
            raise RuntimeError(f"No camera frames inside GT window for {sequence!r}")

        # --- Skip stationary warmup prefix ---
        # Drop frames before the drone has moved DEFAULT_MOTION_THRESH_M from its
        # initial position (the frozen foundation model gives degenerate poses at zero
        # parallax). Matches the EuRoC convention.
        ref_pos = gt_pos_W[0]
        cam_gt_idx = np.clip(
            np.searchsorted(gt_times, cam_times), 0, N_gt - 1,
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
        img_names = img_names[first_moving:]

        # ----------------------------------------------------------------
        # 7. Derive per-IMU-sample biases by GT-pose differentiation.
        #    Cached on disk; window-size keyed.
        # ----------------------------------------------------------------
        self._gt_times   = gt_times
        self._gt_T_W_cam = T_W_cam
        self._gt_T_W_cam_opencv = T_W_cam_opencv
        self._gt_R_W_body = gt_R_W_body
        self._gt_pos_W   = gt_pos_W
        self._gt_vel_W   = gt_vel_W
        self.cam_times   = cam_times
        self.imu_times   = imu_ts
        self._imu_acc    = acc
        self._imu_gyro   = gyro

        gt_bias_acc, gt_bias_gyro = self._compute_or_load_biases(
            gt_R_W_body, gt_pos_W, gt_times,
            gyro_body, acc_body, imu_ts,
            R_body_to_cam_tartan,
        )
        # gt_bias_*: [N_imu, 3], already in TartanAir cam axes.
        self._gt_bias_acc  = gt_bias_acc
        self._gt_bias_gyro = gt_bias_gyro

        # ----------------------------------------------------------------
        # 8. Preprocessed K + image names.
        # ----------------------------------------------------------------
        self._post_hw = predict_post_hw(
            native_h=self.NATIVE_H, native_w=self.NATIVE_W,
            target_size=self.model_spec.target_size,
            patch_size=self.model_spec.patch_size,
            mode=self.model_spec.preproc_mode,
        )
        self.K = self._compute_K_preprocessed(post_hw=self._post_hw)
        self.img_names = img_names

        # frame_stride: subsample the per-frame arrays together so consecutive
        # dataset indices are `frame_stride` original frames apart. The IMU
        # (get_imu_segment) is masked by timestamp between the kept cam_times, so
        # it accumulates across the skipped frames; GT is timestamp-looked-up.
        if self.frame_stride > 1:
            k = self.frame_stride
            self.cam_times = self.cam_times[::k]
            self.img_names = self.img_names[::k]

        # ----------------------------------------------------------------
        # 9. Stereo SGBM (cam1) setup — lazy. Default ON for Snapdragon.
        # ----------------------------------------------------------------
        cam1_cal = calib["cam1"]
        self._cam1_cal = cam1_cal
        self._stereo_state = None
        # Per-sequence on-disk depth cache (keyed by post-preprocess size).
        self._depth_cache_dir = os.path.join(
            self.snap_dir, f"depth_cache_{self._post_hw[0]}x{self._post_hw[1]}",
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.cam_times)

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
    # Bias derivation: IMU − GT-pose-derivative. Done once at __init__,
    # cached to disk keyed by smoothing-window size.
    # ------------------------------------------------------------------
    def _compute_or_load_biases(self, gt_R_W_body, gt_pos_W, gt_times,
                                 gyro_body, acc_body, imu_ts,
                                 R_body_to_cam_tartan):
        cache_path = os.path.join(
            self.snap_dir, f"biases_cache_w{self.BIAS_SMOOTHING_WINDOW}.npz"
        )
        if os.path.isfile(cache_path):
            d = np.load(cache_path)
            return d["bias_acc_cam"], d["bias_gyro_cam"]

        # 1. ω_gt(t) via central differences of rotation matrices.
        omega_gt_body = _central_diff_rotation(gt_R_W_body, gt_times)

        # 2. a_world(t) via second central differences of position.
        a_world_gt = _second_central_diff_position(gt_pos_W, gt_times)

        # 3. Rotate to body frame, remove gravity.
        g_world = np.array(self.GRAVITY_WORLD, dtype=np.float64)
        # a_body = R_W_body^T @ (a_world - g_world)
        a_world_minus_g = a_world_gt - g_world
        a_body_gt = np.einsum('nji,nj->ni', gt_R_W_body, a_world_minus_g)

        # 4. Interpolate ω_gt and a_body_gt onto IMU timestamps. Out-of-range
        # IMU samples (pre-GT calibration window, post-GT landing) get NaN
        # so the bias estimate is undefined there. We zero them at the end —
        # those samples are never consumed by downstream loaders because
        # cam frames are clipped to GT coverage and `get_imu_segment` reads
        # only IMU samples that fall inside cam-frame intervals.
        in_gt = (imu_ts >= gt_times[0]) & (imu_ts <= gt_times[-1])
        omega_at_imu = np.zeros_like(gyro_body)
        a_body_at_imu = np.zeros_like(acc_body)
        omega_at_imu[in_gt]  = _linear_interp_3d(
            gt_times, omega_gt_body, imu_ts[in_gt]
        )
        a_body_at_imu[in_gt] = _linear_interp_3d(
            gt_times, a_body_gt, imu_ts[in_gt]
        )

        # 5. Subtract IMU readings to get per-sample bias estimate.
        bg_body = gyro_body - omega_at_imu
        ba_body = acc_body  - a_body_at_imu
        # Mask out-of-GT samples so they don't pollute the moving average.
        bg_body[~in_gt] = 0.0
        ba_body[~in_gt] = 0.0

        # 6. Smooth (centered moving-average over IMU samples inside the GT
        # window). The masked-zero samples outside GT contribute nothing
        # because the moving-average window inside the helper sees them as
        # zero and they're far enough from the active region that the
        # taper doesn't reach them after `BIAS_SMOOTHING_WINDOW//2` samples.
        bg_body_smooth = _moving_average(bg_body, window=self.BIAS_SMOOTHING_WINDOW)
        ba_body_smooth = _moving_average(ba_body, window=self.BIAS_SMOOTHING_WINDOW)
        # Re-mask: outside GT, the diagnostic should report 0, not a tapered
        # neighbor's value.
        bg_body_smooth[~in_gt] = 0.0
        ba_body_smooth[~in_gt] = 0.0

        # 7. Rotate to TartanAir cam axes (same frame as imu_acc/gyro).
        bias_gyro_cam = bg_body_smooth @ R_body_to_cam_tartan.T
        bias_acc_cam  = ba_body_smooth  @ R_body_to_cam_tartan.T

        # Sanity printout — restrict to in-GT samples (where the bias is
        # actually defined).
        import sys
        n_in = int(in_gt.sum())
        mean_bg = np.linalg.norm(bias_gyro_cam[in_gt], axis=1).mean()
        mean_ba = np.linalg.norm(bias_acc_cam [in_gt], axis=1).mean()
        print(f"[uzhfpv {self.sequence}] derived biases ({n_in}/{len(imu_ts)} samples in GT window):",
              file=sys.stderr)
        print(f"  mean |bg| = {mean_bg:.4f} rad/s (expect ~0.005-0.1)", file=sys.stderr)
        print(f"  mean |ba| = {mean_ba:.4f} m/s² (expect ~0.02-0.5)", file=sys.stderr)

        np.savez(cache_path,
                  bias_acc_cam=bias_acc_cam.astype(np.float64),
                  bias_gyro_cam=bias_gyro_cam.astype(np.float64))
        return bias_acc_cam.astype(np.float64), bias_gyro_cam.astype(np.float64)

    # ------------------------------------------------------------------
    # Stereo SGBM depth (Snapdragon cam0 + cam1) via cv2.fisheye.*.
    # ------------------------------------------------------------------
    def _setup_stereo(self):
        if self._stereo_state is not None:
            return self._stereo_state
        c1 = self._cam1_cal
        if c1["distortion_model"] != "equidistant":
            raise NotImplementedError(
                f"cam1 distortion model {c1['distortion_model']!r} unsupported."
            )
        c1_fx, c1_fy, c1_cx, c1_cy = c1["intrinsics"]
        K1 = np.array([[c1_fx, 0, c1_cx], [0, c1_fy, c1_cy], [0, 0, 1]],
                      dtype=np.float64)
        D1 = np.array(c1["distortion_coeffs"], dtype=np.float64)

        # T_cn_cnm1 is cam1 in cam0's frame (Kalibr convention).
        T_c0_c1 = np.array(c1["T_cn_cnm1"], dtype=np.float64)
        R_c0_c1 = np.ascontiguousarray(T_c0_c1[:3, :3])
        t_c0_c1 = np.ascontiguousarray(T_c0_c1[:3, 3])

        size = (self.NATIVE_W, self.NATIVE_H)
        # cv2.fisheye.stereoRectify signature: (K1, D1, K2, D2, imageSize,
        # R, T, flags, newImageSize, balance, fov_scale) → R1, R2, P1, P2, Q.
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            self._native_K, self._dist_coeffs.reshape(4, 1),
            K1, D1.reshape(4, 1),
            size,
            R_c0_c1, t_c0_c1.reshape(3, 1).copy(),
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=size,
            balance=0.0, fov_scale=1.0,
        )
        rmap0 = cv2.fisheye.initUndistortRectifyMap(
            self._native_K, self._dist_coeffs.reshape(4, 1),
            R1, P1, size, cv2.CV_32FC1,
        )
        rmap1 = cv2.fisheye.initUndistortRectifyMap(
            K1, D1.reshape(4, 1),
            R2, P2, size, cv2.CV_32FC1,
        )
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=5,
            P1=8 * 3 * 5 * 5, P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        fx_rect  = float(P1[0, 0])
        baseline = float(abs(t_c0_c1[0]))
        self._stereo_state = dict(
            rmap0=rmap0, rmap1=rmap1, sgbm=sgbm,
            fx_rect=fx_rect, baseline=baseline,
        )
        return self._stereo_state

    def _project_stereo_to_depth(self, idx):
        # Image names from `left_images.txt` are like `img/image_0_2150.png`.
        # The right image corresponds by index suffix → `image_1_2150.png`.
        left_name = self.img_names[idx]   # 'img/image_0_NNN.png'
        right_name = left_name.replace("image_0_", "image_1_")
        img0_path = os.path.join(self.snap_dir, left_name)
        img1_path = os.path.join(self.snap_dir, right_name)
        if not os.path.exists(img0_path) or not os.path.exists(img1_path):
            return None
        img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            return None
        st = self._setup_stereo()
        r0 = cv2.remap(img0, st["rmap0"][0], st["rmap0"][1],
                        interpolation=cv2.INTER_LINEAR)
        r1 = cv2.remap(img1, st["rmap1"][0], st["rmap1"][1],
                        interpolation=cv2.INTER_LINEAR)
        disp = st["sgbm"].compute(r0, r1).astype(np.float32) / 16.0
        depth_rect = np.zeros_like(disp)
        valid = disp > 0
        depth_rect[valid] = (st["fx_rect"] * st["baseline"]) / disp[valid]
        depth_rect[~np.isfinite(depth_rect)] = 0.0
        depth_rect[depth_rect > 50.0] = 0.0
        # Resize to preprocessed resolution — same dispatch as EuRoC.
        if self.model_spec.fixed_output_hw:
            return self.preprocess_depth(depth_rect).numpy()
        import torch.nn.functional as _F
        d = torch.from_numpy(depth_rect).unsqueeze(0).unsqueeze(0)
        d = _F.interpolate(d, size=self._post_hw, mode="nearest")
        return d.squeeze(0).squeeze(0).numpy()

    def _get_gt_depth(self, idx):
        # Filename-based cache key — left_images.txt names look like
        # `img/image_0_NNN.png`; we just use the NNN suffix as the cache key.
        leaf = os.path.basename(self.img_names[idx])
        cache_path = os.path.join(self._depth_cache_dir, f"{leaf}.npy")
        if os.path.exists(cache_path):
            depth = np.load(cache_path)
        else:
            depth = self._project_stereo_to_depth(idx)
            if depth is None:
                return None
            os.makedirs(self._depth_cache_dir, exist_ok=True)
            np.save(cache_path, depth.astype(np.float32))
        return torch.from_numpy(depth).float()

    # ------------------------------------------------------------------
    # IMU + GT segmentation (mirrors EuRoC).
    # ------------------------------------------------------------------
    def get_imu_segment(self, idx):
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

        imu_t_window = self.imu_times[imu_idxs]
        gt_idxs = np.clip(
            np.searchsorted(self._gt_times, imu_t_window),
            0, len(self._gt_times) - 1,
        )
        gt_vel = self._gt_vel_W[gt_idxs]                       # [M, 3]
        gt_pos = self._gt_T_W_cam[gt_idxs, :3, 3]              # [M, 3]
        gt_ori = np.empty((len(gt_idxs), 4), dtype=np.float32)
        for k, gi in enumerate(gt_idxs):
            gt_ori[k] = R.from_matrix(self._gt_T_W_cam[gi, :3, :3]) \
                          .as_quat().astype(np.float32)

        # Biases are indexed by IMU sample (already at IMU rate).
        gt_b_a = self._gt_bias_acc[imu_idxs]                   # [M, 3]
        gt_b_g = self._gt_bias_gyro[imu_idxs]                  # [M, 3]

        return {
            "imu_acc":       acc,
            "imu_gyro":      gyro,
            "imu_dts":       dts,
            "gt_acc_nograv": torch.zeros_like(acc),
            "gt_vel":        torch.from_numpy(gt_vel.astype(np.float32)),
            "gt_pos":        torch.from_numpy(gt_pos.astype(np.float32)),
            "gt_ori":        torch.from_numpy(gt_ori),
            "gt_bias_acc":   torch.from_numpy(gt_b_a.astype(np.float32)),
            "gt_bias_gyro":  torch.from_numpy(gt_b_g.astype(np.float32)),
        }

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        leaf = self.img_names[idx]   # 'img/image_0_NNN.png'
        img_path = os.path.join(self.snap_dir, leaf)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_und = cv2.remap(img_gray, self._undist_map1, self._undist_map2,
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_und, cv2.COLOR_GRAY2RGB)

        tmp_path = _tmp_image_path(self.sequence, os.path.basename(leaf))
        if not os.path.exists(tmp_path):
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            cv2.imwrite(tmp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        image = self.model_spec.load_fn(
            [tmp_path], mode=self._loader_mode,
        ).squeeze(0)

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
        gt_depth = self._get_gt_depth(idx)
        if gt_depth is not None:
            out["gt_depth"] = gt_depth
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_imucam_chain(calib_dir):
    """Load Kalibr's `camchain-imucam-...yaml`. Returns the nested dict
    with `cam0` / `cam1` entries augmented with the chosen distortion-model
    string (passed through verbatim from the YAML).
    """
    candidates = [
        f for f in os.listdir(calib_dir)
        if f.startswith("camchain-imucam-") and f.endswith("_imu.yaml")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No camchain-imucam-...yaml in {calib_dir}. "
            f"Files present: {os.listdir(calib_dir)}"
        )
    with open(os.path.join(calib_dir, candidates[0])) as f:
        chain = yaml.safe_load(f)
    return chain


def _load_image_list(path):
    """Parse `left_images.txt` of the form:
        # id timestamp image_name
        0 4878.802975732 img/image_0_0.png
    Returns (ids: np.ndarray[int], times: np.ndarray[float], names: List[str]).
    """
    ids, times, names = [], [], []
    with open(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            ids.append(int(parts[0]))
            times.append(float(parts[1]))
            names.append(parts[2])
    return (np.array(ids, dtype=np.int64),
            np.array(times, dtype=np.float64),
            names)


def _tmp_image_path(sequence, leaf):
    """Per-(sequence, frame) cache path for the undistorted RGB-3 image."""
    return os.path.join("/tmp/uzhfpv_undist_cache", sequence, leaf)
