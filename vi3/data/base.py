"""Common interface and helpers for refinement datasets.

Subclasses (TartanAir, EuRoC, ...) implement `__getitem__` to return a
sample dict matching the contract below. Common preprocessing utilities live
here so each new dataset doesn't have to reimplement them.

Sample-dict contract
--------------------
Required:
    image: torch.FloatTensor [3, TARGET_SIZE, TARGET_SIZE], padded/cropped

Optional (presence enables certain grounders / supervisions):
    gt_pose:      torch.FloatTensor [7] = [tx, ty, tz, qx, qy, qz, qw]
                  camera->world pose, in the dataset's world frame
    gt_depth:     torch.FloatTensor [TARGET_SIZE, TARGET_SIZE], metric
    imu_acc:      torch.FloatTensor [M, 3], m/s² (M = imu samples in window)
    imu_gyro:     torch.FloatTensor [M, 3], rad/s
    imu_dts:      torch.FloatTensor [M], seconds; sum == cam_times[idx]-prev
    gt_vel:       torch.FloatTensor [M, 3], world-frame velocity
    gt_pos:       torch.FloatTensor [M, 3], world-frame position
    gt_ori:       torch.FloatTensor [M, 4], world-frame orientation [xyzw]
    gt_acc_nograv:torch.FloatTensor [M, 3], world-frame acc minus gravity
    gt_bias_acc:  torch.FloatTensor [M, 3], accelerometer bias per IMU sample
                  (TartanAir: per-trajectory constant broadcast over M;
                   EuRoC: time-varying drift from sensor calibration)
    gt_bias_gyro: torch.FloatTensor [M, 3], gyroscope bias per IMU sample

Camera convention for `gt_pose` is dataset-specific. The grounders assume
TartanAir's convention (x-forward, y-right, z-down camera axes in NED world);
new datasets must either match it or be wrapped by an adapter.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def predict_post_hw(native_h, native_w, target_size, patch_size, mode):
    """Predict the model's preprocessor output (H, W) from native dims.

    Lets datasets compute K eagerly without having to run the
    preprocessor on a real sample first.

    Matches the existing loaders:
      * "pad" (VGGT): output is `target_size × target_size` square.
      * "crop" (VGGT): output is `target_size × target_size` square.
      * "max_size" (VGGT-Omega): longest side → target_size; shorter
        side rounded to a multiple of patch_size. Errors on extreme
        aspect ratios (< 0.5 or > 2.0); Omega's loader does a center
        crop in that regime and we don't try to model it here.

    Returns (post_h, post_w).
    """
    if mode in ("pad", "crop"):
        return target_size, target_size

    if mode == "max_size":
        aspect_ratio = native_h / max(native_w, 1)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            raise NotImplementedError(
                f"Aspect ratio {aspect_ratio:.3f} would trigger Omega's "
                f"center-crop path in load_and_preprocess_images; we don't "
                f"model that yet. native (h,w)=({native_h},{native_w})."
            )
        if aspect_ratio >= 1.0:
            post_h = target_size
            post_w = max(patch_size,
                          int(round(target_size / aspect_ratio / patch_size)) * patch_size)
        else:
            post_w = target_size
            post_h = max(patch_size,
                          int(round(target_size * aspect_ratio / patch_size)) * patch_size)
        return post_h, post_w

    raise ValueError(f"Unknown preprocessor mode: {mode!r}")


def compute_K_preprocessed(native_fx, native_fy, native_cx, native_cy,
                            native_w, native_h, post_h, post_w,
                            patch_size, target_size, mode="pad"):
    """Derive the intrinsics matrix for an image after the model's preprocessor.

    Handles the two regimes our datasets support:
      * Fixed-output models (VGGT): the preprocessor resizes to a square
        `target_size × target_size` via pad/crop with a known offset.
      * Variable-output models (VGGT-Omega): the preprocessor picks a
        rectangular `post_h × post_w` per image (multiple of `patch_size`)
        with no padding. We use the actually-realized (post_h, post_w).

    Args:
        native_fx/fy: pre-preprocess focal length in pixels.
        native_cx/cy: pre-preprocess principal point in pixels.
        native_w/h:   pre-preprocess image dims.
        post_h/post_w: dims of the image as fed to the model.
        patch_size:    patch divisor (14 for VGGT, 16 for VGGT-Omega).
        target_size:   for fixed-output models only; the canonical
                       longest-side target (518/512). Ignored otherwise.
        mode:          "pad" or "crop" for VGGT; "max_size"/"balanced"
                       for VGGT-Omega. The math degenerates to "scale
                       (native_w, native_h) -> (post_w, post_h), no pad"
                       when the preprocessor doesn't pad.

    Returns:
        torch.FloatTensor [3, 3] — intrinsics for the post-preprocess image.
    """
    if mode == "pad":
        # VGGT-style: scale long edge to target_size, short edge to a
        # multiple of patch_size, then pad to target_size × target_size.
        if native_w >= native_h:
            new_W = target_size
            new_H = round(native_h * (new_W / native_w) / patch_size) * patch_size
        else:
            new_H = target_size
            new_W = round(native_w * (new_H / native_h) / patch_size) * patch_size
        pad_x = (target_size - new_W) // 2
        pad_y = (target_size - new_H) // 2
    elif mode == "crop":
        # VGGT crop mode: long edge → target_size, then center-crop H to
        # target_size if needed.
        new_W = target_size
        new_H = round(native_h * (new_W / native_w) / patch_size) * patch_size
        pad_x = 0
        pad_y = -((new_H - target_size) // 2) if new_H > target_size else 0
    else:
        # Variable-size models (Omega): no padding, post_h/post_w are the
        # exact realized dims. K scale factors derive directly from those.
        new_W, new_H = post_w, post_h
        pad_x, pad_y = 0, 0

    sx = new_W / native_w
    sy = new_H / native_h
    return torch.tensor([
        [native_fx * sx, 0.0,             native_cx * sx + pad_x],
        [0.0,            native_fy * sy,  native_cy * sy + pad_y],
        [0.0,            0.0,             1.0],
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Kinematic helpers shared across real-IMU dataset loaders (UZH-FPV, EuRoC).
# These derive velocity / angular-velocity / linear-acceleration from a
# discrete GT pose chain — needed when the upstream dataset ships pose-only
# GT (no velocity, no biases) and we want full state to feed the Grounder.
# ---------------------------------------------------------------------------

def _central_diff_position(p, t):
    """Per-sample central-difference velocity, with forward/backward
    differences at the endpoints. p: [N, 3], t: [N]. Returns [N, 3]."""
    import numpy as np
    N = len(t)
    v = np.zeros_like(p)
    if N >= 3:
        dt_central = (t[2:] - t[:-2])[:, None]
        v[1:-1] = (p[2:] - p[:-2]) / dt_central
    if N >= 2:
        v[0]  = (p[1]  - p[0])  / (t[1]  - t[0])
        v[-1] = (p[-1] - p[-2]) / (t[-1] - t[-2])
    return v


def _second_central_diff_position(p, t):
    """Per-sample second central-difference acceleration. Uses a uniform
    `dt` approximation (works well when GT is sampled at roughly uniform
    intervals; the proper non-uniform stencil is more code and gives
    indistinguishable results when `dt` variance is small).
    """
    import numpy as np
    N = len(t)
    a = np.zeros_like(p)
    if N >= 3:
        dt = np.median(np.diff(t))     # robust to occasional gaps
        a[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / (dt * dt)
        a[0]  = a[1]
        a[-1] = a[-2]
    return a


def _central_diff_rotation(R_mats, t):
    """Central-difference angular velocity (body frame) from a stack of
    rotation matrices `R_mats[k] = R_W_body[k]`.

    Formula:
        ω_body[k] = vee( R[k-1]^T R[k+1] ) / (t[k+1] - t[k-1])
    """
    import numpy as np
    N = len(t)
    omega = np.zeros((N, 3), dtype=np.float64)
    if N >= 3:
        for k in range(1, N - 1):
            M = R_mats[k - 1].T @ R_mats[k + 1]
            omega[k, 0] = 0.5 * (M[2, 1] - M[1, 2])
            omega[k, 1] = 0.5 * (M[0, 2] - M[2, 0])
            omega[k, 2] = 0.5 * (M[1, 0] - M[0, 1])
            dt = t[k + 1] - t[k - 1]
            if dt > 0:
                omega[k] /= dt
        omega[0]  = omega[1]
        omega[-1] = omega[-2]
    return omega


def _linear_interp_3d(t_src, v_src, t_dst):
    """Linearly interpolate a [N_src, 3] signal sampled at `t_src` onto
    `t_dst` timestamps. Out-of-range samples are clamped to the nearest
    endpoint (matches `np.interp`'s default)."""
    import numpy as np
    out = np.empty((len(t_dst), 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(t_dst, t_src, v_src[:, d])
    return out


def _moving_average(x, window):
    """Centered moving-average along axis 0. `x: [N, D]`. Endpoints use
    a shrinking window (no zero-padding artefacts).
    """
    import numpy as np
    N, D = x.shape
    if window < 2 or N < 2:
        return x.copy()
    if window > N:
        window = N if N % 2 == 1 else N - 1   # keep odd
    half = window // 2
    csum = np.zeros((N + 1, D), dtype=np.float64)
    csum[1:] = np.cumsum(x, axis=0)
    out = np.empty_like(x)
    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


class RefinementDataset(Dataset):
    """Base class for VGGT refinement datasets.

    Holds the shared image/depth resize+pad logic so subclasses only need to
    decide where the bytes come from. Subclasses must set `self.mode`
    (`"pad"` or `"crop"`) and `self.target_size` (default 518, VGGT-native).
    """

    TARGET_SIZE = 518  # VGGT-native resolution
    PATCH_SIZE = 14    # VGGT ViT patch divisor

    # Whether `gt_bias_acc` / `gt_bias_gyro` are TRUE ground-truth biases
    # (shipped/calibrated by the dataset) vs pseudo-GT we derive ourselves
    # (e.g. by differencing GT poses). Default False — a subclass must opt in
    # by setting BIAS_IS_GT = True. Consumed by the grounders'
    # `init_gt_biases`: the bias is only applied when this is True, else it is
    # held at zero. Keeps a "GT init" result from silently absorbing biases we
    # computed rather than measured.
    BIAS_IS_GT = False

    def __init__(self, mode="pad", target_size=TARGET_SIZE):
        if mode not in ("pad", "crop"):
            raise ValueError(f"mode must be 'pad' or 'crop', got {mode!r}")
        self.mode = mode
        self.target_size = target_size

    # ------------------------------------------------------------------
    # Depth preprocessing (shared)
    # ------------------------------------------------------------------
    def load_tartanair_depth(self, depth_path):
        """Load a TartanAir 4-channel 8-bit PNG that bit-casts to float32 depth.

        This is the TartanAir-specific decoding path; kept here because the
        same scheme is used by several Tartan-derived datasets. For datasets
        that use a different encoding, override `load_depth` in the subclass.
        """
        depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            raise ValueError(f"Could not load depth at {depth_path}")
        depth_decoded = depth_rgba.view("<f4")
        return np.squeeze(depth_decoded, axis=-1).astype(np.float32)

    def preprocess_depth(self, depth_raw):
        """Resize and pad/crop a HxW depth map to TARGET_SIZE.

        depth_raw: numpy [H, W] float32, in metric units.
        Returns: torch.FloatTensor [TARGET_SIZE, TARGET_SIZE].
        """
        depth = torch.from_numpy(depth_raw).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        h, w = depth.shape[-2:]

        # Resize so the long edge matches target_size, keeping multiples of
        # PATCH_SIZE for the short edge (matches VGGT image preprocessing).
        if self.mode == "pad":
            if w >= h:
                new_w = self.target_size
                new_h = round(h * (new_w / w) / self.PATCH_SIZE) * self.PATCH_SIZE
            else:
                new_h = self.target_size
                new_w = round(w * (new_h / h) / self.PATCH_SIZE) * self.PATCH_SIZE
        else:  # "crop"
            new_w = self.target_size
            new_h = round(h * (new_w / w) / self.PATCH_SIZE) * self.PATCH_SIZE

        # NEAREST to preserve raw depth values (no edge blending).
        depth = F.interpolate(depth, size=(new_h, new_w), mode="nearest")

        if self.mode == "crop" and new_h > self.target_size:
            start_y = (new_h - self.target_size) // 2
            depth = depth[:, :, start_y:start_y + self.target_size, :]

        if self.mode == "pad":
            h_padding = self.target_size - depth.shape[2]
            w_padding = self.target_size - depth.shape[3]
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            # 0.0 is best for masking during loss; large values would skew
            # any metric that uses the padded region.
            depth = F.pad(depth, (pad_left, pad_right, pad_top, pad_bottom),
                          mode="constant", value=0.0)

        return depth.squeeze()
