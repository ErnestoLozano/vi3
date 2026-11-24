"""GT-free frame-0 gravity initializers.

Two image/IMU sources for the frame-0-local gravity vector (both drop-in for the
grounder's GT-derived `R0ᵀ · GRAVITY_WORLD`):

* `estimate_gravity_in_frame0` — from the IMU stream alone: rotation-transport the
  raw accelerometer samples back to frame 0 via the bias-corrected gyro chain and
  average. Direction from the IMU; magnitude locked to a known constant.
* `estimate_gravity_from_image` — from frame-0's image via GeoCalib (single-image
  gravity estimator). Direction from vision; magnitude locked to the constant.

Both return a `[3]` tensor in frame-0 camera axes (TartanAir convention),
gravity-DOWN, ‖g‖ = g_magnitude (9.81 m/s² on Earth).
"""

import torch
import pypose as pp

from vi3.transforms import M_change_of_basis


def estimate_gravity_in_frame0(samples, bg, ba, g_magnitude=9.81, device=None):
    """Closed-form gravity-direction estimate in frame-0-local coordinates.

    No GT pose dependency. Uses only the IMU stream (imu_acc, imu_gyro, imu_dts)
    and the initialized gyro bias `bg`.

    Math:
        Starting from R_{0→0} = I, walk per-IMU-sample across all frames after
        frame 0 (frame-0's IMU window is the warmup and is skipped):
            R_{0→k} = R_{0→k-1} · Exp((gyro_k - bg) · dt_k)
        Accumulate the rotation-transported, bias-corrected accelerometer:
            S = mean_k(R_{0→k} · (a_k - ba))
        The accelerometer at rest reads specific force = -g_local, so the
        IMU-implied gravity (in frame-0 coords) is:
            g_rel = -S · g_magnitude / ||S||
        Magnitude is locked to a known constant — robust to constant accel bias
        (which affects ||S|| but not its direction much).

    Args:
        samples: list of N sample dicts (the grounder's sample shape).
        bg, ba:  [3] tensors. Frame-0-local biases (the grounder's self.bg/self.ba).
        g_magnitude: known gravity magnitude in m/s² (9.81 on Earth).
        device:  where to assemble the result; defaults to bg.device.

    Returns:
        g_rel: [3] tensor on `device` with ||g_rel|| == g_magnitude. Same
               convention as `R0_world.t() @ Dataset.GRAVITY_WORLD`. Returns
               zeros (sentinel: "couldn't estimate") if no usable IMU samples.
    """
    if device is None:
        device = bg.device
    bg = bg.to(device)
    ba = ba.to(device)

    transported = []
    R_0_to_k = pp.identity_SO3(1, device=device)
    for j in range(1, len(samples)):
        acc_j  = samples[j]["imu_acc"].to(device)              # [L, 3]
        gyro_j = samples[j]["imu_gyro"].to(device)             # [L, 3]
        dts_j  = samples[j]["imu_dts"].to(device)              # [L]
        L = acc_j.shape[0]
        if L == 0 or float(dts_j.sum()) <= 0:
            continue
        for k in range(L):
            # Update R_{0→k} BEFORE using it: matches the preintegrator's
            # discretization convention (rotation accumulated through the k-th
            # sample interval is applied to the k-th sample's accel).
            R_0_to_k = R_0_to_k * pp.so3(
                ((gyro_j[k] - bg) * dts_j[k]).unsqueeze(0)
            ).Exp()
            a_unbiased = acc_j[k] - ba
            transported.append((R_0_to_k * a_unbiased).squeeze(0))

    if not transported:
        return torch.zeros(3, device=device)

    S = torch.stack(transported, dim=0).mean(dim=0)            # [3]
    norm_S = S.norm() + 1e-12
    return -S * (g_magnitude / norm_S)


def estimate_gravity_from_image(image, K, model, g_magnitude=9.81, device=None):
    """Frame-0-local gravity from a single image via GeoCalib.

    GeoCalib returns a unit gravity vector in OpenCV camera axes (x-right, y-down,
    z-forward), pointing UP (anti-gravity: [0,-1,0] at zero roll/pitch). We negate
    to gravity-DOWN, change basis OpenCV→TartanAir (the grounder's camera axes), and
    scale to `g_magnitude`. Because GeoCalib runs on frame-0's image, its output is
    already frame-0 camera-local — no R0 rotation is needed.

    Args:
        image: [3, H, W] (or [1, 3, H, W]) RGB image, range [0, 1] — the grounder's
            `samples[0]["image"]` is directly compatible (ToTensor, no norm).
        K: [3, 3] pixel-unit intrinsics or None. When given, K[0,0] is passed as a
            focal prior to GeoCalib; else GeoCalib estimates focal itself.
        model: a built GeoCalib instance (see `vi3.models.geocalib_adapter`).
        g_magnitude: gravity magnitude in m/s² (9.81 on Earth).
        device: where to assemble the result; defaults to the image's device.

    Returns:
        g_rel: [3] tensor, ‖g_rel‖ == g_magnitude. Same convention as
               `R0ᵀ · GRAVITY_WORLD` (gravity-DOWN, frame-0 TartanAir cam axes).
    """
    if device is None:
        device = image.device
    img = image.to(device)
    if img.dim() == 4:
        img = img[0]

    priors = None
    if K is not None:
        priors = {"focal": torch.tensor(float(K[0, 0]), device=device)}

    out = model.calibrate(img, priors=priors)
    up = out["gravity"].vec3d[0].to(device)                    # OpenCV axes, UP
    g_down_opencv = -up                                        # gravity-DOWN
    M = M_change_of_basis(device=device, dtype=g_down_opencv.dtype)
    g_tartan = M @ g_down_opencv                               # → TartanAir axes
    return g_tartan * g_magnitude
