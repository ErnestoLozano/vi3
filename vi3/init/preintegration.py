import torch
import torch.nn as nn
import pypose as pp

class IMUPreIntegrator(nn.Module):
    """Forster-style IMU preintegration.

    Args:
      acc_density:  accelerometer noise density (m/s²/√Hz). Used to populate
                    the covariance-propagation noise matrix `Q`, where the
                    math expects density² and multiplies by `dt` to give the
                    per-segment noise variance accumulated in time `dt`.
      gyro_density: gyroscope noise density (rad/s/√Hz). Same role.

    Outputs `delta_p, delta_v, delta_r` (state means) and `cov` (9×9
    propagated covariance). Only the state means are consumed by the grounders;
    `cov` is kept for uncertainty-weighted losses.
    """

    def __init__(self, acc_density=0.04, gyro_density=0.0035):
        super().__init__()
        self.acc_density  = acc_density
        self.gyro_density = gyro_density

    def forward(self, acc, gyro, dts, ba, bg, need_cov=True):
        """`ba`/`bg` may be a constant `[3]` (broadcast to every sample) or a
        per-sample `[N, 3]` time-varying bias aligned with `acc`/`gyro`. The
        per-sample form lets a GT-bias init feed the dataset's time-varying
        bias through unchanged.

        `need_cov=False` skips the covariance propagation (a large share of the cost)
        and returns `cov=None`; the state means are bit-identical."""
        device = acc.device
        N = acc.shape[0]
        # Per-sample bias indexing: a [N, 3] bias is indexed by sample, a [3]
        # constant is used as-is (broadcasts). `_b(x, i)` picks the right one.
        ba_ps = ba.dim() == 2
        bg_ps = bg.dim() == 2

        # Preintegrated terms init
        dR = pp.identity_SO3(1, device=device)
        dV = torch.zeros(1, 3, device=device)
        dP = torch.zeros(1, 3, device=device)

        # Covariance terms init. Q is a continuous-time noise PSD matrix
        # (density²); the propagation step below uses `Q * dt` to convert
        # to per-segment noise variance, which matches Forster (2015) Eq. 39.
        cov = torch.zeros((9, 9), device=device)
        Q = torch.eye(6, device=device)
        Q[:3, :3] *= self.acc_density  ** 2
        Q[3:, 3:] *= self.gyro_density ** 2

        pre_ps = [] # path visualization

        # Bias-correct the whole stream and precompute every incremental
        # rotation `Exp((w - bg)·dt)` in one batched call; the compose stays serial.
        acc_ub_all = acc - (ba if ba_ps else ba.unsqueeze(0))
        gyro_ub_all = gyro - (bg if bg_ps else bg.unsqueeze(0))
        dR_inc = pp.so3(gyro_ub_all * dts.unsqueeze(-1)).Exp()      # [N] SO3

        for i in range(N):
            # time delta
            dt = dts[i]
            # bias correction (per-sample bias if [N,3], else constant [3])
            acc_unbiased = acc_ub_all[i]
            gyro_unbiased = gyro_ub_all[i]

            # --- State Integration ---
            # IMPORTANT: dR * acc rotates the LOCAL acceleration into the
            # local preintegration frame.
            # Order matters (dR used in dP and dV not the updated one yet)
            dP = dP + dV * dt + 0.5 * (dR * acc_unbiased) * dt**2
            dV = dV + (dR * acc_unbiased) * dt
            dR = dR * dR_inc[i]

            pre_ps.append(dP.clone())
            if not need_cov:
                continue

            R = dR.matrix().squeeze(0)

            # --- Covariance Propagation ---
            # State ordering: [ rot | vel | pos ]
            F = torch.eye(9, device=device)
            # Rotation error propagation (first-order SO(3) approximation for now)
            F[0:3, 0:3] = torch.eye(3, device=device) - pp.vec2skew(gyro_unbiased * dt)
            # Velocity wrt rotation
            F[3:6, 0:3] = -R @ pp.vec2skew(acc_unbiased) * dt
            # Position wrt rotation
            F[6:9, 0:3] = -0.5 * R @ pp.vec2skew(acc_unbiased) * dt**2
            # Position wrt velocity
            F[6:9, 3:6] = torch.eye(3, device=device) * dt

            # Noise ordering: [ accel_noise | gyro_noise ]
            G = torch.zeros((9, 6), device=device)
            # Gyro noise -> rotation
            G[0:3, 3:6] = torch.eye(3, device=device) * dt
            # Accel noise -> velocity
            G[3:6, 0:3] = R * dt
            # Accel noise -> position
            G[6:9, 0:3] = 0.5 * R * dt**2

            # Covariance
            cov = (
                F @ cov @ F.transpose(-1, -2)
                + G @ (Q * dt) @ G.transpose(-1, -2)
            )

        out = {
            "delta_p": dP.squeeze(0),
            "delta_v": dV.squeeze(0),
            "delta_r": dR,
            "cov": cov if need_cov else None,
            "pre_paths": [p.squeeze(0) for p in pre_ps],
        }
        return out
    