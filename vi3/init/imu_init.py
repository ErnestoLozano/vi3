"""The GT-free visual-inertial initialization of a window, and its IMU targets.

`IMUInit` is the init block of vi3 — the part every grounding method
(``vi3.grounding_methods``) starts from. Given the window's frames, their IMU
streams and the frozen foundation model's feed-forward poses, it estimates

  * frame-0 gravity — GeoCalib on the first image, cross-checked by the LS-g
    consensus guard (``g_ls_reject_deg``) against the gravity the IMU and the
    vision trajectory jointly imply (``_gravity_consistency_guard``);
  * the gyro bias — in closed form from the gyro stream and the model's relative
    rotations (``init_bg_from_imu``; held at zero where it is not observable);
  * the metric scale S of the vision trajectory and the initial velocity v0 —
    from IMU velocity DIFFERENCES (v0-independent), ``v0 = S · v_bar_0`` with
    ``|v0|`` from all pairs (``_v0_from_vel_scale``);

and integrates the IMU from frame 0 with them into the target trajectory
``{"p", "q", "v0", "g_rel"}`` (frame-0-relative, metric). The ``gt`` preset swaps
in GT gravity, GT v0 and GT biases where the dataset has them (EuRoC) — see
``vi3.init.presets``.

Biases are FIXED — set once by the init (plain tensors, never trained), so the
targets are static. Frame-0 gravity has three sources (``init_g_source``):
``geocalib`` (``estimate_gravity_from_image``), ``imu`` (closed-form accelerometer
attitude, ``estimate_gravity_in_frame0``) and ``gt_r0``.
"""

import math

import pypose as pp
import torch

from vggt.utils.rotation import quat_to_mat

from vi3.init.preintegration import IMUPreIntegrator


class IMUInit:
    """Per-window IMU init + target trajectory (see the module docstring).

    Build one per window with the resolved config, then call `build(samples)`.
    The preintegrator's continuous-time noise densities are sourced from the
    dataset (``IMU_ACC_NOISE_DENSITY``, ``IMU_GYRO_NOISE_DENSITY``); they only
    affect the propagated covariance, which the init does not use.
    """

    # Fallback for datasets that don't declare noise densities.
    _FALLBACK_ACC_NOISE_DENSITY  = 0.2
    _FALLBACK_GYRO_NOISE_DENSITY = 0.02

    def __init__(self, model_base, dataset, config, device="cuda"):
        self.model_base = model_base
        self.dataset = dataset
        self.device = device
        # Feed-forward dtype for the model's poses — bf16 on Ampere+, fp16 elsewhere.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        # Frame-0 gravity source (default GeoCalib, GT-free):
        #   "geocalib" — single-image GeoCalib estimate (GT-free), checked by the
        #                LS-g consensus guard. See `estimate_gravity_from_image`.
        #   "imu"      — closed-form accelerometer attitude (GT-free). See
        #                `estimate_gravity_in_frame0`.
        #   "gt_r0"    — R0ᵀ · GRAVITY_WORLD (GT pose at frame 0; the `gt` init).
        # Resolved early enough to also feed the vel-scale v0 solve.
        self._init_g_source = str(config.get("init_g_source", "geocalib")).lower()
        if self._init_g_source not in ("gt_r0", "imu", "geocalib"):
            raise ValueError(
                f"init_g_source must be 'gt_r0', 'imu', or 'geocalib', "
                f"got {self._init_g_source!r}"
            )
        # Lazily-built, cached GeoCalib model (heavy + downloads weights once).
        self._geocalib_model = None
        # GeoCalib-gravity outlier guard: GeoCalib is rejected only when it is
        # farther from the converged LS-g gravities than the two LS-g seeds (IMU,
        # GeoCalib) are from each other. This angle is the floor of that per-window
        # bar, so a very tight consensus cannot reject GeoCalib for a few degrees.
        # See `_gravity_consistency_guard`.
        self._g_ls_reject_deg = float(config.get("g_ls_reject_deg", 20.0))
        # Source for v0: "vel_scale" (default, GT-free — recover the vision
        # trajectory's metric scale S from the IMU velocity differences, then
        # v0 = S · v_bar_0; see `_v0_from_vel_scale`) or "gt" (GT velocity).
        self._init_v0_pose_source = config.get("init_v0_pose_source", "vel_scale")
        if self._init_v0_pose_source not in ("gt", "vel_scale"):
            raise ValueError(
                f"init_v0_pose_source must be 'gt' or 'vel_scale', "
                f"got {self._init_v0_pose_source!r}"
            )
        # GT-bias init: set ba/bg from the dataset's GT bias fields, per-IMU-sample
        # (time-varying), but ONLY when the dataset's bias is true GT
        # (`BIAS_IS_GT`); elsewhere bias is held at zero. See
        # `_init_gt_biases_from_dataset`.
        self._init_gt_biases = bool(config.get("init_gt_biases", False))
        # Closed-form GT-free gyro-bias init (default on): recover bg from the
        # gyro stream + the foundation model's vision relative rotations (see
        # `vi3.init.bias`); held at zero on datasets whose bias is not observable
        # (`BIAS_IS_GT=False`). The GT-free replacement for `init_gt_biases` on
        # the rotation axis — bias=0 corrupts the gyro-integrated rotation
        # target. ba stays 0 (absorbed by α). Mutually exclusive with
        # `init_gt_biases`.
        self._init_bg_from_imu = bool(config.get("init_bg_from_imu", True))
        if self._init_bg_from_imu and self._init_gt_biases:
            raise ValueError(
                "init_bg_from_imu and init_gt_biases are mutually exclusive "
                "(both set the gyro bias)."
            )
        self._g_init_done  = False
        self._g_rel_cached = None
        self._g_rel_gt_cached = None
        self._bg_hat_cached = None
        self._bg_gt_cached  = None
        self._v0_logged    = False
        self._v0_mag_msg   = None
        # v0 estimate vs GT, cached at init for CSV export (any init source).
        # `_v0_hat`/`_v0_gt_cached` are frame-0-local [3] tensors; `_s_hat` is the
        # vel_scale factor (nan for the `gt` source). Reported by `diagnostics`.
        self._v0_hat        = None
        self._v0_gt_cached  = None
        self._s_hat         = float("nan")
        # --- Diagnostics (read-only; reported by `diagnostics`) ---
        # OLS vs TLS scale: `_v0_from_vel_scale` computes both. The gravity guard
        # caches the raw GeoCalib / IMU-accel / LS-g (IMU-seeded) gravities and its
        # decision.
        self._s_ols_cached  = float("nan")     # OLS scale (biased/attenuated)
        self._s_tls_cached  = float("nan")     # TLS scale (the one actually used)
        self._fit_cos_cached = float("nan")    # vision/IMU fit cosine (anti-align gate)
        self._g_geocalib_cached = None         # single-image GeoCalib gravity
        self._g_imu_cached      = None         # accel-attitude IMU gravity
        self._g_lsg_cached      = None         # LS-g fixed point (IMU-seeded, gA)
        self._guard_fired       = None         # 1.0 if GeoCalib rejected, else 0.0
        self._g_dist_cached     = None         # min(dA,dB): GeoCalib->nearest LS-g
        self._g_dseed_cached    = None         # d_seed: LS-g seed agreement (width)
        # Cached predicted poses (per-condition) so the vel-scale v0 solve doesn't
        # re-forward.
        self._predicted_poses = None
        # Per-window memo of the IMU preintegration (see `_preint_cached`).
        self._preint_cache = {}
        # Biases are fixed: set once by the init, held constant through refinement.
        self.ba = torch.zeros(3, device=self.device)
        self.bg = torch.zeros(3, device=self.device)
        # Optional per-frame time-varying GT bias: list (len == window) of [M_i, 3]
        # tensors aligned with each frame's IMU segment. Populated by
        # `_init_gt_biases_from_dataset` on a BIAS_IS_GT dataset; None otherwise.
        self._ba_per_frame = None
        self._bg_per_frame = None
        self._gt_biases_done = False
        self._bg_from_imu_done = False

        # v0 MAGNITUDE estimator (direction is always the g-free first-pair
        # v_bar_0). "robust_allpairs" (default): recover |v0| from ALL pairs by
        # backing each metric vision velocity S·v_bar[k] out to t=0 with the IMU
        # Δv gained since frame 0, then taking the median projection onto v0's
        # direction — same all-pairs robustness S already has. GT-free.
        # "first_pair": |v0| = S·‖v_bar_0‖, a single finite difference (noisier on
        # low-displacement windows).
        self._v0_mag_mode = str(
            config.get("v0_mag_mode", "robust_allpairs")).lower()

    @property
    def acc_noise_density(self) -> float:
        return float(getattr(self.dataset, "IMU_ACC_NOISE_DENSITY",
                              self._FALLBACK_ACC_NOISE_DENSITY))

    @property
    def gyro_noise_density(self) -> float:
        return float(getattr(self.dataset, "IMU_GYRO_NOISE_DENSITY",
                              self._FALLBACK_GYRO_NOISE_DENSITY))

    def diagnostics(self):
        """Init estimates vs their GT/reference, for the results CSV: v0, the
        scale S, gravity angle, gyro-bias error, and the gravity-guard internals.
        Each is absent when its source is GT itself or has no reference."""
        out = {}
        # v0 estimate (and S) vs GT.
        if self._v0_hat is not None:
            vh = self._v0_hat
            out["v0_hat_mag"] = float(vh.norm())
            out["v0_s_hat"] = float(self._s_hat)
            if self._v0_gt_cached is not None:
                vg = self._v0_gt_cached
                err = vh - vg
                gtmag = float(vg.norm())
                out["v0_gt_mag"] = gtmag
                out["v0_err"] = float(err.norm())
                out["v0_err_rel"] = float(err.norm() / (gtmag + 1e-12))
                cos = float((vh @ vg) / (vh.norm() * vg.norm() + 1e-12))
                out["v0_cos"] = max(-1.0, min(1.0, cos))
        # Gravity and gyro bias vs their GT/reference (absent when the source is
        # GT itself, or there is no reference).
        g_hat = self._g_rel_cached
        g_gt = self._g_rel_gt_cached
        if g_hat is not None and g_gt is not None:
            cos = float((g_hat @ g_gt) / (g_hat.norm() * g_gt.norm() + 1e-12))
            out["g_angle_deg"] = math.degrees(
                math.acos(max(-1.0, min(1.0, cos))))
            out["g_err_rel"] = float(
                (g_hat - g_gt).norm() / (g_gt.norm() + 1e-12))
        bg_hat = self._bg_hat_cached
        bg_gt = self._bg_gt_cached
        if bg_hat is not None and bg_gt is not None:
            out["bg_err"] = float((bg_hat - bg_gt).norm())
            out["bg_err_rel"] = float(
                (bg_hat - bg_gt).norm() / (bg_gt.norm() + 1e-12))
        # Scale-fit and gravity-guard internals: OLS vs TLS scale, the fit cosine,
        # whether the guard rejected GeoCalib, and each gravity source's angle.
        if not math.isnan(self._s_ols_cached):
            out["init_s_ols"] = self._s_ols_cached
            out["init_s_tls"] = self._s_tls_cached
        if not math.isnan(self._fit_cos_cached):
            out["init_fit_cos"] = self._fit_cos_cached
        if self._guard_fired is not None:
            out["init_guard_fired"] = self._guard_fired
        if self._g_dist_cached is not None:
            out["init_g_dist"] = self._g_dist_cached
            out["init_g_dseed"] = self._g_dseed_cached
        g_gt = self._g_rel_gt_cached
        if g_gt is not None:
            def _ang_gt(gh):
                if gh is None:
                    return None
                c = float((gh @ g_gt) / (gh.norm() * g_gt.norm() + 1e-12))
                return math.degrees(math.acos(max(-1.0, min(1.0, c))))
            for key, gh in (("geocalib", self._g_geocalib_cached),
                            ("imu",      self._g_imu_cached),
                            ("lsg",      self._g_lsg_cached)):
                a = _ang_gt(gh)
                if a is not None:
                    out[f"init_g_{key}_deg"] = a
        return out

    # ------------------------------------------------------------------
    # Target generation
    # ------------------------------------------------------------------
    def build(self, samples):
        """Run the init on this window's samples; return the IMU targets."""
        # Bias init runs first (before any preintegration): either the GT-bias
        # path (true GT bias) or the closed-form GT-free gyro-bias estimate; both
        # set self.bg before targets integrate. Mutually exclusive (enforced in
        # `__init__`).
        if self._init_gt_biases and not self._gt_biases_done:
            self._init_gt_biases_from_dataset(samples)
            self._gt_biases_done = True
        if self._init_bg_from_imu and not self._bg_from_imu_done:
            self._init_bg_from_imu_closed_form(samples)
            self._bg_from_imu_done = True
        return self._build_imu_targets(samples)

    def _get_predicted_poses(self, samples):
        """Cached accessor: the un-adapted model's per-frame poses in the
        target-pose convention (`samples[i]["gt_pose"]` shape). First call runs
        the forward pass; subsequent calls within the same condition reuse it.
        """
        if self._predicted_poses is not None:
            return self._predicted_poses

        from vi3.transforms import pose_enc_to_target_pose_torch
        from vi3.models.pi3_adapter import set_forward_intrinsics

        images = torch.stack([s["image"].to(self.device) for s in samples], dim=0)
        # Pi3 adapter needs K + image_hw; no-op for VGGT / Omega.
        K = torch.stack([s["K"].to(self.device) for s in samples]) \
            if "K" in samples[0] else None
        set_forward_intrinsics(
            self.model_base, K, (images.shape[-2], images.shape[-1]),
        )
        was_training = self.model_base.training
        self.model_base.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            out = self.model_base(images.unsqueeze(0))
        self.model_base.train(was_training)
        pose_enc = out["pose_enc"][0].float()                  # [N, 9]
        self._predicted_poses = [pose_enc_to_target_pose_torch(p) for p in pose_enc]
        return self._predicted_poses

    @staticmethod
    def _preint_key(acc, gyro, dts, ba, bg, preintegrator):
        """Content signature for `_preint_cached` (shape + two reductions per
        input). `id()` is unusable: call sites pass fresh `.to(device)` copies."""
        def sig(t):
            return (tuple(t.shape), float(t.sum()), float(t.abs().sum()))
        return (sig(acc), sig(gyro), sig(dts), sig(ba), sig(bg),
                preintegrator.acc_density, preintegrator.gyro_density)

    def _preint_cached(self, preintegrator, acc, gyro, dts, ba, bg):
        """Memoized `IMUPreIntegrator.forward` for the init solves. The
        preintegrated terms depend only on (acc, gyro, dts, ba, bg), never on
        gravity or S, yet the gravity guard's fixed point re-integrates the same
        segments many times per window. One `IMUInit` (and memo) per window.
        """
        key = self._preint_key(acc, gyro, dts, ba, bg, preintegrator)
        hit = self._preint_cache.get(key)
        if hit is None:
            hit = preintegrator(acc, gyro, dts, ba, bg, need_cov=False)
            self._preint_cache[key] = hit
        return hit

    def _v0_from_vel_scale(self, samples, g_rel, device):
        """Vision-IMU v0 init (`init_v0_pose_source="vel_scale"`): all-pairs
        velocity-difference joint least-squares.

        Vision velocities up to scale (frame-0-local):
            v_bar_i = (p_vis_{i+1} - p_vis_i) / dt_i ,   Dv_bar_ij = v_bar_j - v_bar_i.
        Metric IMU velocity CHANGE per pair (v0 cancels in the difference):
            Dv_imu_ij = v_j - v_i = g*(t_j - t_i) + sum_{i<k<=j} R_{k-1} delta_v_k.
        A single scale S aligns them, solved jointly over ALL pairs i<j. Ordinary
        least squares (S = <Dvb,Dvi>/<Dvb,Dvb>) is attenuated toward 0 because the
        predictor Dv_bar is noisy (regression dilution), so we use TOTAL least
        squares (errors-in-variables), the geometric-mean scalar
            S = sign(<Dvb,Dvi>) * sqrt( sum||Dv_imu_ij||^2 / sum||Dv_bar_ij||^2 ),
        which is unbiased. The sign is kept from the OLS correlation so a real
        vision/IMU direction disagreement still flips v0. All pairs (not only
        consecutive ones) matter: long-baseline pairs carry Dv above the noise
        floor, while a consecutive Dv is a second difference of position. Then
        v0 = S * v_bar_0.

        v0's direction is the vision finite difference v_bar_0 (the foundation
        model's translation direction). The IMU chain mirrors the target loop in
        `_build_imu_targets`. Returns ``(v0, S)``.
        """
        preintegrator = IMUPreIntegrator(
            acc_density=self.acc_noise_density,
            gyro_density=self.gyro_noise_density,
        )
        predicted_poses = self._get_predicted_poses(samples)
        N = len(samples)
        vis_t = [predicted_poses[i][:3].to(device) for i in range(N)]

        dt0 = max(float(samples[1]["imu_dts"].sum()), 1e-9)
        v_bar_0 = (vis_t[1] - vis_t[0]) / dt0

        # Per-pair(consecutive) vision velocity v_bar[k] (pair k = frames k,k+1)
        # and metric IMU velocity change dv_imu[k] = g*dt + R_k * delta_v -- same
        # terms and order as the target loop's velocity update.
        v_bar, dv_imu = [], []
        R_i = pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))  # frame-0 -> i
        for i in range(1, N):
            acc_i  = samples[i]["imu_acc"].to(device)
            gyro_i = samples[i]["imu_gyro"].to(device)
            dts_i  = samples[i]["imu_dts"].to(device)
            if self._ba_per_frame is not None:
                ba_i, bg_i = self._ba_per_frame[i], self._bg_per_frame[i]
            else:
                ba_i, bg_i = self.ba, self.bg
            pre = self._preint_cached(preintegrator, acc_i, gyro_i, dts_i,
                                      ba_i, bg_i)
            dt = dts_i.sum()
            dtf = max(float(dt), 1e-9)
            v_bar.append((vis_t[i] - vis_t[i - 1]) / dtf)
            dv_imu.append(g_rel * dt + R_i * pre["delta_v"])
            R_i = R_i * pre["delta_r"].squeeze(0)

        # All-pairs scale S via total least squares: both Dv_bar and Dv_imu are
        # noisy; for a scalar through the origin TLS is the geometric mean
        # sqrt(<Dvi,Dvi>/<Dvb,Dvb>). The sign comes from the OLS correlation.
        # dv_imu over pair (i,j) is the cumulative sum of dv_imu[i+1..j].
        corr = num_bb = num_ii = 0.0
        for i in range(len(v_bar)):
            acc = torch.zeros(3, device=device)
            for j in range(i + 1, len(v_bar)):
                acc = acc + dv_imu[j]                 # Dv_imu_ij
                dvb = v_bar[j] - v_bar[i]             # Dv_bar_ij
                corr   += float((dvb * acc).sum())    # <Dvb, Dvi>  (sign)
                num_bb += float((dvb * dvb).sum())    # <Dvb, Dvb>
                num_ii += float((acc * acc).sum())    # <Dvi, Dvi>
        if num_bb > 1e-12 and num_ii > 0.0:
            # Fit cosine: how well the vision and IMU velocity-difference directions
            # agree (GT-free; the estimator's own confidence), in [-1, +1].
            fit_cos = corr / float((num_bb * num_ii) ** 0.5)
            # Diagnostics: the OLS scale and the signed TLS scale, and fit_cos.
            _sgn = 1.0 if corr >= 0 else -1.0
            self._s_ols_cached = float(corr / num_bb)
            self._s_tls_cached = float(_sgn * (num_ii / num_bb) ** 0.5)
            self._fit_cos_cached = float(fit_cos)
            # Reject only genuine anti-alignment (fit_cos < -0.5): the vision
            # trajectory then disagrees with the inertial motion and its scale is
            # meaningless, so fall back to S=1 (the foundation model's own scale).
            # Otherwise, including fit_cos near 0, the TLS magnitude is used.
            if fit_cos < -0.5:
                s_hat = 1.0
                print(f"             [vel-scale] vision/IMU anti-aligned "
                      f"(fit_cos={fit_cos:+.3f} < -0.5) — rejecting estimate, "
                      f"S=1.0 (defer to foundation model scale).")
            else:
                s_hat = float((num_ii / num_bb) ** 0.5)   # TLS magnitude (sign +)
        else:
            s_hat = 1.0
            print("             [vel-scale] ill-conditioned (den~0); S=1.0")

        if self._v0_mag_mode == "robust_allpairs" and len(v_bar) >= 2:
            # DIRECTION: the gravity-free first-pair v_bar_0. MAGNITUDE: rather
            # than the single noisy first pair, recover ‖v0‖ from every pair. The metric vision
            # velocity at pair k is S·v_bar[k]; the IMU says velocity gained
            # since frame 0 is Σ_{0<m≤k} dv_imu[m], so v0 ≈ S·v_bar[k] − that
            # cumulative Δv. Project each estimate onto v0's direction and take
            # the median.
            v0_dir = v_bar_0 / (v_bar_0.norm() + 1e-9)
            acc_dv = torch.zeros(3, device=device)
            mags = []
            for k in range(len(v_bar)):
                acc_dv = acc_dv + dv_imu[k]              # Σ dv_imu[0..k]
                v0_k = s_hat * v_bar[k] - acc_dv         # back out to t=0
                mags.append(float((v0_k * v0_dir).sum()))  # component along dir
            mags.sort()
            m = len(mags)
            med = mags[m // 2] if m % 2 else 0.5 * (mags[m // 2 - 1] + mags[m // 2])
            # Sign self-consistency. The back-out subtracts Σ dv_imu = Σ(g·dt + R·Δv);
            # a wrong gravity can flip the back-projected v0_k and drive the median
            # negative, i.e. v0 pointing against the vision direction v0_dir. Keep
            # the all-pairs median only when it agrees in sign with the vision
            # motion (med > 0); otherwise fall back to the gravity-free first-pair
            # estimate S·v_bar_0.
            if med > 0.0:
                v0 = med * v0_dir
                mode_used = f"median (n={m} pairs)"
            else:
                v0 = s_hat * v_bar_0                      # back-out sign-inconsistent
                mode_used = "first_pair (back-out points against vision)"
            # The gravity guard re-runs this solve; only the last message is logged.
            self._v0_mag_msg = (
                f"             [vel-scale] v0 mag robust_allpairs: "
                f"first_pair |v0|={float((s_hat * v_bar_0).norm()):.2f}, "
                f"median={med:.2f} -> using {mode_used}, "
                f"|v0|={float(v0.norm()):.2f} m/s")
        else:
            v0 = s_hat * v_bar_0                          # first_pair fallback
        return v0, s_hat

    def _g_from_ls(self, samples, s_hat, device):
        """All-pairs LS for frame-0 gravity, given a metric scale ``s_hat`` and
        ba=0. The counterpart of ``_v0_from_vel_scale``: with S given, the single
        3-vector g is well constrained by the long-baseline pairs.

        Model (velocity-difference form, same terms as vel-scale):
            S · Δv_bar_ij  =  Δv_imu_ij  =  g·Δt_ij + Σ_{i<k≤j} R_{k-1}·Δv_k
        Rearranged, LINEAR in g:
            g·Δt_ij  =  S·Δv_bar_ij − Σ R·Δv_k    ≡  b_ij
        Stacking all pairs i<j: minimise Σ‖Δt_ij·g − b_ij‖² → closed form
            g = (Σ Δt_ij·b_ij) / (Σ Δt_ij²).
        The magnitude is then re-locked to ‖GRAVITY_WORLD‖ (the motion model
        recovers the direction; a residual accel bias only scales ‖·‖). It
        subtracts the vision motion (S·Δv_bar) rather than assuming a = −g, so it
        is as reliable as S. ba is held at 0. Returns a [3] gravity-down tensor, or
        zeros if unsolvable.
        """
        preintegrator = IMUPreIntegrator(
            acc_density=self.acc_noise_density,
            gyro_density=self.gyro_noise_density,
        )
        predicted_poses = self._get_predicted_poses(samples)
        N = len(samples)
        if N < 3:
            return torch.zeros(3, device=device)
        vis_t = [predicted_poses[i][:3].to(device) for i in range(N)]

        # Per-frame: metric vision velocity S·v_bar[k], the rotation-transported IMU
        # velocity change R·Δv (NO gravity term — that's what we solve for), and dt.
        s_vbar, rdv, dts = [], [], []
        R_i = pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))  # frame-0 -> i
        for i in range(1, N):
            acc_i  = samples[i]["imu_acc"].to(device)
            gyro_i = samples[i]["imu_gyro"].to(device)
            dts_i  = samples[i]["imu_dts"].to(device)
            if self._ba_per_frame is not None:
                ba_i, bg_i = self._ba_per_frame[i], self._bg_per_frame[i]
            else:
                ba_i, bg_i = self.ba, self.bg
            pre = self._preint_cached(preintegrator, acc_i, gyro_i, dts_i,
                                      ba_i, bg_i)
            dt = dts_i.sum()
            dtf = max(float(dt), 1e-9)
            s_vbar.append(s_hat * (vis_t[i] - vis_t[i - 1]) / dtf)
            rdv.append((R_i * pre["delta_v"]).squeeze(0))
            dts.append(dt)
            R_i = R_i * pre["delta_r"].squeeze(0)

        # All-pairs: b_ij = S·Δv_bar_ij − Σ R·Δv ; solve g = Σ Δt·b / Σ Δt².
        num = torch.zeros(3, device=device)
        den = 0.0
        M = len(s_vbar)
        for i in range(M):
            rdv_acc = torch.zeros(3, device=device)
            dt_acc = 0.0
            for j in range(i + 1, M):
                rdv_acc = rdv_acc + rdv[j]
                dt_acc = dt_acc + float(dts[j])
                dvbar_ij = s_vbar[j] - s_vbar[i]          # S·Δv_bar_ij
                b_ij = dvbar_ij - rdv_acc                 # = g·Δt_ij
                num = num + dt_acc * b_ij
                den = den + dt_acc * dt_acc
        if den <= 1e-12:
            return torch.zeros(3, device=device)
        g_dir = num / den
        norm_g = g_dir.norm() + 1e-12
        g_mag = float(torch.tensor(self.dataset.GRAVITY_WORLD).norm())
        return g_dir * (g_mag / norm_g)

    def _g_s_fixed_point(self, samples, g_init, device, n_iter=5):
        """Alternate (S given g) ⇄ (g given S) to a fixed point. Starts from
        ``g_init``, estimates S via the existing vel-scale solve, re-estimates g via
        ``_g_from_ls``, repeats. Returns ``(g, s_hat, traj)`` where traj is the list
        of per-iter (S, g) for diagnostics. S and g are coupled through different
        projections (S from magnitudes, g from the vector residual), so the
        iteration can move away from its seed rather than just confirming it."""
        g = g_init
        s_hat = 1.0
        traj = []
        for _ in range(n_iter):
            _v0, s_new = self._v0_from_vel_scale(samples, g, device)
            g_new = self._g_from_ls(samples, s_new, device)
            if g_new.norm() < 1e-6:
                break
            converged = False
            if g.norm() > 1e-6:
                cos = float(torch.clamp(
                    (g / g.norm() * (g_new / g_new.norm())).sum(), -1.0, 1.0))
                converged = cos > 0.99999
            g, s_hat = g_new, s_new
            traj.append((s_hat, g.clone()))
            if converged:
                break
        return g, s_hat, traj

    # ------------------------------------------------------------------
    # GT-bias initialization (true GT only)
    # ------------------------------------------------------------------
    def _init_gt_biases_from_dataset(self, samples):
        """Set ba/bg from the dataset's GT bias fields — per-IMU-sample
        (time-varying) — but only when the dataset's bias is true GT.

        On a dataset with `BIAS_IS_GT == False` (bias not measured, e.g. UZH-FPV)
        the bias is held at zero and a note is printed, so a "GT init" never
        silently absorbs pseudo-GT. On a BIAS_IS_GT dataset we keep the full
        per-frame `[M_i, 3]` segments (consumed by the preintegration call) and
        also set scalar `self.ba`/`self.bg` to the window mean for single-constant
        consumers.
        """
        fmt = lambda v: '[' + ', '.join(f'{x:+.5f}' for x in v.tolist()) + ']'
        ds_name = type(self.dataset).__name__
        if not getattr(self.dataset, "BIAS_IS_GT", False):
            self._ba_per_frame = None
            self._bg_per_frame = None
            print(f"             [note] init_gt_biases on {ds_name} "
                  f"(BIAS_IS_GT=False): bias is derived (pseudo-GT), "
                  f"holding ba = bg = 0.")
            return

        ba_pf, bg_pf, ba_means, bg_means = [], [], [], []
        for s in samples:
            b_a = s.get("gt_bias_acc")
            b_g = s.get("gt_bias_gyro")
            if b_a is None or b_g is None:
                raise ValueError(
                    f"init_gt_biases: {ds_name} declares BIAS_IS_GT=True but a "
                    f"sample is missing gt_bias_acc/gt_bias_gyro.")
            b_a = b_a.to(self.device).float()
            b_g = b_g.to(self.device).float()
            # Normalize to [M_i, 3] (a constant-per-frame bias may arrive [3]).
            if b_a.dim() == 1:
                b_a = b_a.unsqueeze(0)
            if b_g.dim() == 1:
                b_g = b_g.unsqueeze(0)
            ba_pf.append(b_a)
            bg_pf.append(b_g)
            ba_means.append(b_a.mean(dim=0))
            bg_means.append(b_g.mean(dim=0))

        self._ba_per_frame = ba_pf
        self._bg_per_frame = bg_pf
        ba_mean = torch.stack(ba_means).mean(dim=0)
        bg_mean = torch.stack(bg_means).mean(dim=0)
        with torch.no_grad():
            self.ba.copy_(ba_mean)
            self.bg.copy_(bg_mean)
        print(f"             init_gt_biases on {ds_name} (BIAS_IS_GT=True): "
              f"per-sample GT bias applied.")
        print(f"             ba_gt (win-mean) = {fmt(ba_mean)}")
        print(f"             bg_gt (win-mean) = {fmt(bg_mean)}")

    def _resolve_gravity(self, samples, gravity, g_rel_gt, device):
        """Frame-0-local gravity from the active `init_g_source`.

        Returns a [3] tensor (gravity-DOWN, frame-0 cam axes, ‖g‖=‖GRAVITY_WORLD‖):
          "gt_r0"    → g_rel_gt (GT pose at frame 0).
          "imu"      → closed-form accelerometer attitude (uses self.bg/self.ba).
          "geocalib" → single-image GeoCalib estimate on frame 0.
        The GT-free sources log their angle vs g_rel_gt (when GT is available) via
        `_log_g_vs_gt`; computed once per condition (`_g_init_done` guard).
        """
        if self._init_g_source == "gt_r0":
            return g_rel_gt
        if self._g_init_done:                     # computed once per condition
            return self._g_rel_cached
        g_magnitude = float(gravity.norm())
        if self._init_g_source == "imu":
            from vi3.init.gravity import estimate_gravity_in_frame0
            g_rel = estimate_gravity_in_frame0(
                samples, self.bg, self.ba,
                g_magnitude=g_magnitude, device=device,
            )
        else:  # "geocalib"
            from vi3.init.gravity import estimate_gravity_from_image
            from vi3.models.geocalib_adapter import build_geocalib
            if self._geocalib_model is None:
                self._geocalib_model = build_geocalib(device=device)
            g_rel = estimate_gravity_from_image(
                samples[0]["image"], samples[0].get("K"),
                self._geocalib_model, g_magnitude=g_magnitude, device=device,
            )
        if g_rel_gt is not None:
            self._log_g_vs_gt(g_rel, g_rel_gt)
        self._g_rel_cached = g_rel
        self._g_rel_gt_cached = g_rel_gt      # GT reference for init-error export
        self._g_init_done = True
        return g_rel

    def _v0_consistency_score(self, samples, g_rel, device):
        """Gravity-quality signal: how the v0 all-pairs back-out under gravity
        ``g_rel`` agrees in sign with the gravity-free vision velocity. Returns the
        median / first-pair ratio: > 0 when the back-out points the way the vision
        moves, ≤ 0 when it points backwards (a wrong gravity flips Σ dv_imu).
        Independent of LS-g, which minimises the gravity residual instead. Same
        math as `_v0_from_vel_scale`, without the guard."""
        pre = IMUPreIntegrator(acc_density=self.acc_noise_density,
                               gyro_density=self.gyro_noise_density)
        pred = self._get_predicted_poses(samples)
        N = len(samples)
        vis_t = [pred[i][:3].to(device) for i in range(N)]
        dt0 = max(float(samples[1]["imu_dts"].sum()), 1e-9)
        v_bar_0 = (vis_t[1] - vis_t[0]) / dt0
        v_bar, dv_imu = [], []
        R_i = pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))
        for i in range(1, N):
            acc_i  = samples[i]["imu_acc"].to(device)
            gyro_i = samples[i]["imu_gyro"].to(device)
            dts_i  = samples[i]["imu_dts"].to(device)
            p = self._preint_cached(pre, acc_i, gyro_i, dts_i, self.ba, self.bg)
            dt = dts_i.sum()
            v_bar.append((vis_t[i] - vis_t[i - 1]) / max(float(dt), 1e-9))
            dv_imu.append(g_rel * dt + R_i * p["delta_v"])
            R_i = R_i * p["delta_r"].squeeze(0)
        # S under this gravity (magnitude-only TLS, matches the main solver).
        num_bb = num_ii = 0.0
        for i in range(len(v_bar)):
            acc = torch.zeros(3, device=device)
            for j in range(i + 1, len(v_bar)):
                acc = acc + dv_imu[j]
                dvb = v_bar[j] - v_bar[i]
                num_bb += float((dvb * dvb).sum())
                num_ii += float((acc * acc).sum())
        s_hat = float((num_ii / num_bb) ** 0.5) if num_bb > 1e-12 else 1.0
        v0_dir = v_bar_0 / (v_bar_0.norm() + 1e-9)
        acc_dv = torch.zeros(3, device=device)
        mags = []
        for k in range(len(v_bar)):
            acc_dv = acc_dv + dv_imu[k]
            mags.append(float(((s_hat * v_bar[k] - acc_dv) * v0_dir).sum()))
        mags.sort()
        mm = len(mags)
        med = mags[mm // 2] if mm % 2 else 0.5 * (mags[mm // 2 - 1] + mags[mm // 2])
        fp = float((s_hat * v_bar_0).norm())
        return med / fp if fp > 1e-9 else 0.0

    def _gravity_consistency_guard(self, samples, g_geocalib, v0_gc, s_gc, device):
        """Reject a grossly wrong GeoCalib gravity, by angle (independent of the
        scale S, which also absorbs the foundation model's own scale).

        GeoCalib stays the primary source; as an independent, inertial check, the
        g⇄S fixed point (`_g_s_fixed_point`) is run from two
        seeds, the closed-form IMU gravity (gA) and GeoCalib itself (gB). On an
        informative window both converge to the same direction. GeoCalib is
        rejected only when it lies farther from both converged gravities than they
        lie from each other, with a floor of `_g_ls_reject_deg` (default 20°); then
        the IMU-seeded gravity gA and its scale are adopted, unless the v0 sign
        check (`_v0_consistency_score`) favours GeoCalib. Returns (v0, s_hat, g_rel).
        """
        from vi3.init.gravity import estimate_gravity_in_frame0
        g_mag = float(g_geocalib.norm())
        g_imu = estimate_gravity_in_frame0(
            samples, self.bg, self.ba, g_magnitude=g_mag, device=device)
        if float(g_imu.norm()) < 1e-6:          # IMU gravity unavailable → keep GC
            return v0_gc, s_gc, g_geocalib

        def _ang(a, b):                          # angle between two [3] gravities
            na, nb = a.norm(), b.norm()
            if na < 1e-9 or nb < 1e-9:
                return 180.0
            c = float(torch.clamp((a / na * (b / nb)).sum(), -1.0, 1.0))
            return math.degrees(math.acos(c))

        # Two independent seeds → converged LS-g gravity each.
        gA, sA, _ = self._g_s_fixed_point(samples, g_imu, device)       # IMU-seeded
        gB, sB, _ = self._g_s_fixed_point(samples, g_geocalib, device)  # GC-seeded
        dA, dB = _ang(g_geocalib, gA), _ang(g_geocalib, gB)
        # Diagnostics: the candidate gravities (angle vs GT in `diagnostics`).
        # gA = LS-g (IMU-seeded, the one adopted on reject).
        self._g_geocalib_cached = g_geocalib.detach().clone()
        self._g_imu_cached      = g_imu.detach().clone()
        self._g_lsg_cached      = gA.detach().clone()
        # The guard rejects iff min(dA,dB) > max(d_seed, thr); both distances are
        # kept for the diagnostics.
        self._g_dist_cached  = float(min(dA, dB))     # GeoCalib's dist to nearest LS-g
        self._g_dseed_cached = float(_ang(gA, gB))    # consensus width (seed agreement)
        # Consensus outlier rule. The two fixed points' disagreement
        # d_seed = ∠(gA, gB) is the consensus width: how tightly the inertial
        # evidence pins gravity in this window. GeoCalib is the outlier when it sits
        # farther from the consensus than the consensus is wide (floored at
        # `_g_ls_reject_deg`). The raw accelerometer gravity is only a seed, not the
        # reference; the reference is the converged LS-g gravity.
        d_seed = _ang(gA, gB)
        consensus_w = max(d_seed, self._g_ls_reject_deg)
        if min(dA, dB) > consensus_w:
            # Sign safety: never swap TO a gravity whose v0 back-out points backwards.
            c_gc = self._v0_consistency_score(samples, g_geocalib, device)
            c_ls = self._v0_consistency_score(samples, gA, device)
            if c_gc > 0.0 >= c_ls:
                print(f"             [gravity-guard] reject VETOED: GeoCalib "
                      f"{min(dA,dB):.0f}° from LS-g (consensus width {consensus_w:.0f}°) "
                      f"but its v0 back-out is consistent (c_gc={c_gc:+.2f}) while LS-g "
                      f"is not (c_ls={c_ls:+.2f}) — keeping GeoCalib.")
                self._guard_fired = 0.0          # vetoed -> GeoCalib kept
                return v0_gc, s_gc, g_geocalib
            v0_ls, s_ls = self._v0_from_vel_scale(samples, gA, device)  # IMU-seeded
            self._g_rel_cached = gA                  # keep init-error export honest
            print(f"             [gravity-guard] GeoCalib rejected as outlier: "
                  f"{min(dA,dB):.1f}° from LS-g > consensus width {consensus_w:.1f}° "
                  f"(seeds agree to {d_seed:.1f}°) — adopting IMU-seeded LS-g "
                  f"(S {s_gc:.2f}→{s_ls:.2f}).")
            self._guard_fired = 1.0              # GeoCalib rejected -> LS-g adopted
            return v0_ls, s_ls, gA
        self._guard_fired = 0.0                  # GeoCalib kept (inside consensus)
        return v0_gc, s_gc, g_geocalib

    def _init_bg_from_imu_closed_form(self, samples):
        """Set self.bg from the closed-form GT-free gyro-bias estimate.

        Uses the gyro stream + the frozen foundation model's vision relative rotations
        (no GT pose). ba is left at zero (accel bias is absorbed by α). See
        `vi3.init.bias.estimate_gyro_bias_window`.

        Only estimated where the dataset declares an observable gyro bias
        (`BIAS_IS_GT`, e.g. EuRoC): the closed form needs a bias well above its
        discretization noise. Elsewhere (TartanAir: bias-free by construction;
        UZH-FPV: small bias) bg is held at 0.
        """
        if not getattr(self.dataset, "BIAS_IS_GT", False):
            print("             init_bg_from_imu: dataset bias not observable "
                  f"(BIAS_IS_GT=False on {type(self.dataset).__name__}) — "
                  "holding bg=0.")
            return
        from vi3.init.bias import estimate_gyro_bias_window
        predicted_poses = self._get_predicted_poses(samples)
        bg_hat = estimate_gyro_bias_window(
            samples, self.device, predicted_poses=predicted_poses)
        with torch.no_grad():
            self.bg.copy_(bg_hat)
        self._bg_hat_cached = bg_hat.detach()     # for init-error export
        fmt = lambda v: '[' + ', '.join(f'{x:+.5f}' for x in v.tolist()) + ']'
        print(f"             init_bg_from_imu (GT-free, vision R_ij): "
              f"bg_hat = {fmt(bg_hat)}")
        # Scored only against a true GT bias (BIAS_IS_GT); bg_err is absent
        # wherever there is no such reference.
        if getattr(self.dataset, "BIAS_IS_GT", False):
            bg_gt = [s.get("gt_bias_gyro") for s in samples]
            if all(b is not None for b in bg_gt):
                bg_gt_mean = torch.stack([
                    (b.to(self.device).float().unsqueeze(0) if b.dim() == 1
                     else b.to(self.device).float()).mean(dim=0)
                    for b in bg_gt]).mean(dim=0)
                self._bg_gt_cached = bg_gt_mean.detach()
                rel = (bg_hat - bg_gt_mean).norm() / (bg_gt_mean.norm() + 1e-12)
                print(f"             bg_gt (win-mean) = {fmt(bg_gt_mean)}  "
                      f"|err|/|gt| = {float(rel):.3f}")

    def _log_v0_vs_gt(self, v0_hat, v0_gt):
        """Compare estimated v0 against GT-derived v0 (frame-0 axes)."""
        err = v0_hat - v0_gt
        rel = err.norm() / (v0_gt.norm() + 1e-12)
        fmt = lambda v: '[' + ', '.join(f'{x:+.4f}' for x in v.tolist()) + ']'
        print(f"             v0_hat = {fmt(v0_hat)}  |v0_hat|={v0_hat.norm():.3f}")
        print(f"             v0_gt  = {fmt(v0_gt)}  |v0_gt|={v0_gt.norm():.3f}")
        print(f"             v0 err = {fmt(err)}  |err|={float(err.norm()):.4f}  "
              f"|err|/|gt|={float(rel):.3f}")

    def _log_g_vs_gt(self, g_hat, g_gt):
        """Compare estimated gravity-in-frame-0 to GT-derived reference."""
        cos = (g_hat @ g_gt) / (g_hat.norm() * g_gt.norm() + 1e-12)
        cos = cos.clamp(-1.0, 1.0)
        angle_deg = torch.rad2deg(torch.acos(cos))
        err_norm = (g_hat - g_gt).norm()
        fmt = lambda v: '[' + ', '.join(f'{x:+.4f}' for x in v.tolist()) + ']'
        print(f"             g_hat  = {fmt(g_hat)}")
        print(f"             g_gt   = {fmt(g_gt)}")
        print(f"             g err  = {fmt(g_hat - g_gt)}  "
              f"|err|={float(err_norm):.4f}  ∠={float(angle_deg):.2f}°")

    def _build_imu_targets(self, samples):
        """Forster preintegration -> frame-0-relative (p, q) targets.

        Steps:
          1. Preintegrate each IMU segment with the current biases.
          2. Compose rotations: q_i = delta_r_1 * delta_r_2 * ... * delta_r_i
          3. Bootstrap v0 (GT velocity, or the vel-scale GT-free estimate).
          4. Forward-integrate position with v0 + gravity + R_i * delta_p.

        Gravity is read from the dataset's `GRAVITY_WORLD` constant (world-down in
        that dataset's world frame) and rotated into frame-0-local coordinates via
        the GT camera-to-world rotation at frame 0, so integration is independent
        of the dataset's world convention.
        """
        device = self.device
        gravity = torch.tensor(self.dataset.GRAVITY_WORLD,
                               device=device, dtype=torch.float32)
        preintegrator = IMUPreIntegrator(
            acc_density=self.acc_noise_density,
            gyro_density=self.gyro_noise_density,
        )

        # R0_world (GT pose at frame 0) is needed by the `gt` v0 path, the `gt_r0`
        # gravity source, and the GT-comparison logs.
        need_R0_for_v0 = self._init_v0_pose_source == "gt"
        need_R0_for_g = self._init_g_source == "gt_r0"
        gt_pose_available = "gt_pose" in samples[0]
        R0_world = None
        if need_R0_for_v0 or need_R0_for_g or gt_pose_available:
            q0_world = samples[0]["gt_pose"][3:7].to(device)
            R0_world = quat_to_mat(q0_world / (torch.norm(q0_world) + 1e-12))

        # GT-derived frame-0 gravity (the GT-comparison reference, and the `gt_r0`
        # source). g_rel_gt = R0ᵀ · GRAVITY_WORLD.
        g_rel_gt = R0_world.t() @ gravity if R0_world is not None else None

        # Resolve the active frame-0 gravity first, so the vel-scale v0 solve
        # integrates against the same gravity as the targets.
        g_rel = self._resolve_gravity(samples, gravity, g_rel_gt, device)

        # v0 bootstrap.
        v0_gt = None
        if R0_world is not None and len(samples) > 1 and "gt_vel" in samples[1]:
            v0_gt = R0_world.t() @ samples[1]["gt_vel"][0].to(device)
        s_hat = float("nan")
        if self._init_v0_pose_source == "vel_scale" and len(samples) > 1:
            # Recover the vision trajectory's metric scale S from the IMU, then
            # v0 = S * v_bar_0, integrating against the ACTIVE gravity `g_rel`.
            v0, s_hat = self._v0_from_vel_scale(samples, g_rel, device)
            # GeoCalib gravity outlier guard (geocalib source only): cross-check
            # against the LS-g gravity from two seeds and adopt it when GeoCalib is
            # the outlier. See `_gravity_consistency_guard`.
            if self._init_g_source == "geocalib":
                v0, s_hat, g_rel = self._gravity_consistency_guard(
                    samples, g_rel, v0, s_hat, device)
            if not self._v0_logged:
                if self._v0_mag_msg:
                    print(self._v0_mag_msg)
                print(f"             vel-scale init: S={s_hat:.4f}")
                if v0_gt is not None:
                    self._log_v0_vs_gt(v0, v0_gt)
                self._v0_logged = True
        else:
            v0_world = (samples[1]["gt_vel"][0].to(device)
                        if len(samples) > 1 else torch.zeros(3, device=device))
            v0 = R0_world.t() @ v0_world if R0_world is not None else v0_world

        # Cache the v0 estimate vs GT for `diagnostics`. `_s_hat`
        # is set only by the vel_scale branch (else nan).
        self._v0_hat = v0.detach()
        self._v0_gt_cached = v0_gt.detach() if v0_gt is not None else None
        self._s_hat = float(s_hat)

        # Compose IMU trajectory relative to frame 0.
        t_imu = [torch.zeros(3, device=device)]
        q_imu_so3 = [pp.SO3(torch.tensor([0., 0., 0., 1.], device=device))]
        v_i = v0
        for i in range(1, len(samples)):
            acc_i  = samples[i]["imu_acc"].to(device)
            gyro_i = samples[i]["imu_gyro"].to(device)
            dts_i  = samples[i]["imu_dts"].to(device)
            # Per-sample GT bias for this segment when available (time-varying
            # GT-bias init); else the constant self.ba/bg. The preintegrator
            # accepts either [M_i, 3] or [3].
            if self._ba_per_frame is not None:
                ba_i, bg_i = self._ba_per_frame[i], self._bg_per_frame[i]
                if ba_i.shape[0] != acc_i.shape[0]:
                    raise ValueError(
                        f"GT-bias segment length {ba_i.shape[0]} != IMU segment "
                        f"length {acc_i.shape[0]} at frame {i}.")
            else:
                ba_i, bg_i = self.ba, self.bg
            pre = preintegrator(acc_i, gyro_i, dts_i, ba_i, bg_i)
            dt = dts_i.sum()
            R_i = q_imu_so3[-1]

            # Full position delta in frame-0 world: v_i*dt + 0.5*g*dt^2 + R_i*delta_p
            p_next = t_imu[-1] + v_i * dt + 0.5 * g_rel * dt * dt + R_i * pre["delta_p"]
            # Rotation composition: R_{i+1} = R_i * delta_r
            q_next_so3 = R_i * pre["delta_r"].squeeze(0)
            # Velocity: v_{i+1} = v_i + g*dt + R_i * delta_v
            v_next = v_i + g_rel * dt + R_i * pre["delta_v"]

            t_imu.append(p_next)
            q_imu_so3.append(q_next_so3)
            v_i = v_next

        q_imu = [q.tensor() for q in q_imu_so3]
        return {"p": t_imu, "q": q_imu, "v0": v0, "g_rel": g_rel}
