"""Named IMU-init presets.

`IMUInit` has per-knob flags; the two supported init paths are exposed as
named presets so scripts say ``--init gt-free`` (the default) / ``--init gt``:

    gt-free — the paper recipe, no GT on any init axis: gravity from GeoCalib,
              checked by the LS-g consensus guard (``g_ls_reject_deg``); gyro
              bias in closed form from the gyro stream + vision rotations
              (``init_bg_from_imu``); v0 = S · v_bar_0, with the vision
              trajectory's metric scale S recovered from IMU velocity
              DIFFERENCES and |v0| from all pairs (``v0_mag_mode``). These are
              also the defaults of ``IMUInit``. See ``IMUInit._v0_from_vel_scale``.
    gt      — the GT-anchored reference: gravity from ``R0ᵀ·GRAVITY_WORLD`` (GT
              pose at frame 0), v0 from GT velocity, GT biases on datasets whose
              bias is true GT (``BIAS_IS_GT``: EuRoC), zero elsewhere.

Low-level flags still override, so escape hatches remain.
"""

from typing import Dict, Optional


PRESETS: Dict[str, Dict[str, object]] = {
    # The GT-anchored reference: GT gravity, GT v0, GT biases where the dataset
    # has them (EuRoC; held at zero elsewhere).
    "gt": dict(
        init_g_source="gt_r0",
        init_v0_pose_source="gt",
        init_bg_from_imu=False,
        init_gt_biases=True,
    ),
    # The fully GT-free recipe of the paper (= the `IMUInit` defaults, spelled out).
    "gt-free": dict(
        init_v0_pose_source="vel_scale",
        init_bg_from_imu=True,
        init_g_source="geocalib",
        v0_mag_mode="robust_allpairs",
        g_ls_reject_deg=20.0,
    ),
}


def resolve_preset(name: Optional[str]) -> Optional[Dict[str, object]]:
    """Look up a preset by name. Returns the dict of flag-overrides, or ``None``
    if ``name`` is None / empty. Raises ``KeyError`` (with the known list) on a
    typo.
    """
    if name is None or name == "":
        return None
    key = name.strip().lower()
    if key not in PRESETS:
        raise KeyError(
            f"Unknown init preset {name!r}. Known presets: {sorted(PRESETS)}."
        )
    return dict(PRESETS[key])


def list_presets() -> str:
    """Multi-line summary for CLI help text."""
    lines = ["Named IMU-init presets:"]
    for k, v in PRESETS.items():
        lines.append(f"  {k:<7} g={v['init_g_source']}  v0={v['init_v0_pose_source']}")
    return "\n".join(lines)
