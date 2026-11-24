"""Domain-specific math utilities not covered by VGGT.

VGGT provides `quat_to_mat` and `mat_to_quat` in `vggt.utils.rotation`; use
those for quaternion <-> rotation-matrix conversions. The helpers here are
for things VGGT does not provide:

* Hamilton-product quaternion arithmetic (`quat_mul`, `quat_inv`)
* Geodesic angular distance between two quaternions (`rotation_geodesic_loss`)
* Procrustes (Umeyama) point-cloud alignment + structural RMSE
"""

import torch

from vggt.utils.rotation import quat_to_mat


def quat_inv(q):
    """Invert a unit quaternion [x, y, z, w]."""
    inv = q.clone()
    inv[:3] *= -1
    return inv / (torch.norm(inv) + 1e-12)


def quat_mul(q, r):
    """Hamilton product of two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = r
    product = torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])
    return product / (torch.norm(product) + 1e-12)


def rotation_geodesic_loss(q_pred, q_gt):
    """Geodesic angular distance between two unit quaternions [x, y, z, w].

    Input shape: [4] (no batch dimension).
    Returns the angle in radians.
    """
    q1 = q_pred / torch.norm(q_pred)
    q2 = q_gt / torch.norm(q_gt)

    R_pred = quat_to_mat(q1)
    R_gt = quat_to_mat(q2)

    delta_R = torch.matmul(R_gt.to(R_pred.device, dtype=R_pred.dtype).t(), R_pred)
    tr = torch.trace(delta_R)

    cos_theta = (tr - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_theta)


def procrustes_alignment(source, target):
    """Find (s, R, t) such that target ≈ s * R @ source + t (Umeyama).

    source, target: [3, N] tensors.
    Returns (aligned_source, s, R, t).
    """
    mu_s = source.mean(dim=1, keepdim=True)
    mu_t = target.mean(dim=1, keepdim=True)
    s_centered = source - mu_s
    t_centered = target - mu_t

    H = s_centered @ t_centered.T
    U, _, V = torch.linalg.svd(H)
    R = V.T @ U.T

    if torch.det(R) < 0:
        V_copy = V.T.clone()
        V_copy[:, 2] *= -1
        R = V_copy @ U.T

    rotated_s = R @ s_centered
    numerator = torch.sum(t_centered * rotated_s)
    var_s = torch.sum(s_centered ** 2)
    s = numerator / (var_s + 1e-12)

    t = mu_t - s * (R @ mu_s)
    aligned_source = s * (R @ source) + t
    return aligned_source, s, R, t
