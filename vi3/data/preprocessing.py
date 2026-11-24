"""Batch preparation and pose transformations.

Decouples the per-sample dataset access from the grounder so each grounder
can pull only the fields it needs without re-implementing the boilerplate.
"""

import torch
from scipy.spatial.transform import Rotation as R

from vi3.transforms import vec_to_se3


def prepare_batch(dataset, indices, device):
    """Fetch a list of samples, stack images, and move tensors to `device`.

    Returns:
        {
          "images":  torch.FloatTensor [N, 3, H, W],
          "samples": list of N dicts (the original per-sample dict, with all
                     tensor fields moved to `device`).
        }

    The image tensor is stacked because VGGT consumes a [N,3,H,W] input;
    everything else stays as a list because per-sample tensor shapes (e.g. the
    IMU window length) can differ across samples.
    """
    batch = [dataset[i] for i in indices]
    images = torch.stack([d["image"] for d in batch]).to(device)

    moved = []
    for d in batch:
        m = {}
        for k, v in d.items():
            m[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        moved.append(m)

    return {"images": images, "samples": moved}


def poses_relative_to_frame0(gt_poses, device=None):
    """Anchor a list of camera->world poses to frame 0.

    gt_poses: list of N tensors of shape [7] = [tx, ty, tz, qx, qy, qz, qw].
    Returns:
        p_rel: list of N tensors of shape [3] (frame-0-local positions)
        q_rel: list of N tensors of shape [4] (frame-0-local quaternions [xyzw])
    """
    if device is None:
        device = gt_poses[0].device

    Ms = [vec_to_se3(p.view(-1)[:7], device=device) for p in gt_poses]
    inv_M0 = torch.inverse(Ms[0])

    p_rel, q_rel = [], []
    for M in Ms:
        M_rel = inv_M0 @ M
        p_rel.append(M_rel[:3, 3].to(device))
        # SO(3) -> xyzw via scipy (non-differentiable; this is target-building, not training)
        q = R.from_matrix(M_rel[:3, :3].detach().cpu().numpy()).as_quat()
        q_rel.append(torch.from_numpy(q).to(device=device, dtype=gt_poses[0].dtype))
    return p_rel, q_rel
