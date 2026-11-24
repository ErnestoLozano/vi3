"""Forward a (grounded) foundation model on one window and score it against GT."""

import torch

from vi3.data.preprocessing import poses_relative_to_frame0, prepare_batch
from vi3.eval.metrics import (aggregate_sequence_metrics, calculate_metrics,
                              window_alignment_metrics)
from vi3.transforms import pose_enc_to_target_pose


class Evaluator:
    """Forward a model on a list of frames and compute pose/depth metrics.

    Decoupled from the grounder: any callable that takes images and returns the
    VGGT-style {"pose_enc", "depth"} dict can be evaluated. Targets are pulled
    from the dataset's GT pose / depth fields (frame-0-relative).
    """

    def __init__(self, dataset, device="cuda"):
        self.dataset = dataset
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, frame_indices, return_outputs=False):
        """Compute metrics on `frame_indices`, aggregated over the whole window
        (`aggregate_sequence_metrics`), plus the window's Sim(3) ATE and RPE.

        return_outputs:
            When True, returns `(metrics, model_outputs, per_frame_metrics)`.
            `model_outputs` is the raw VGGT forward dict (pose_enc, depth,
            depth_conf, world_points, ...); `per_frame_metrics` is the list
            of per-frame metric dicts (useful for signed-axis error logging).
        """
        prep = prepare_batch(self.dataset, frame_indices, self.device)
        images = prep["images"]
        samples = prep["samples"]

        gt_poses = [s["gt_pose"] for s in samples]
        # gt_depth is optional — datasets without per-pixel depth GT (e.g.
        # EuRoC) simply omit the key, and the depth metric becomes NaN.
        gt_depths = [s.get("gt_depth") for s in samples]
        p_gt, q_gt = poses_relative_to_frame0(gt_poses, device=self.device)

        was_training = model.training
        model.eval()
        # Pi3 adapter needs per-frame K + image_hw to synthesize FoV
        # channels. VGGT / Omega's set_forward_intrinsics is a no-op.
        from vi3.models.pi3_adapter import set_forward_intrinsics
        set_forward_intrinsics(
            model,
            torch.stack([s["K"].to(self.device) for s in samples]),
            (images.shape[-2], images.shape[-1]),
        )
        out = model(images.unsqueeze(0))
        model.train(was_training)

        per_frame = []
        for i in range(len(frame_indices)):
            gt_pose_vec = torch.cat([p_gt[i], q_gt[i]])
            per_frame.append(calculate_metrics(
                out["pose_enc"][0, i], out["depth"][0, i], gt_pose_vec, gt_depths[i],
            ))

        metrics = aggregate_sequence_metrics(per_frame)
        # Scale-invariant window metrics: Sim(3)-aligned ATE and RPE.
        pred = torch.stack([pose_enc_to_target_pose(out["pose_enc"][0, i])
                            for i in range(len(frame_indices))]).cpu().numpy()
        gt = torch.stack([torch.cat([p_gt[i], q_gt[i]])
                          for i in range(len(frame_indices))]).cpu().numpy()
        metrics.update(window_alignment_metrics(pred, gt))
        if return_outputs:
            return metrics, out, per_frame
        return metrics
