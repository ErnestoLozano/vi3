"""Any4D wrapped to the standard {pose_enc, depth, ...} foundation model dict.

Any4D ("Unified Feed-Forward Metric 4D Reconstruction") is a metric multi-view
model on the UniCeption stack: a DINOv2-large encoder, a multi-view
`info_sharing` transformer + `scale_token`, and `pose_head` / `dense_head` /
`scale_head`. Its `forward(views)` takes a LIST of per-view dicts, and
`postprocess_model_outputs_for_inference` emits `camera_poses` (cam→world
OpenCV), `depth_z`, `pts3d`, `intrinsics` and `conf`. Three details:

  1. Conditioning toggle: Any4D gates its geometric inputs by the mutable
     `net.geometric_input_config` dict of probabilities (`torch.rand < prob`,
     deterministic at 0/1). `_run_net` saves that dict, overrides the
     relevant probs, forwards, and restores.
  2. `postprocess(...)` is called with masks OFF (Any4D defaults them on); we
     want the raw dense outputs.
  3. The scene-flow head (the "4D" part) is ignored; only pose/depth/points are
     read.

Weight loading (Hydra build + checkpoint) lives in `vi3.models.build_model`; this
class just wraps a constructed Any4D network.
"""

import torch
import torch.nn as nn

from vi3.models.foundation_model import FoundationModel, outputs_from_poses

# DINOv2 / ImageNet normalization (what Any4D's encoder expects). Our datasets
# deliver images in [0,1]; the encoder wants dinov2 (ImageNet) mean/std, so we
# renormalize at the adapter boundary.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _Any4DAdapter(FoundationModel):
    """Wrap Any4D → the standard foundation model dict (`vi3.models.foundation_model`), exposing
    the head/aggregator surrogate names the TTR grounder unfreezes.

    Optional pose conditioning via `self.cond_poses` [N,4,4] cam→world OpenCV and
    `self.cond_metric` (bool): True → anchor absolute scale (is_metric_scale);
    False → geometry only. `self.cond_intrinsics` toggles feeding K as a
    calibration prior.
    """

    def __init__(self, net: nn.Module):
        super().__init__()                # cached_K / cached_image_hw
        self.net = net

        # Pose-conditioning state (set by the `cond` grounding method; None = plain
        # feed-forward).
        self.cond_poses = None        # [N,4,4] cam→world OpenCV, or None
        self.cond_metric = True       # is_metric_scale for the conditioning poses
        self.cond_intrinsics = True   # also feed cached_K as a calibration prior

        # TTR surrogates — same attribute names the grounder walks for VGGT/Omega.
        #   camera_head    → Any4D's pose head
        #   dense_head     → its depth/point DPT head (already named dense_head)
        #   aggregator.global_blocks → the multi-view transformer's block list
        #   aggregator.frame_blocks  → empty (no per-frame-only blocks; like Pi3)
        self.camera_head = nn.ModuleDict({"pose": net.pose_head})
        self.dense_head = net.dense_head
        self.aggregator = nn.Module()
        self.aggregator.global_blocks = net.info_sharing.self_attention_blocks
        self.aggregator.frame_blocks = nn.ModuleList([])

        # DINOv2 encoder's declared norm type (must be echoed in each view dict).
        self._data_norm_type = getattr(net.encoder, "data_norm_type", "dinov2")

    # ------------------------------------------------------------------
    def _build_views(self, images):
        """[B,N,3,H,W] (range [0,1]) → list of N per-view dicts for Any4D.

        B is the sample batch (always 1 in our pipeline). Each view carries the
        per-frame image (batch B) plus, when conditioning, its intrinsics / pose.
        """
        device = images.device
        B, N = images.shape[:2]
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

        K = None
        if self.cached_K is not None:
            K = self.cached_K.to(device=device, dtype=images.dtype)
            if K.dim() == 3:                       # [N,3,3] → [B,N,3,3]
                K = K.unsqueeze(0).expand(B, -1, -1, -1)

        # Feed intrinsics as a calibration prior only when pose-conditioning (poses
        # travel with rays/K, as in Pi3X); the plain forward is images-only, like
        # the other foundation models.
        feed_K = (self.cond_poses is not None) and self.cond_intrinsics and (K is not None)

        views = []
        for i in range(N):
            img = (images[:, i] - mean) / std      # [B,3,H,W] dinov2-normalized
            view = {"img": img, "data_norm_type": [self._data_norm_type]}
            if feed_K:
                view["intrinsics"] = K[:, i]                       # [B,3,3]
            if self.cond_poses is not None:
                pose = self.cond_poses[i].to(device=device, dtype=images.dtype)
                view["camera_poses"] = pose.unsqueeze(0).expand(B, 4, 4)   # [B,4,4]
                view["is_metric_scale"] = torch.tensor(
                    [bool(self.cond_metric)] * B, device=device)
            views.append(view)
        return views

    # ------------------------------------------------------------------
    def _geometric_input_overrides(self):
        """The `geometric_input_config` prob-key overrides for the current mode.

        Any4D has no `_configure_geometric_input_config`; conditioning is gated by
        `torch.rand < prob`, which is deterministic at the 0/1 extremes. We mirror
        the shipped task configs:
          * base (no cond)      → images_only: overall_prob=0.
          * pose-cond metric    → posed_sfm: overall/cam on, pose_scale_norm off.
          * pose-cond up-to-scl → posed_sfm_non_metric: pose_scale_norm_all_prob=1.
        Intrinsics travel WITH poses via the ray-dirs input (feed_K in _build_views);
        we leave ray_dirs_prob at 0 here so the model's own ray/intrinsics head runs
        exactly as in base — the K prior reaches it through the view dict, not a
        forced ray-dirs conditioning branch.
        """
        if self.cond_poses is None:
            return {"overall_prob": 0}
        return {
            "overall_prob": 1,
            "dropout_prob": 0,
            "cam_prob": 1,
            "ray_dirs_prob": 0,
            "depth_prob": 0,
            "doppler_prob": 0,
            # metric anchors the prior's absolute scale; up-to-scale skips it.
            "pose_scale_norm_all_prob": 0 if self.cond_metric else 1,
        }

    def _run_net(self, views):
        """Return Any4D's list of per-view output dicts.

        The same differentiable `forward()` + postprocess serves TTR (grad on)
        and the frozen model (grad off); postprocess is a plain function.
        """
        from any4d.utils.inference import (
            preprocess_input_views_for_inference,
            postprocess_model_outputs_for_inference,
        )
        proc = preprocess_input_views_for_inference(views)

        # Toggle conditioning by overriding the mutable geometric_input_config
        # prob dict, restoring it in `finally`.
        gic = self.net.geometric_input_config
        saved = {k: gic[k] for k in self._geometric_input_overrides()}
        gic.update(self._geometric_input_overrides())
        try:
            preds = self.net.forward(proc)
            # The postprocess recovers intrinsics via linear solves with no bf16
            # kernel — run it in fp32 (and outside autocast). Cast the raw head
            # outputs up first; the heavy forward() stays bf16.
            preds = [{k: (v.float() if torch.is_tensor(v) and v.is_floating_point()
                          else v) for k, v in p.items()} for p in preds]
            with torch.autocast(device_type=proc[0]["img"].device.type, enabled=False):
                preds = postprocess_model_outputs_for_inference(
                    raw_outputs=preds, input_views=proc,
                    apply_mask=False, mask_edges=False, apply_confidence_mask=False,
                )
        finally:
            gic.update(saved)
        return preds

    def forward(self, images: torch.Tensor):
        # images: [B, N, 3, H, W] in [0,1]
        if self.cached_K is None or self.cached_image_hw is None:
            raise RuntimeError(
                "_Any4DAdapter.forward called without cached_K / cached_image_hw. "
                "Plumb them in via set_forward_intrinsics."
            )
        views = self._build_views(images)
        preds = self._run_net(views)          # list of N per-view dicts

        # Restack per-view outputs → [B, N, ...].
        T_cw = torch.stack([p["camera_poses"] for p in preds], dim=1)   # [B,N,4,4] cam→world
        depth = torch.stack([p["depth_z"] for p in preds], dim=1)       # [B,N,H,W,1]
        pts = torch.stack([p["pts3d"] for p in preds], dim=1)           # [B,N,H,W,3] world
        conf = torch.stack([p["conf"] for p in preds], dim=1)           # [B,N,H,W]
        Kpred = torch.stack([p["intrinsics"] for p in preds], dim=1)    # [B,N,3,3]

        # FoV from Any4D's OWN predicted intrinsics (it has an intrinsics head).
        return outputs_from_poses(T_cw, depth, Kpred, self.cached_image_hw,
                                  world_points=pts, depth_conf=conf.unsqueeze(-1))
