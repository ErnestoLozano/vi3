# `_apply_depth_alignment_safe` is adapted from Depth Anything 3
# (https://github.com/ByteDance-Seed/Depth-Anything-3, src/depth_anything_3/model/da3.py,
# `NestedDepthAnything3Net._apply_depth_alignment`), modified to write its outputs out of place:
#   Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#   Licensed under the Apache License, Version 2.0 (third_party/Depth-Anything-3/LICENSE).

"""DA3 (Depth Anything 3) wrapped to the standard {pose_enc, depth, ...} dict.

DA3 is a DINOv2-based multi-view model: it predicts z-depth + camera poses +
intrinsics from a stack of views, with optional pose/intrinsics conditioning. The
metric `DA3NESTED-GIANT-LARGE-1.1` checkpoint outputs are already in real-world
metres and it has pose estimation + pose conditioning (the lighter `Metric-Large`
model has neither, so it cannot be used here). This adapter re-expresses DA3's
output in the standard foundation model dict (`vi3.models.foundation_model`).

A few DA3-specific points:

  1. Input is a SINGLE `[B,N,3,H,W]` tensor (not a list of view-dicts). Conditioning
     is just the presence/absence of the `extrinsics`/`intrinsics` forward kwargs —
     no probability dict, so no save/restore is needed.
  2. TTR must call the INNER net `self.api.model(...)` (`DepthAnything3Net`,
     a plain differentiable `forward`) — NOT `self.api(...)`, which is
     `@torch.inference_mode` and would detach the graph.
  3. Pose mode is FORCED by differentiability. `use_ray_pose=True` derives pose via
     RANSAC + SVD (`get_extrinsic_from_camray`), which is non-differentiable and
     stochastic → unusable for gradient-based TTR. So the adapter defaults to
     `use_ray_pose=False` (pose from the `cam_dec` MLP, backprop-clean);
     `self.use_ray_pose=True` selects DA3's ray-head pose for gradient-free use.

DA3 emits no native world-point map, so `world_points` is derived from depth + K +
pose with `vi3.transforms.derive_world_points_from_depth` (same helper VGGT-Omega
uses). The 3D-Gaussian / sky / ray-export branches are ignored.

Nested checkpoint note: the metric `DA3NESTED-GIANT-LARGE-1.1` inner net is a
`NestedDepthAnything3Net` whose geometry (poses + depth + the DINOv2 blocks that also
receive the pose conditioning) lives under `.da3`, with a metric-scale alignment from
`.da3_metric` applied on top. The stock alignment mutates depth/extrinsics IN-PLACE,
which breaks autograd — so the adapter installs an autograd-safe override
(`_apply_depth_alignment_safe`: out-of-place + detached scale). With that, a SINGLE
metric path serves both frozen eval and TTR, so the trajectory scale the grounder's α
trains against matches the scale it is evaluated at. Surrogates (cam_dec / head /
blocks) are wired against the geometry net `.da3`.
"""

import os
import types

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from vi3.models.foundation_model import FoundationModel, _se3_inverse, outputs_from_poses
from vi3.transforms import derive_world_points_from_depth


def _apply_depth_alignment_safe(self, output, metric_output):
    """Autograd-safe replacement for `NestedDepthAnything3Net._apply_depth_alignment`.

    The stock method mutates the `.da3` depth/extrinsics IN-PLACE
    (`output.depth *= scale_factor`, `output.extrinsics[...,:3,3] *= scale_factor`),
    which breaks autograd's version counter (ExpBackward0 error) so TTR cannot
    backprop. This version does the SAME metric-scale alignment OUT-OF-PLACE and
    DETACHES `scale_factor` — the scale is a per-window scalar correction derived from
    masked least-squares (non-differentiable selection anyway); the gradient must flow
    through `.da3`'s geometry, not through the alignment scalar. Making it safe lets a
    SINGLE nested (metric) path serve both frozen eval AND TTR, so the trajectory scale
    the grounder's α trains against matches the scale it is evaluated at.
    """
    from depth_anything_3.utils.alignment import (
        compute_sky_mask, sample_tensor_for_quantile, compute_alignment_mask,
        least_squares_scale_scalar,
    )
    non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
    assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"
    depth_conf_ns = output.depth_conf[non_sky_mask]
    depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
    median_conf = torch.quantile(depth_conf_sampled, 0.5)
    align_mask = compute_alignment_mask(
        output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf)
    valid_depth = output.depth[align_mask]
    valid_metric_depth = metric_output.depth[align_mask]
    scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth).detach()

    # OUT-OF-PLACE metric rescale of depth + the extrinsic translation column.
    output.depth = output.depth * scale_factor
    ext = output.extrinsics
    rot = ext[..., :3, :3]
    t = ext[..., :3, 3] * scale_factor                    # [B,N,3]
    top = torch.cat([rot, t.unsqueeze(-1)], dim=-1)       # [B,N,3,4]
    if ext.shape[-2] == 4:
        output.extrinsics = torch.cat([top, ext[..., 3:4, :]], dim=-2)  # [B,N,4,4]
    else:
        output.extrinsics = top
    output.is_metric = 1
    output.scale_factor = float(scale_factor.item())
    return output


def _install_grad_checkpointing(vit):
    """Wrap the ViT's `process_attention` so each transformer block recomputes its
    activations in the backward pass instead of storing them — active when env
    `DA3_GRAD_CKPT` is set and not "0". Mirrors the Pi3X `PI3X_GRAD_CKPT` idiom: checkpoint ONLY when
    grad is enabled AND the block has trainable params (frozen blocks store nothing
    extra). The giant DINOv2 encoder's per-frame activations are the memory that OOMs
    DA3 at n_ctx 8; this trades one extra forward per unfrozen block for a large
    activation saving. `use_reentrant=False` (the modern, autograd-correct variant).
    """
    orig = vit.process_attention

    def process_attention_ckpt(self, x, block, attn_type="global", pos=None,
                               attn_mask=None):
        if (os.environ.get("DA3_GRAD_CKPT", "") not in ("", "0")
                and torch.is_grad_enabled()
                and any(p.requires_grad for p in block.parameters())):
            return _grad_checkpoint(orig, x, block, attn_type, pos, attn_mask,
                                    use_reentrant=False)
        return orig(x, block, attn_type=attn_type, pos=pos, attn_mask=attn_mask)

    vit.process_attention = types.MethodType(process_attention_ckpt, vit)

# DINOv2 / ImageNet normalization (what DA3's encoder expects). Our datasets deliver
# images in [0,1]; renormalize at the adapter boundary.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _DA3Adapter(FoundationModel):
    """Wrap DA3 → the standard foundation model dict (`vi3.models.foundation_model`), exposing the
    head/aggregator surrogate names the TTR grounder unfreezes.

    `api` is a constructed `DepthAnything3` (so `api.model` is the inner
    differentiable `DepthAnything3Net`). Optional pose conditioning via
    `self.cond_poses` [N,4,4] cam→world OpenCV and `self.cond_intrinsics` (feed K as a
    prior). `self.cond_metric` is kept for interface parity but DA3's metric-ness is
    fixed by the checkpoint, not a per-forward flag. `self.use_ray_pose` selects the
    pose head (False, the default: the differentiable cam_dec; True: the ray head,
    gradient-free).
    """

    def __init__(self, api: nn.Module):
        super().__init__()                # cached_K / cached_image_hw
        self.api = api
        net = api.model                    # inner net (DepthAnything3Net or Nested)
        # The NESTED metric model (`NestedDepthAnything3Net`) has no direct
        # cam_dec/head/foundation model — its geometry (poses + depth + the DINOv2 blocks that
        # also receive the pose conditioning) lives under `.da3`; a fixed metric-scale
        # alignment from `.da3_metric` is applied on top. So resolve the "geometry net"
        # for surrogate wiring. The plain models expose these directly.
        self._is_nested = hasattr(net, "da3")
        geo = net.da3 if self._is_nested else net
        self._geo = geo                          # differentiable geometry net (.da3)
        # Make the nested metric-scale alignment autograd-safe (out-of-place +
        # detached scale) so ONE metric path serves both frozen eval and TTR — see
        # `_apply_depth_alignment_safe` — and α trains and is evaluated at the same
        # scale.
        if self._is_nested:
            net._apply_depth_alignment = types.MethodType(
                _apply_depth_alignment_safe, net)

        # Conditioning state (None/default = plain feed-forward).
        self.cond_poses = None        # [N,4,4] cam→world OpenCV, or None
        self.cond_metric = True       # parity only (scale is checkpoint-fixed for DA3)
        self.cond_intrinsics = True   # also feed cached_K as a calibration prior

        # Pose head selector. TTR requires cam_dec (differentiable); True selects
        # DA3's ray-head pose (RANSAC/SVD, not differentiable).
        self.use_ray_pose = False

        # TTR surrogates — same attribute names the grounder walks for VGGT/Omega.
        #   camera_head → DA3's pose decoder MLP (cam_dec; used when use_ray_pose=False)
        #   dense_head  → the DualDPT depth head
        #   aggregator.global_blocks → the DINOv2 transformer blocks (global attention
        #                              is folded in from `alt_start`)
        #   aggregator.frame_blocks  → empty (no separate per-frame-only blocks)
        self.camera_head = nn.ModuleDict({"pose": geo.cam_dec})
        self.dense_head = geo.head
        self.aggregator = nn.Module()
        self.aggregator.global_blocks = geo.backbone.pretrained.blocks
        self.aggregator.frame_blocks = nn.ModuleList([])

        # Gradient checkpointing on the (giant) DINOv2 encoder — needed to fit TTR at
        # n_ctx 8 on a 32 GB GPU. Active when env DA3_GRAD_CKPT is set and not "0"
        # (scripts/run.py sets it to 1).
        _install_grad_checkpointing(geo.backbone.pretrained)

    # ------------------------------------------------------------------
    def _build_conditions(self, images):
        """Build DA3's `extrinsics`/`intrinsics` forward kwargs when conditioning.

        Returns a dict (possibly empty). Poses are made relative to view-0 in the
        world→cam (w2c) convention DA3's `cam_enc` expects:
            w2c_i @ c2w_0.
        """
        device = images.device
        B, N = images.shape[:2]
        conditions = {}
        if self.cond_poses is None:
            return conditions

        # DA3 STRUCTURALLY requires intrinsics to condition extrinsics: its `cam_enc`
        # calls `extri_intri_to_pose_encoding(ext, ixt, ...)`, which unconditionally
        # subscripts `ixt[...,1,1]` (no None path) — so extrinsics-only conditioning
        # crashes. This is unlike Pi3X (whose pose head takes poses + mask, no K) — for
        # DA3 there is NO pose-only path, so we always feed the cached K here regardless
        # of `cond_intrinsics`. K is the known camera calibration (not extra GT), so this
        # is not a leak; `cond_intrinsics` is retained only for interface parity.
        if self.cached_K is None:
            raise RuntimeError(
                "_DA3Adapter: pose conditioning needs cached_K (DA3's cam_enc encodes "
                "pose+intrinsics jointly; extrinsics-only is unsupported). Plumb K via "
                "set_forward_intrinsics."
            )
        K = self.cached_K.to(device=device, dtype=images.dtype)
        if K.dim() == 3:                           # [N,3,3] → [B,N,3,3]
            K = K.unsqueeze(0).expand(B, -1, -1, -1)
        conditions["intrinsics"] = K

        # Extrinsics: per-view c2w → w2c, made relative to view-0.
        c2w = self.cond_poses.to(device=device, dtype=images.dtype)   # [N,4,4] cam→world
        w2c = _se3_inverse(c2w)                                        # [N,4,4]
        c2w_0 = c2w[0]                                                 # [4,4]
        rel_w2c = torch.einsum("nij,jk->nik", w2c, c2w_0)             # [N,4,4]
        conditions["extrinsics"] = rel_w2c.unsqueeze(0).expand(B, -1, -1, -1)  # [B,N,4,4]
        return conditions

    def _run_net(self, images):
        """Run DA3's inner net and return its raw output dict.

        A SINGLE path serves both frozen eval and TTR: the full inner net (the nested
        metric model when applicable), producing metric depth + extrinsics. This is
        possible because `_apply_depth_alignment` is patched to be autograd-safe
        (out-of-place + detached scale) in __init__, so the metric scale the
        grounder's α trains against equals the scale it is evaluated at.

        The inner net takes the image as the first POSITIONAL arg `x` (the outer
        `DepthAnything3.forward` uses `image=` but is `@inference_mode`).
        """
        conditions = self._build_conditions(images)
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=(images.device.type == "cuda")):
            out = self.api.model(
                images,
                export_feat_layers=[],
                use_ray_pose=self.use_ray_pose,
                ref_view_strategy="first",         # fixed frame-0 anchor
                **conditions,
            )
        return out

    def forward(self, images: torch.Tensor):
        # images: [B, N, 3, H, W] in [0,1]
        if self.cached_K is None or self.cached_image_hw is None:
            raise RuntimeError(
                "_DA3Adapter.forward called without cached_K / cached_image_hw. "
                "Plumb them in via set_forward_intrinsics."
            )
        device = images.device

        # DINOv2-normalize → [B,N,3,H,W].
        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
        img_norm = (images - mean) / std

        out = self._run_net(img_norm)

        # High precision for the geometry transforms (out of autocast).
        with torch.autocast(device_type=device.type, enabled=False):
            # DA3 stores extrinsics as world→cam (OpenCV). Cast to fp32 first.
            T_wc_raw = out["extrinsics"].float()                     # [B,N,4,4] (or 3,4)
            if T_wc_raw.shape[-2] == 3:                              # [B,N,3,4] → [B,N,4,4]
                pad = torch.zeros_like(T_wc_raw[..., :1, :])
                pad[..., 0, 3] = 1.0
                T_wc_raw = torch.cat([T_wc_raw, pad], dim=-2)
            Kpred = out["intrinsics"].float()                        # [B,N,3,3]
            depth = out["depth"].float()                             # [B,N,H,W] z-depth
            conf = out["depth_conf"].float()                         # [B,N,H,W]
            if depth.dim() == 5 and depth.shape[-1] == 1:            # [B,N,H,W,1]→[B,N,H,W]
                depth = depth[..., 0]
            if conf.dim() == 5 and conf.shape[-1] == 1:
                conf = conf[..., 0]

            # DA3 predicts world→cam; FoV from its OWN predicted intrinsics.
            out = outputs_from_poses(_se3_inverse(T_wc_raw), depth, Kpred,
                                     self.cached_image_hw,
                                     depth_conf=conf.unsqueeze(-1))
            # DA3 has no native world-point map — derive from depth + predicted K +
            # pose_enc (same helper VGGT-Omega uses; self-consistent by construction).
            out["world_points"] = derive_world_points_from_depth(
                out["depth"], out["pose_enc"], Kpred, self.cached_image_hw)
        return out
