"""The shared base of the grounding methods.

A grounding method turns a frozen 3D foundation model metric with the IMU. Each
window starts from the IMU init block (``vi3.init.IMUInit``: gravity, gyro bias,
metric scale and v0, and the IMU target trajectory they integrate); `Grounder`
adds what the methods share on top of it:

  * a fresh copy of the foundation model per window, wrapped so its translations
    and depth are scaled by one scalar α (the metric scale every method sets);
  * the per-window bookkeeping (init diagnostics and the α trace in `history`).

The methods differ only in how they use the targets — `train()`:
``closed_form.ClosedFormGrounder`` (Umeyama scale, + gyro rotations),
``ttr.TTRGrounder`` (test-time refinement) and
``conditioning.ConditioningGrounder`` (the trajectory as the model's pose input).

Model-agnostic: every foundation model is normalized by its adapter to the same
``{pose_enc, depth, world_points}`` output (``vi3.models.foundation_model``), so
the scale wrapper operates on that dict.
"""

import copy

import torch
import torch.nn as nn

from vi3.init import IMUInit


class _ScaleWrappedModel(nn.Module):
    """Wraps any supported foundation model so its forward output is scaled by a
    learnable α. The wrapped model returns the shared adapter dict
    (``{pose_enc, depth, world_points, ...}``); we scale only the translation
    channels of ``pose_enc`` plus ``depth``/``world_points``, leaving rotation,
    FoV, and confidences untouched — so it is model-agnostic.

    Forwards unknown attribute lookups to the wrapped inner model so callers that
    reach into model-specific attributes (e.g. ``model.camera_head`` in
    ``TTRGrounder._make_optimizer``) keep working transparently.
    """

    def __init__(self, model, alpha):
        super().__init__()
        self.model = model
        # Do NOT register alpha as a child Parameter: the grounder owns it (TTR
        # gives it its own optimizer group). Registered, it would also show up in
        # `model.parameters()`. object.__setattr__ sidesteps nn.Module registration.
        object.__setattr__(self, "_alpha_ref", alpha)

    @property
    def alpha(self):
        return self._alpha_ref

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        a = self._alpha_ref.to(device=out["pose_enc"].device, dtype=out["pose_enc"].dtype)

        # pose_enc layout: [t(3), q(4), fov_h, fov_w]. Scale only the translation
        # channels (0:3). Build the scaled pose_enc out-of-place via cat — an
        # in-place write on a clone can still touch shared autograd state when the
        # inner model returns a view of a buffer reused on the next forward pass.
        t = out["pose_enc"][..., :3] * a            # [..., 3]
        rest = out["pose_enc"][..., 3:]             # [..., q + fov]
        out["pose_enc"] = torch.cat([t, rest], dim=-1)

        if "depth" in out and out["depth"] is not None:
            out["depth"] = out["depth"] * a
        if "world_points" in out and out["world_points"] is not None:
            out["world_points"] = out["world_points"] * a
        return out

    def __getattr__(self, name):
        # nn.Module.__getattr__ runs when normal lookup fails. Try the parent
        # first (self.model, alpha, buffers, submodules); on failure fall through
        # to the inner model so callers see model-specific attributes.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class Grounder:
    """Base of the grounding methods: the window's `IMUInit` (``self.init``) and
    the α-scaled copy of the foundation model. Subclasses implement
    `train(frame_indices, config) -> (grounded_model, history)`.
    """

    def __init__(self, model_base, dataset, device="cuda"):
        self.device = device
        self.model_base = model_base.to(device)
        self.dataset = dataset
        # Aggregator/camera-head dtypes — bf16 on Ampere+, fp16 elsewhere.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16

    def train(self, frame_indices, config):
        """Ground the foundation model on `frame_indices`; return (model, history)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # FoundationModel copy (α-scaled)
    # ------------------------------------------------------------------
    def reset_model(self, unfreeze_n=0, unfreeze_depth_head=False):
        """A fresh copy of the foundation model (see `_copy_model`) wrapped so its
        outputs are α-scaled."""
        inner = self._copy_model(unfreeze_n=unfreeze_n,
                                 unfreeze_depth_head=unfreeze_depth_head)
        return _ScaleWrappedModel(inner, self.alpha)

    def _copy_model(self, unfreeze_n=0, unfreeze_depth_head=False):
        """Deep-copy the base model and unfreeze the camera head, the last
        `unfreeze_n` aggregator blocks and, optionally, the depth head.

        Block naming: VGGT has `aggregator.global_blocks` (inter-frame attention)
        and `aggregator.frame_blocks`; VGGT-Ω has `aggregator.inter_frame_blocks`
        with two attention types (`aggregator.inter_frame_attention_types`:
        "global" and sparse "register" blocks). The last `unfreeze_n` *global*
        inter-frame blocks and the last `unfreeze_n` frame blocks are unfrozen.
        The depth head is `depth_head` (VGGT) or `dense_head` (VGGT-Ω).
        """
        model = copy.deepcopy(self.model_base)
        model.requires_grad_(False)
        # Fresh depth-anchor reference per window (TTR captures it at step 0).
        self._depth_anchor_ref = None
        # VGGT uses this flag; Omega ignores it.
        if hasattr(model, "pred_cameras"):
            model.pred_cameras = True

        if hasattr(model, "camera_head") and model.camera_head is not None:
            model.camera_head.requires_grad_(True)

        if unfreeze_depth_head:
            depth_head = self._get_depth_head(model)
            if depth_head is None:
                raise AttributeError(
                    "`unfreeze_depth_head=True` but the model has no depth "
                    "head (neither `depth_head` for VGGT nor `dense_head` "
                    "for VGGT-Omega found; or it was built with "
                    "enable_depth=False)."
                )
            depth_head.requires_grad_(True)

        if unfreeze_n > 0:
            inter_blocks, frame_blocks = self._aggregator_blocks(model)
            inter_types = self._inter_block_types(model, n_blocks=len(inter_blocks))
            global_idxs = [i for i, t in enumerate(inter_types) if t == "global"]
            for i in global_idxs[-unfreeze_n:]:
                inter_blocks[i].requires_grad_(True)
            for i in range(max(0, len(frame_blocks) - unfreeze_n), len(frame_blocks)):
                frame_blocks[i].requires_grad_(True)

        return model.to(self.device)

    @staticmethod
    def _aggregator_blocks(model):
        """Return (inter_frame_blocks, frame_blocks) regardless of model.

        VGGT calls them `global_blocks`/`frame_blocks`; VGGT-Omega uses
        `inter_frame_blocks`/`frame_blocks`. Both are nn.ModuleList; we
        pick whichever attribute exists.
        """
        agg = model.aggregator
        inter = getattr(agg, "global_blocks", None)
        if inter is None:
            inter = getattr(agg, "inter_frame_blocks", None)
        if inter is None:
            raise AttributeError(
                "Model's aggregator has neither `global_blocks` (VGGT) nor "
                "`inter_frame_blocks` (Omega)."
            )
        return inter, agg.frame_blocks

    @staticmethod
    def _inter_block_types(model, n_blocks):
        """Return a list of attention-type strings, one per inter-frame block.

        For VGGT (no per-block attention-type distinction) we return all
        "global". For Omega we read `aggregator.inter_frame_attention_types`.
        """
        types = getattr(model.aggregator, "inter_frame_attention_types", None)
        if types is None:
            return ["global"] * n_blocks
        if len(types) != n_blocks:
            raise ValueError(
                f"Aggregator block count {n_blocks} does not match "
                f"len(inter_frame_attention_types)={len(types)}."
            )
        return list(types)

    @staticmethod
    def _get_depth_head(model):
        """Return the depth-producing head submodule, or `None`.

        VGGT names it `depth_head` (a `DPTHead`); VGGT-Omega names it
        `dense_head` (a `DenseHead` that produces depth + depth_conf).
        Either may be `None` if the model was constructed with
        `enable_depth=False`. Either may also be absent as an attribute
        on third-party models with different naming.
        """
        for name in ("depth_head", "dense_head"):
            head = getattr(model, name, None)
            if head is not None:
                return head
        return None

    # ------------------------------------------------------------------
    # Per window: IMU init + targets (vi3.init), α, logging
    # ------------------------------------------------------------------
    def _new_window(self, config):
        # The window's IMU init (gravity, bias, S, v0) and its targets.
        self.init = IMUInit(self.model_base, self.dataset, config, self.device)
        # The scale α, set by each method (CFA: Umeyama scale, TTR: initialized
        # from the IMU path length and refined, conditioning: 1).
        self.alpha = nn.Parameter(torch.ones((), device=self.device))
        self._diag_emitted = False

    def _post_step(self, step, components, history):
        history.setdefault("ba", []).append(self.init.ba.detach().cpu().numpy().tolist())
        history.setdefault("bg", []).append(self.init.bg.detach().cpu().numpy().tolist())
        # The init's estimates vs GT, once (constant across steps).
        # Scalars go straight into `history` so run.py writes them to the CSV.
        if not self._diag_emitted:
            self._diag_emitted = True
            history.update(self.init.diagnostics())
        # α trace.
        alpha_val = float(self.alpha.detach())
        history.setdefault("alpha", []).append(alpha_val)
        if step % 20 == 0:
            print(f"             α = {alpha_val:.4f}")

    def _build_targets(self, samples, config):
        """Run the window's IMU init; return its target trajectory."""
        return self.init.build(samples)
