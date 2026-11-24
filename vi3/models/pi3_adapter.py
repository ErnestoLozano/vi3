"""Pi3 foundation model adapter — re-expresses Pi3's output in VGGT's interface.

The grounder / evaluator / plotting code call `set_forward_intrinsics`
to plumb per-condition `K` + `image_hw` into the adapter just before
each forward. The helper walks through any wrapping `nn.Module`
(e.g. `_ScaleWrappedModel`) to find the actual adapter — assignment
on the wrapper itself does NOT propagate via `__getattr__`, which
only delegates reads.


Pi3 is permutation-equivariant and returns absolute per-frame
camera-to-world poses with an arbitrary world frame. The rest of TTR
assumes world = camera 0 (frame-0-anchored), uses pose_enc =
world->cam with FoV channels, and reads depth as [B, N, H, W, 1].

The adapter does the minimum I/O translation:
  1. Rebase Pi3's poses so cam 0 sits at the world origin
     (T_i^rebased = T_0^-1 @ T_i). This is a bijective SE(3)
     re-expression — no information lost — and the only thing that
     gives the rest of TTR a privileged frame-0 to anchor against.
  2. Invert cam->world to world->cam OpenCV extrinsic and emit a
     VGGT-style 9-vec pose_enc = [t(3), q(4), fov_h, fov_w].
  3. Synthesize fov_h/fov_w from per-frame K (plumbed in via
     `cached_K`/`cached_image_hw` by the grounder before each forward).
     Pi3 has no FoV head, so this keeps the pose_enc FoV consistent with
     the dataset K.
  4. Surface depth from local_points[..., 2:3] (Pi3 stores depth as
     the z component of its local pointmap).
  5. Rebase Pi3's world points by the same T_0^-1 so world_points
     lives in cam-0's frame, same as VGGT.

The adapter also exposes `camera_head`, `aggregator`, and
`dense_head` so `Grounder.reset_model` / `TTRGrounder._make_optimizer` can
walk Pi3 with the same code paths it uses for VGGT/Omega.
"""

import torch
import torch.nn as nn

from vi3.models.foundation_model import FoundationModel, outputs_from_poses


def set_forward_intrinsics(model, K, image_hw) -> bool:
    """Set `cached_K` / `cached_image_hw` on the adapter inside `model`.

    Walks the `.model` chain so wrappers (e.g. _ScaleWrappedModel) don't eat the
    assignment, and sets the state on the first module that declares it (any
    `vi3.models.foundation_model.FoundationModel`). Returns False when there is none — a no-op
    for foundation models that need no intrinsics (VGGT / VGGT-Omega).
    """
    m = model
    while True:
        if "cached_K" in vars(m):
            m.cached_K = K
            m.cached_image_hw = image_hw
            return True
        # Common wrapper pattern: wrapped model lives at `.model`.
        inner = getattr(m, "model", None)
        if inner is None or inner is m:
            return False
        m = inner


class _Pi3VGGTAdapter(FoundationModel):
    """Wrap Pi3 so it returns the VGGT-style {pose_enc, depth, ...} dict.

    The grounder must set `self.cached_K: [N, 3, 3]` and
    `self.cached_image_hw: (H, W)` before calling forward. Both are
    held as plain attributes (not buffers) — they're per-condition
    inputs that change every reset_model.
    """

    def __init__(self, pi3: nn.Module):
        super().__init__()
        self.pi3 = pi3

        # Per-condition state. The grounder / evaluator fills these in
        # before each forward — see TTRGrounder and
        # Evaluator.evaluate. Plain attributes (not nn.Parameter or
        # buffer): we don't want autograd or device migration here.
        self.cached_K = None
        self.cached_image_hw = None

        # Surrogates so the grounders can find the head + aggregator
        # by the same attribute names it uses for VGGT/Omega.
        # `pi3.decoder` is the equivariant trunk (analog of VGGT's
        # global_blocks). Pi3 has no per-frame-only blocks; expose an
        # empty list for `frame_blocks` so the last-N unfreeze rule
        # is a no-op there.
        self.camera_head = nn.ModuleDict({
            "decoder": pi3.camera_decoder,
            "head":    pi3.camera_head,
        })
        self.aggregator = nn.Module()
        self.aggregator.global_blocks = pi3.decoder
        self.aggregator.frame_blocks = nn.ModuleList([])
        self.dense_head = nn.ModuleDict({
            "decoder": pi3.point_decoder,
            "head":    pi3.point_head,
        })

    def forward(self, images: torch.Tensor):
        # images: [B, N, 3, H, W]
        if self.cached_K is None or self.cached_image_hw is None:
            raise RuntimeError(
                "_Pi3VGGTAdapter.forward called without cached_K / "
                "cached_image_hw. Plumb them in from the grounder."
            )

        out = self.pi3(images)
        # Depth = the z of Pi3's local pointmap; K (the dataset's) gives the FoV
        # channels, since Pi3 has no intrinsics head.
        return outputs_from_poses(
            out["camera_poses"], out["local_points"][..., 2:3], self.cached_K,
            self.cached_image_hw, world_points=out["points"],
            depth_conf=out["conf"])
