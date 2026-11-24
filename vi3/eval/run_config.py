"""Run configuration helpers of `scripts/run.py`: seeding, the evaluation windows
and the per-run config (IMU-init preset + `--set` overrides).
"""

def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (CPU + CUDA) RNGs for reproducible runs.

    The dominant source of run-to-run variation in TTR is the attention backward
    pass; forcing the deterministic MATH SDPA kernel below removes it for models
    that respect the global SDPA flags.

    Model support:
      * VGGT / VGGT-Ω use plain `F.scaled_dot_product_attention` with no
        hardcoded backend, so the global flags below make them reproducible.
      * Pi3 is NOT made deterministic: it hardcodes `sdpa_kernel(FLASH)` on bf16
        (overriding the globals), and forcing the MATH kernel there blows GPU
        memory (the full attention matrix doesn't fit), so seeded Pi3 runs
        still vary slightly run to run.

    `deterministic=True` additionally enables cuDNN deterministic mode (slower,
    but stable). Call once at startup and before each method (see
    `run_cell(seed=...)`) so a window's result does not depend on what ran
    before it.
    """
    import os
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace config required for deterministic matmuls on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Force the deterministic MATH SDPA kernel for models that respect the
        # global flags (VGGT / VGGT-Ω). Flash / mem-efficient SDPA have
        # non-deterministic backward passes (atomic adds in non-fixed order).
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        # Best-effort: don't hard-fail if some op lacks a deterministic kernel.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch without warn_only
            pass


def compute_window_starts(seq_len, n_ctx, n_windows, eval_span, inset=0.10):
    """Window start indices placed within an inset band of the sequence.

    A window occupies `eval_span` frames (= n_ctx in context-mode eval); the last
    valid start is `last = seq_len - eval_span`.

        n_windows = 1      -> [0]   (single window at the start)
        n_windows = N >= 2 -> N evenly-spaced starts across the inset band
                              [inset*last, (1-inset)*last]

    The inset band avoids the sequence's stationary tail and barely-moving start
    (both give low-parallax, unreliable model poses), keeping all N windows in the
    cruising-motion regime so they stay comparable. The stride widens with length,
    so every sequence gets ~N windows at roughly equal cost; short sequences that
    can't fit N distinct starts de-duplicate down to fewer.
    """
    last = seq_len - eval_span
    if last <= 0 or n_windows <= 1:
        return [0]
    lo, hi = inset * last, (1.0 - inset) * last
    starts = [round(lo + i * (hi - lo) / (n_windows - 1)) for i in range(n_windows)]
    starts = [max(0, min(last, s)) for s in starts]
    seen, out = set(), []
    for s in starts:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def build_config(init="gt-free", overrides=None):
    """The configuration every method of a run receives: the IMU-init preset
    (``vi3.init.presets``) plus explicit overrides. Method options not given here
    take the method's own defaults, which are the paper's (e.g.
    `TTRGrounder.DEFAULTS`)."""
    from vi3.init.presets import resolve_preset
    return {**resolve_preset(init), **(overrides or {})}


def parse_overrides(items):
    """`KEY=VALUE` strings → dict; values parsed as Python literals when they
    are (numbers, booleans, ...), else kept as strings."""
    import ast
    out = {}
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        try:
            out[key.strip()] = ast.literal_eval(val.strip())
        except (ValueError, SyntaxError):
            out[key.strip()] = val.strip()
    return out
