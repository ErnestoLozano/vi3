"""The per-window grounding unit and the method registry.

`run_cell` runs one or more grounding methods on one window (a "cell": one
sequence's frames) and returns per-method metrics. `scripts/run.py` loops
sequences and windows, calls it once per window, and turns the results into CSV
rows, the metrics table and, with --viz, the Rerun scene. No viz imports here.
"""

import gc
import time

import torch

from vi3.grounding_methods import ClosedFormGrounder, ConditioningGrounder, TTRGrounder


# Method registry: (label, grounder_class, extra_config). `extra_cfg` layers on
# top of the run's config, per method.
#
# The frozen-model "baseline" column is logged per-cell by the driver, not a
# grounder here. All methods share the same IMU init (`vi3.init`):
#   cfa_gyro — closed-form IMU scale + gyro rotations (default; fast, general)
#   cfa      — the closed-form IMU scale alone, keeping the model's rotations
#   ttr      — IMU-supervised test-time refinement with a learnable scale α
#              (also refines the model's geometry; best with an accurate IMU
#              trajectory)
#   cond     — the IMU trajectory as the model's pose input (models that take
#              poses: Pi3X, DA3, Any4D)
METHOD_SPECS = [
    ("cfa_gyro", ClosedFormGrounder, {"gyro": True}),
    ("cfa", ClosedFormGrounder, {"gyro": False}),
    ("ttr", TTRGrounder, {}),
    ("cond", ConditioningGrounder, {}),
]
DEFAULT_METHODS = ("cfa_gyro",)


def select_methods(names=None):
    """Filter METHOD_SPECS to `names` (comma-list or iterable), preserving the
    requested order. None → DEFAULT_METHODS. 'all' → every method. 'none' / []
    → empty. Unknown labels raise.
    """
    if names is None:
        names = DEFAULT_METHODS
    if names == "all":
        return list(METHOD_SPECS)
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if not names or names == ["none"]:
        return []
    by_label = {m[0]: m for m in METHOD_SPECS}
    unknown = [n for n in names if n not in by_label]
    if unknown:
        raise ValueError(
            f"Unknown method label(s) {unknown}. "
            f"Available: {[m[0] for m in METHOD_SPECS]}.")
    return [by_label[n] for n in names]


def run_cell(dataset, model_base, method_specs, config, evaluator,
             frame_indices, *, device="cuda", want_outputs=False, seed=None,
             seed_deterministic=True):
    """Ground + evaluate each method on `frame_indices`.

    Args:
        dataset:        RefinementDataset for this cell.
        model_base:     the un-adapted model (built once, reused per cell).
        method_specs:   list of (label, grounder_cls, extra_cfg) — e.g. from
                        `select_methods(...)`.
        config:         the run's config (`vi3.eval.run_config.build_config`).
        evaluator:      an `Evaluator` bound to `dataset`.
        frame_indices:  dataset indices of the window.
        want_outputs:   if True, also return model_outputs + per_frame metrics
                        (for the Rerun adapter). Costs extra eval memory.
        seed:           if not None, re-seed all RNGs to this value before each
                        method's `train`, so the cell is reproducible and method
                        order doesn't matter (each method draws from the same
                        RNG state). None = leave RNG as-is (nondeterministic).

    Returns:
        list of dicts, one per method, each:
            {label, metrics, secs, history,
             model_out? per_frame? grounder?}   (last three only when want_outputs)
    """
    results = []
    for label, grounder_cls, extra_cfg in method_specs:
        cfg = {**config, **extra_cfg}
        # Re-seed before each method so the cell is reproducible and method
        # order doesn't matter (each method draws from the same RNG state).
        if seed is not None:
            from vi3.eval.run_config import seed_everything
            seed_everything(seed, deterministic=seed_deterministic)
        t0 = time.time()
        grounder = grounder_cls(model_base, dataset, device=device)
        adapted, history = grounder.train(frame_indices, cfg)

        if want_outputs:
            metrics, model_out, per_frame = evaluator.evaluate(
                adapted, frame_indices, return_outputs=True)
        else:
            metrics = evaluator.evaluate(adapted, frame_indices)
            model_out = per_frame = None
        secs = time.time() - t0

        res = {"label": label, "metrics": metrics, "secs": secs, "history": history}
        if want_outputs:
            # The viz needs the outputs and the grounder's IMU state (targets,
            # init); its grounded model copy is freed below.
            res.update(model_out=model_out, per_frame=per_frame, grounder=grounder)
        results.append(res)
        del grounder, adapted
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def free_cell(res):
    """Drop the retained outputs / grounder from a `run_cell` result and free
    CUDA memory. Call after the driver is done with them (Rerun logging)."""
    res.pop("grounder", None)
    res.pop("model_out", None)
    res.pop("per_frame", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
