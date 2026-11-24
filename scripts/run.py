"""
Ground a 3D foundation model with the IMU and evaluate it, window by window.

What to run on: `--suite NAME` (tartanair_full / euroc_full / uzhfpv_full / all)
or `--conditions LABEL[,LABEL...]` (any dataset: House/easy/P000, MH_01_easy,
indoor_forward_3). Each sequence gets `--n_windows` windows (default 3, the
paper's); `--window K` keeps only the K-th. With neither, it runs the demo cell
(TartanAir House/easy/P000, window 1).

Each window logs the frozen model (`base`) and the selected methods (`--methods`,
default `cfa_gyro`): `cfa_gyro` / `cfa` — closed-form IMU scale (+ gyro
rotations), `ttr` — IMU-supervised test-time refinement, `cond` — the IMU
trajectory as the model's pose input. All share the IMU init (`--init gt-free`
by default). The defaults are the paper configuration; `--set KEY=VALUE`
overrides any method or init option.

The run prints the paper's metrics table (saved to summary.txt) and writes one
CSV row per window (comparison.csv). `--viz` also shows GT, the frozen model and
each grounded method in Rerun (`--viz scene.rrd` saves the recording instead).

Examples:
    python scripts/run.py                                    # the demo cell
    python scripts/run.py --viz                              # ... and its 3D view
    python scripts/run.py --suite euroc_full --model vggt_omega
    python scripts/run.py --suite euroc_full --model vggt_omega --methods ttr
"""

import argparse
import os
import time
import traceback

# Before torch is imported: expandable segments keep TTR's memory from fragmenting on
# the larger models (DA3, Pi3X) on a 32 GB GPU, and gradient checkpointing lets their
# TTR fit at 8 frames (set PI3X_GRAD_CKPT=0 / DA3_GRAD_CKPT=0 for faster TTR on a
# larger GPU).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PI3X_GRAD_CKPT", "1")
os.environ.setdefault("DA3_GRAD_CKPT", "1")

import pandas as pd
import torch

from vi3.benchmarks import DEMO_CONDITION, DEMO_WINDOW, SUITES, resolve_conditions
from vi3.data.preprocessing import prepare_batch
from vi3.eval import Evaluator
from vi3.eval.run_cell import (DEFAULT_METHODS, METHOD_SPECS, free_cell, run_cell,
                               select_methods)
from vi3.eval.run_config import (build_config, compute_window_starts,
                                 parse_overrides, seed_everything)
from vi3.eval.summary import TABLE_METRICS, format_final_table
from vi3.grounding_methods.conditioning import pose_input_adapter
from vi3.init.presets import PRESETS
from vi3.models import available_models, build_model
from vi3.paths import REPO_ROOT, RESULTS_ROOT
from vi3.transforms import derive_world_points_from_depth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIZ_VIEWER = "viewer"        # --viz with no path: open the Rerun viewer
VIZ_POINT_SUBSAMPLE = 4      # stride of the logged point clouds

# CSV metric columns `<label>_<column>`: the table's metrics. Two are named
# differently in the Evaluator's metrics dict.
METRIC_KEYS = {"trans_e": "p_total", "rot_e": "q_total"}
METRIC_COLUMNS = [key for _, key, _, _ in TABLE_METRICS]


def metric_cols(label, metrics):
    """`{label}_{column}` CSV columns for one method's metrics dict."""
    return {f"{label}_{c}": metrics.get(METRIC_KEYS.get(c, c), float("nan"))
            for c in METRIC_COLUMNS}


# ---------------------------------------------------------------------------
# Rerun output (--viz). run_cell imports no viz code; these consume its outputs.
# ---------------------------------------------------------------------------

def _log_method_to_rerun(session, label, model_out, per_frame, samples, image_hw):
    """Log one method's frusta, point cloud, depth and per-frame errors."""
    intr = torch.stack([s["K"] for s in samples]).unsqueeze(0)
    world_pts = derive_world_points_from_depth(
        model_out["depth"], model_out["pose_enc"], intr, image_hw)[0]
    session.log_method_3d(label, model_out["pose_enc"][0], world_pts, samples, image_hw,
                          point_subsample=VIZ_POINT_SUBSAMPLE)
    session.log_method_2d(label, model_out["depth"][0, ..., 0],
                          model_out["depth_conf"][0], image_hw)
    session.log_per_frame_errors(label, per_frame)


def _viz_window(session, dataset, frame_indices, methods, model_base, evaluator):
    """Per-window scene setup: layout, GT and the frozen model. Returns the
    samples and image size used to log the methods."""
    session.setup_blueprint(n_cams=len(frame_indices), methods=["baseline"] + methods)
    prep = prepare_batch(dataset, frame_indices, DEVICE)
    samples, image_hw = prep["samples"], tuple(prep["images"].shape[-2:])
    session.log_gt_static(samples, image_hw, log_depth_pointcloud=True)
    _, base_out, base_pf = evaluator.evaluate(model_base.to(DEVICE), frame_indices,
                                              return_outputs=True)
    _log_method_to_rerun(session, "baseline", base_out, base_pf, samples, image_hw)
    return samples, image_hw


def _viz_method(session, res, samples, image_hw):
    """Log one grounded method's scene and the IMU target trajectory."""
    _log_method_to_rerun(session, res["label"], res["model_out"], res["per_frame"],
                         samples, image_hw)
    grounder = res["grounder"]
    t = grounder.cached_targets
    session.log_imu_trajectory(t["p"], t["q"], samples, image_hw)
    session.log_imu_dense_path(samples, grounder)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # What to run on.
    p.add_argument("--suite", default=None, choices=sorted(SUITES),
                   help="A whole dataset (or all of them).")
    p.add_argument("--conditions", default=None,
                   help="Comma-separated sequence labels, e.g. 'MH_01_easy,V1_01_easy' "
                        "(any dataset). With neither --suite nor --conditions, runs "
                        f"the demo cell {DEMO_CONDITION}, window {DEMO_WINDOW}.")
    p.add_argument("--n_windows", type=int, default=3,
                   help="Windows per sequence, spread evenly over 10-90%% of it "
                        "(where the camera is in motion). Default 3: the paper's "
                        "evaluation windows. Each window is grounded "
                        "independently and gets its own CSV row.")
    p.add_argument("--window", type=int, default=None,
                   help="Run only the k-th (0-based) of the --n_windows windows. "
                        "Default: all of them.")
    p.add_argument("--n_ctx", type=int, default=8,
                   help="Frames per window. Default: 8.")
    p.add_argument("--frame_stride", type=int, default=2,
                   help="Keep every k-th camera frame, for more motion per window "
                        "(the IMU is integrated across the skipped frames). Default: 2.")
    # The model and the methods.
    p.add_argument("--model", default="vggt",
                   help=f"Foundation model: one of {available_models()}, or "
                        "`pkg.module:name` for one registered by an external "
                        "module (see vi3/models/README.md). Default: vggt.")
    p.add_argument("--methods", default=None,
                   help=f"Comma-separated grounding methods from "
                        f"{','.join(m[0] for m in METHOD_SPECS)}, or 'all' "
                        f"(default: {','.join(DEFAULT_METHODS)}).")
    p.add_argument("--init", default="gt-free", choices=sorted(PRESETS),
                   help="IMU init: 'gt-free' (default, the paper's) or 'gt' "
                        "(GT gravity, v0 and, where the dataset has them, biases).")
    p.add_argument("--steps", type=int, default=None,
                   help="TTR steps per window (default 160).")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override any method or init option "
                        "(see TTRGrounder.DEFAULTS and vi3/init). Repeatable.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed, re-set before every method. Default: 42. "
                        "Negative = no seeding.")
    p.add_argument("--deterministic", action="store_true",
                   help="Also force deterministic attention / cuDNN (slower; "
                        "does not fit Pi3).")
    # Output.
    p.add_argument("--name", default=None,
                   help="Results go to <results_root>/comparison_<name>/. Default: "
                        "<suite>_<model>, or demo_<model> / <model> without a suite.")
    p.add_argument("--results_root", default=None,
                   help="Results directory (default: results/ in the repo).")
    p.add_argument("--viz", nargs="?", const=VIZ_VIEWER, default=None,
                   metavar="OUT.rrd",
                   help="Show GT (real colours), the frozen model (red) and each "
                        "grounded method (green): frusta, point clouds, depth, "
                        "per-frame errors. Bare --viz opens the Rerun viewer; "
                        "--viz scene.rrd saves the recording instead. Meant for "
                        "one window (the viewer keeps the last one).")
    args = p.parse_args()

    if args.suite is not None and args.conditions is not None:
        p.error("Pass at most one of --suite or --conditions.")
    demo = args.suite is None and args.conditions is None
    if demo:
        args.conditions = DEMO_CONDITION
        if args.window is None:
            args.window = DEMO_WINDOW
    if args.name is None:
        args.name = (f"{args.suite}_{args.model}" if args.suite
                     else f"demo_{args.model}" if demo else args.model)
    try:
        args.overrides = parse_overrides(args.set)
    except ValueError as e:
        p.error(str(e))
    if args.steps is not None:
        args.overrides["steps"] = args.steps
    return args


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed = args.seed if args.seed >= 0 else None
    if seed is not None:
        seed_everything(seed, deterministic=args.deterministic)

    root = args.results_root or RESULTS_ROOT
    results_dir = os.path.join(root if os.path.isabs(root) else os.path.join(REPO_ROOT, root),
                               f"comparison_{args.name}")
    csv_path = os.path.join(results_dir, "comparison.csv")
    os.makedirs(results_dir, exist_ok=True)

    try:
        methods = select_methods(args.methods)
    except ValueError as e:
        raise SystemExit(str(e))
    config = build_config(args.init, args.overrides)
    conditions = resolve_conditions(suite=args.suite, conditions=args.conditions)
    print(f"Run: {args.name}  |  model: {args.model}  |  methods: "
          f"{[m[0] for m in methods]}  |  init: {args.init}  |  n_ctx: {args.n_ctx}  "
          f"|  windows: {args.n_windows}" + (f" (#{args.window})" if args.window is not None else ""))
    if args.overrides:
        print(f"Overrides: {args.overrides}")
    print(f"Sequences: {len(conditions)}  |  output: {results_dir}")

    print(f"Loading model {args.model} ...")
    model_base, model_spec = build_model(args.model)
    # `cond` needs a model with a pose input (Pi3X, DA3, Any4D, ...).
    if any(m[0] == "cond" for m in methods) and pose_input_adapter(model_base) is None:
        if args.methods != "all":
            raise SystemExit(f"--methods cond needs a model that takes poses as input; "
                             f"{args.model} does not.")
        methods = [m for m in methods if m[0] != "cond"]
        print(f"  note: {args.model} takes no pose input; skipping cond.")

    session = None
    if args.viz is not None:
        from vi3.viz import RerunSession
        session = RerunSession(app_name=f"vi3_{args.name}", spawn=args.viz == VIZ_VIEWER)
        if len(conditions) > 1 or (args.n_windows > 1 and args.window is None):
            print("  [viz] note: several windows — the view keeps the LAST one; "
                  "pick one with --conditions <one> --window <k>.")

    rows = []
    t_global = time.time()
    for counter, cond in enumerate(conditions, start=1):
        print(f"\n{'=' * 78}\n[{counter:02d}/{len(conditions)}] {cond.dataset}: "
              f"{cond.label}\n{'=' * 78}")
        try:
            dataset = cond.build_dataset(model_spec=model_spec,
                                         frame_stride=args.frame_stride)
        except FileNotFoundError as e:
            print(f"  SKIP — dataset not found: {e}")
            continue
        evaluator = Evaluator(dataset, device=DEVICE)

        starts = compute_window_starts(len(dataset), args.n_ctx, args.n_windows, args.n_ctx)
        print(f"  windows @ starts {starts} (seq_len={len(dataset)})")
        keep = None if args.window is None else min(args.window, len(starts) - 1)
        for win_idx, start in enumerate(starts):
            if keep is not None and win_idx != keep:
                continue
            frame_indices = list(range(start, start + args.n_ctx))
            try:
                base_metrics = evaluator.evaluate(model_base.to(DEVICE), frame_indices)
            except Exception as e:
                print(f"  [win {win_idx} @ {start}] baseline ERROR — {e}")
                traceback.print_exc()
                continue
            row = {"dataset": cond.dataset, "condition_label": cond.label,
                   "window_idx": win_idx, "window_start": start,
                   **metric_cols("base", base_metrics)}
            print(f"  [win {win_idx} @ {start}] baseline: trans={row['base_trans_e']:.3f} m  "
                  f"rot={row['base_rot_e']:.3f}°  scale={row['base_scale']:.3f}")

            if session is not None:
                samples, image_hw = _viz_window(session, dataset, frame_indices,
                                                [m[0] for m in methods], model_base, evaluator)
            results = run_cell(dataset, model_base, methods, config, evaluator,
                               frame_indices, device=DEVICE,
                               want_outputs=session is not None,
                               seed=seed, seed_deterministic=args.deterministic)
            for res in results:
                label, metrics, secs = res["label"], res["metrics"], res["secs"]
                row.update(metric_cols(label, metrics))
                row[f"{label}_secs"] = secs
                # The IMU init's estimates vs GT (`IMUInit.diagnostics`).
                for k, v in res["history"].items():
                    if isinstance(v, (int, float)):
                        row[f"{label}_{k}"] = v
                print(f"  {label:<13}: trans={metrics['p_total']:.3f} m  "
                      f"rot={metrics['q_total']:.3f}°  scale={metrics['scale']:.3f}  "
                      f"({secs:.0f}s)")
                if session is not None:
                    _viz_method(session, res, samples, image_hw)
                free_cell(res)
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n\nTotal wallclock: {(time.time() - t_global) / 60:.1f} min")
    print(f"CSV written to: {csv_path}")
    if session is not None and args.viz != VIZ_VIEWER:
        session._rr.save(args.viz)
        print(f"Rerun recording saved to {args.viz}  (open with: rerun {args.viz})")

    if rows:
        title = " METRICS — median over windows per dataset"
        table = format_final_table(pd.DataFrame(rows), [m[0] for m in methods])
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{table}")
        out_txt = os.path.join(results_dir, "summary.txt")
        with open(out_txt, "w") as fh:
            fh.write(f"{title}\n{table}\n")
        print(f"\nTable written to: {out_txt}")


if __name__ == "__main__":
    main()
