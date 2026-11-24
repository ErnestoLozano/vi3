"""The end-of-run metrics table of `scripts/run.py`.

`run.py` writes one row per (condition, window) to `comparison.csv`;
`format_final_table` turns it into the paper's grounding table: rows = metrics,
columns = base + each method; per-window medians per dataset, then their mean
over datasets.
"""

import pandas as pd


# Rows of the final table, grouped as in the paper's grounding table:
# (group, column key, header, format). The CSV column is `base_<key>` /
# `<method>_<key>`. Absolute = frame-0 anchored translation / rotation error and
# metric depth, no alignment (not an ATE); scale-invariant = Sim(3)-aligned ATE, relative errors between consecutive frames, and per-frame
# scale-aligned depth.
ABSOLUTE, SCALE_INV = "Absolute (no alignment)", "Scale-invariant (aligned / relative)"
TABLE_METRICS = [
    (ABSOLUTE,  "trans_e",          "Trans. (m, ↓)",             "{:.3f}"),
    (ABSOLUTE,  "rot_e",            "Rot. (°, ↓)",               "{:.2f}"),
    (ABSOLUTE,  "depth_absrel_abs", "Depth AbsRel (↓)",          "{:.3f}"),
    (ABSOLUTE,  "depth_d1_abs",     "Depth δ<1.25 (↑)",          "{:.3f}"),
    (ABSOLUTE,  "scale",            "Pose scale (1 = metric)",   "{:.2f}"),
    (SCALE_INV, "sim3_trans_e",     "ATE, Sim(3) (m, ↓)",        "{:.3f}"),
    (SCALE_INV, "rpe_rot_e",        "RPE rotation (°, ↓)",       "{:.2f}"),
    (SCALE_INV, "rpe_dir_e",        "RPE trans. direction (°, ↓)", "{:.1f}"),
    (SCALE_INV, "depth_absrel",     "Depth AbsRel (↓)",          "{:.3f}"),
    (SCALE_INV, "depth_d1",         "Depth δ<1.25 (↑)",          "{:.3f}"),
]


def format_final_table(df, methods, metrics=None):
    """The paper's grounding table: rows = metrics (grouped absolute /
    scale-invariant), columns = base + each method.

    One block per dataset (each cell = median over that dataset's windows) and,
    with more than one dataset, a final block with the mean of those medians
    (the paper's aggregation). Metrics default to every row of `TABLE_METRICS`,
    skipping any that is all-NaN in a block. Returns the table text."""
    want = metrics or [m[1] for m in TABLE_METRICS]
    cols = ["base"] + list(methods)

    def med(sub, col):
        s = sub[col].dropna() if col in sub.columns else ()
        return float(s.median()) if len(s) else float("nan")

    def block_values(sub):
        return {key: {c: med(sub, f"{c}_{key}") for c in cols}
                for _, key, _, _ in TABLE_METRICS if key in want}

    order = ["tartanair", "euroc", "uzhfpv"]          # the paper's column order
    datasets = (sorted(df["dataset"].unique(),
                       key=lambda d: (order.index(d) if d in order else 99, d))
                if "dataset" in df.columns else [])
    blocks = ([(d, block_values(df[df["dataset"] == d])) for d in datasets]
              if datasets else [("all", block_values(df))])
    if len(blocks) > 1:
        mean = {key: {c: pd.Series([b[key][c] for _, b in blocks]).mean()
                      for c in cols} for key in blocks[0][1]}
        blocks.append((f"mean over {len(datasets)} datasets", mean))

    lines = []
    for label, vals in blocks:
        rows, group = [], None          # (text, cells) with group headings
        for g, key, header, fmt in TABLE_METRICS:
            if key not in vals or all(pd.isna(v) for v in vals[key].values()):
                continue
            if g != group:
                rows.append((g, None))
                group = g
            rows.append(("  " + header, ["-" if pd.isna(vals[key][c])
                                         else fmt.format(vals[key][c]) for c in cols]))
        if not rows:
            continue
        w = max(len(t) for t, _ in rows)
        cw = [max(len(c), *(len(r[i]) for _, r in rows if r)) for i, c in enumerate(cols)]
        lines.append("")
        if len(blocks) > 1:
            lines.append(f"[{label}]")
        lines += [f"{'':<{w}}  " + "  ".join(f"{c:>{cw[i]}}" for i, c in enumerate(cols)),
                  "-" * (w + 2 + sum(x + 2 for x in cw))]
        for text, cells in rows:
            lines.append(text if cells is None else
                         f"{text:<{w}}  " + "  ".join(f"{cells[i]:>{cw[i]}}"
                                                      for i in range(len(cols))))
    return "\n".join(lines)
