"""Shared path helpers — keep all repo-relative path logic in one place."""

import os

# Resolved at import time. We are inside the installed `vi3` package, which
# sits next to third_party/ (vendored foundation models) and scripts/ at the repo root.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
# Vendored foundation models + GeoCalib (plain copies of their upstream repos).
THIRD_PARTY = os.path.join(REPO_ROOT, "third_party")

# Default locations relative to the repo root; the dataset roots and checkpoints
# can be overridden with the environment variables below.
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
DATA_ROOT = os.path.join(REPO_ROOT, "data")
TARTANAIR_ROOT = os.environ.get("TARTANAIR_ROOT", os.path.join(DATA_ROOT, "tartanair"))
EUROC_ROOT = os.environ.get("EUROC_ROOT", os.path.join(DATA_ROOT, "euroc-mav"))
UZHFPV_ROOT = os.environ.get("UZHFPV_ROOT", os.path.join(DATA_ROOT, "uzh-fpv", "extracted"))
# Checkpoints loaded from a file rather than `from_pretrained` (HF repo, file).
# Looked up in order: the environment variable, `CKPT_DIR/<file>`, then a
# download from the Hugging Face hub into its cache. VGGT-Ω's repo is gated, so
# without an approved, logged-in HF account its file goes in `CKPT_DIR`.
CKPT_DIR = os.path.join(REPO_ROOT, "ckpt")
VGGT_OMEGA_CKPT = os.environ.get("VGGT_OMEGA_CKPT")
VGGT_OMEGA_HF = ("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")
ANY4D_CKPT = os.environ.get("ANY4D_CKPT")
ANY4D_HF = ("airlabshare/any4d-checkpoint", "any4d_4v_combined.pth")
# Any4D: the vendored repo, whose `configs/` feed the Hydra model factory.
ANY4D_ROOT = os.environ.get(
    "ANY4D_ROOT",
    os.path.join(THIRD_PARTY, "Any4D"),
)
# DA3 (Depth Anything 3): vendored repo whose package lives under `src/`
# (`third_party/Depth-Anything-3/src/depth_anything_3`). The adapter puts DA3_SRC_ROOT on
# sys.path so `import depth_anything_3` resolves without an install. Weights come
# from the HF hub (DA3_HF_MODEL); no local checkpoint. See `vi3.models.da3_adapter`.
# Only DA3NESTED-GIANT-LARGE-1.1 has metric scale AND pose estimation AND pose
# conditioning (the Metric-Large model has no poses) — so it is the default; still
# env-overridable for a heavier/lighter swap.
DA3_ROOT = os.environ.get(
    "DA3_ROOT",
    os.path.join(THIRD_PARTY, "Depth-Anything-3"),
)
DA3_SRC_ROOT = os.environ.get(
    "DA3_SRC_ROOT",
    os.path.join(DA3_ROOT, "src"),
)
DA3_HF_MODEL = os.environ.get(
    "DA3_HF_MODEL",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
)
# GeoCalib (single-image gravity estimator) is vendored in third_party/; the
# adapter puts this on sys.path so `import geocalib` works without a separate
# install. See `vi3.models.geocalib_adapter`.
GEOCALIB_ROOT = os.environ.get(
    "GEOCALIB_ROOT",
    os.path.join(THIRD_PARTY, "GeoCalib"),
)
