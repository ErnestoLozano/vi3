# Vendored: GeoCalib

Single-image camera + gravity estimator. Used here **only** for its gravity
estimate as the GT-free frame-0 gravity source of the `gt-free` init — see
`vi3/models/geocalib_adapter.py` and `vi3/init/gravity.py::estimate_gravity_from_image`.

- **Upstream:** https://github.com/cvg/GeoCalib.git
- **Commit:** 97b8968e7798a66bf04fcf791fb535624241bda7
- **License:** Apache-2.0 (see `LICENSE`)

Vendored as plain files (its `.git` was removed) so `import geocalib` works with the
repo on `sys.path` — no submodule init. The adapter puts `GEOCALIB_ROOT` (this dir)
on `sys.path`. Model weights (`geocalib-pinhole.tar`, ~111 MB) download from the
upstream GitHub releases on first use and cache under the torch hub dir.

**Trimmed to the inference surface**: only the `geocalib/` package is kept. The
upstream `siclib/` (training/eval), `demo.ipynb`, `assets/`, and `gradio_app.py`
were removed (none imported by `geocalib/`). To update, re-clone upstream at a newer
commit and re-apply this trim.
