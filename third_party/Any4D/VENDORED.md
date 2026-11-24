# Vendored: Any4D

Metric multi-view feed-forward model, used as a scale-aware backbone (`--model any4d`;
pose conditioning is supported by its adapter).
See `vi3/models/any4d_adapter.py`.

- **Upstream:** https://github.com/Any-4D/Any4D.git
- **Commit:** aa9f1b0d7ecdf44a6ef8cef93387be0093bdd497
- **License:** Apache-2.0 (see `LICENSE.txt`)

Vendored as plain files (its `.git` was removed) so `import any4d` works with this dir
on `sys.path` — no install. `vi3.models.build_model` puts `ANY4D_ROOT` (this dir) on
`sys.path` and builds the model from `configs/` via Hydra. The checkpoint
(`any4d_4v_combined.pth`, ~7.7 GB) downloads from the HF hub
(`airlabshare/any4d-checkpoint`) to `ckpt/` on first use; `ANY4D_CKPT` overrides the
path. The DINOv2 encoder skeleton is fetched once via `torch.hub` (its weights are
then overwritten by the checkpoint).

**Trimmed to the inference surface**: only the `any4d/` package (without
`any4d/external/MoGe`, imported lazily and unused here) and `configs/` are kept. The
upstream `assets/`, `scripts/` and packaging files were removed. To update, re-clone
upstream at a newer commit and re-apply this trim.
