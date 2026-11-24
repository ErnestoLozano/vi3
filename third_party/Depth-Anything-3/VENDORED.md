# Vendored: Depth Anything 3

- **Upstream:** https://github.com/ByteDance-Seed/Depth-Anything-3
- **Commit:** 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
- **License:** Apache-2.0 (see `LICENSE`)

Plain copy of the package under `src/`, unmodified. **Trimmed**: `app/`, `bench/`, `cli.py` and `services/` were
removed (not imported by the inference API). `vi3.models.da3_adapter` puts `src/` on `sys.path`; weights download
from the Hugging Face hub on first use.
