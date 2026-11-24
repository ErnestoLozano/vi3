# Vendored: Pi3 / Pi3X

- **Upstream:** https://github.com/yyfz/Pi3
- **Commit:** a65e87e (main, 2026-11-18)
- **License:** BSD-style (see `LICENSE`)

Plain copy, with one modification, marked `[VI3 modification]` in the file:
`pi3/models/pi3x.py` checkpoints the decoder blocks' activations when `PI3X_GRAD_CKPT` is set (not "0") and the
block is trained (VI3's test-time refinement), so TTR fits in GPU memory. Outputs are unchanged. Installed editable
(`pip install -e third_party/pi3 --no-deps`).
