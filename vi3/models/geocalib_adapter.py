"""GeoCalib loader — single entry point for the image-based gravity source.

GeoCalib (cvg/GeoCalib) estimates camera intrinsics + gravity direction from a
single image. We use only its gravity output as a GT-free frame-0 gravity source
for the `gt-free` init (see `vi3.init.gravity.estimate_gravity_from_image`).

The clone lives at the repo root (`GEOCALIB_ROOT`); this module puts it on
`sys.path` so `import geocalib` works without a separate install. Weights
(`geocalib-pinhole.tar`, ~111 MB) download from GitHub releases on first use and
cache under the torch hub dir — so the FIRST call needs network access.
"""

import sys

from vi3.paths import GEOCALIB_ROOT

_MODEL_CACHE = {}


def build_geocalib(device="cuda", weights="pinhole"):
    """Return a GeoCalib model on `device`, cached per (device, weights).

    Downloads the checkpoint on first use (network). Subsequent calls reuse the
    same instance — the model is heavy, so callers should NOT rebuild per window.
    """
    key = (str(device), weights)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if GEOCALIB_ROOT not in sys.path:
        sys.path.insert(0, GEOCALIB_ROOT)
    from geocalib import GeoCalib  # noqa: E402 (path set just above)

    model = GeoCalib(weights=weights).to(device).eval()
    _MODEL_CACHE[key] = model
    return model
