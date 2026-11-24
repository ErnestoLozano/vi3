"""Rerun 3D visualization: `RerunSession` (one window's scene: GT, the frozen
model and each grounding method)."""

__all__ = ["RerunSession"]


def __getattr__(name):
    # Lazy: keep the rerun-sdk import optional. `from vi3.viz import RerunSession`
    # only pulls it in when actually requested.
    if name == "RerunSession":
        from vi3.viz.rerun_view import RerunSession
        return RerunSession
    raise AttributeError(f"module 'vi3.viz' has no attribute {name!r}")
