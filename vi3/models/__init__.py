"""Model factory: build feedforward 3R model + the preprocessing config it needs.

Supported backends:

* ``"vggt"``       — Meta's VGGT-1B. 518×518 padded input, patch size 14.
                    Loaded via ``VGGT.from_pretrained("facebook/VGGT-1B")``.
* ``"vggt_omega"`` — Meta's VGGT-Omega. Variable HxW (multiple of 16), patch
                    size 16. Its checkpoint comes from the gated HF repo
                    ``facebook/VGGT-Omega`` (or ``$VGGT_OMEGA_CKPT``).
* ``"pi3"``        — Pi3 (yyfz233/Pi3). Permutation-equivariant, returns
                    cam->world poses + per-pixel local pointmap. Wrapped
                    in ``_Pi3VGGTAdapter`` (frame-0 rebase + I/O remap)
                    so it surfaces the same VGGT-style output dict.
* ``"pi3x"``       — Pi3X (yyfz233/Pi3X): Pi3 + a learned metric-scale token.
* ``"da3"``        — Depth Anything 3 (``vi3.paths.DA3_HF_MODEL``), metric.
* ``"any4d"``      — Any4D, metric; built from its vendored Hydra config, with
                    the checkpoint from the HF hub (or ``$ANY4D_CKPT``).

All share the output contract of ``vi3.models.foundation_model`` (`pose_enc` =
`[t(3), q(4), fov_h, fov_w]` in OpenCV world->cam axes, frame 0 = world), so the
rest of the pipeline is model-agnostic. Each foundation model is one
``register_model(name, spec, build)`` call at the bottom of this file; a new
one plugs in the same way (see ``vi3/models/README.md``), including from outside
the repo: ``build_model("my_pkg.my_module:name")`` imports ``my_pkg.my_module``,
which registers ``name``.

Example:
    >>> from vi3.models import build_model
    >>> model, spec = build_model("vggt_omega")
    >>> spec.kind, spec.target_size, spec.patch_size
    ('vggt_omega', 512, 16)
"""

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """Backend-specific preprocessing + dispatch metadata.

    Datasets read this to route image loading through the correct helper
    and to recompute K with the right patch divisor / target size.
    """
    kind: str                                       # registered foundation model name
    target_size: int                                # 518 (VGGT) | 512 (Omega)
    patch_size: int                                 # 14 (VGGT) | 16 (Omega)
    fixed_output_hw: bool                           # VGGT always 518x518; Omega variable
    preproc_mode: str                               # default mode passed to the loader
    load_fn: Callable[[list, str], torch.Tensor]    # load_and_preprocess_images


def square_spec(kind: str, target_size: int = 518, patch_size: int = 14) -> ModelSpec:
    """The common preprocessing: images padded to a `target_size` square with
    VGGT's loader, in [0, 1] (adapters renormalize internally if needed)."""
    from vggt.utils.load_fn import load_and_preprocess_images
    return ModelSpec(
        kind=kind,
        target_size=target_size,
        patch_size=patch_size,
        fixed_output_hw=True,
        preproc_mode="pad",
        load_fn=load_and_preprocess_images,
    )


def _vggt_omega_spec() -> ModelSpec:
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    return ModelSpec(
        kind="vggt_omega",
        target_size=512,
        patch_size=16,
        fixed_output_hw=False,
        # `max_size` keeps a deterministic, predictable output shape per
        # aspect ratio (longest side → 512, shorter side rounded to a
        # multiple of 16), unlike `balanced`, which targets a token count.
        preproc_mode="max_size",
        load_fn=load_and_preprocess_images,
    )


# name -> (spec factory, model factory). Filled by `register_model`.
_REGISTRY: Dict[str, Tuple[Callable[[], ModelSpec], Callable[[], nn.Module]]] = {}


def register_model(name: str, spec: Callable[[], ModelSpec],
                      build: Callable[[], nn.Module]) -> None:
    """Make a foundation model available as `--model name`.

    Args:
        name:  the foundation model's name (also `ModelSpec.kind`).
        spec:  zero-arg factory returning its `ModelSpec` (e.g.
               `lambda: square_spec(name)`).
        build: zero-arg factory returning the model (on CPU), wrapped so that
               its forward follows the contract of `vi3.models.foundation_model`.
    """
    _REGISTRY[name] = (spec, build)


def available_models() -> List[str]:
    """Names of the registered foundation models."""
    return sorted(_REGISTRY)


def _lookup(kind: str):
    """Registry entry for `kind`; `pkg.module:name` first imports `pkg.module`
    (which registers `name`)."""
    if kind not in _REGISTRY and ":" in kind:
        module, kind = kind.rsplit(":", 1)
        importlib.import_module(module)
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown model kind {kind!r}. Known: {available_models()}.")
    return _REGISTRY[kind]


def build_model_spec(kind: str) -> ModelSpec:
    """Look up the ModelSpec for a backend without instantiating the model.

    Useful when a dataset is constructed before the (heavy) model is loaded.
    """
    return _lookup(kind)[0]()


def _build_da3_net() -> nn.Module:
    """Construct DA3 via its HF-hub `from_pretrained`. DA3's package lives under the
    clone's `src/` dir, which we add to sys.path (so `import depth_anything_3` works
    without a pip install). Weights download from the hub on first use.
    """
    import sys
    from vi3.paths import DA3_SRC_ROOT, DA3_HF_MODEL

    if DA3_SRC_ROOT not in sys.path:
        sys.path.insert(0, DA3_SRC_ROOT)

    from depth_anything_3.api import DepthAnything3
    print(f"Loading DA3 from HF hub: {DA3_HF_MODEL}")
    model = DepthAnything3.from_pretrained(DA3_HF_MODEL)
    return model


def _checkpoint(local, repo, filename, *, env):
    """Path of a checkpoint file: `local` (from `$env`) when set, else
    `CKPT_DIR/<filename>` when present, else the file downloaded from the Hugging
    Face repo into the HF cache."""
    import os
    from vi3.paths import CKPT_DIR
    if local:
        if not os.path.exists(local):
            raise FileNotFoundError(f"${env} points at {local!r}, which does not exist.")
        return local
    stored = os.path.join(CKPT_DIR, filename)
    if os.path.exists(stored):
        return stored
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, filename)
    except Exception as e:
        raise RuntimeError(
            f"Could not download the checkpoint {filename} from Hugging Face "
            f"({type(e).__name__}).\n"
            f"  Download it from https://huggingface.co/{repo}/blob/main/{filename}\n"
            f"  and save it as {stored} (or point ${env} at it).\n"
            f"  The repo may be gated: request access on its page first; with an "
            f"approved account, `huggingface-cli login` lets VI3 download it "
            f"itself.") from e


def _build_any4d_net() -> nn.Module:
    """Construct + load Any4D. Any4D has no `from_pretrained`; it builds through
    its Hydra config (`third_party/Any4D/configs/train.yaml`, `model=any4d`,
    `model/task=images_only`) and loads `ckpt["model"]` (strict=False), exactly
    as its demo does. The vendored repo is put on sys.path (no pip install); the
    checkpoint comes from `_checkpoint`.
    """
    import os
    import sys
    import warnings

    import hydra

    from vi3.paths import ANY4D_CKPT, ANY4D_HF, ANY4D_ROOT

    if ANY4D_ROOT not in sys.path:
        sys.path.insert(0, ANY4D_ROOT)
    from any4d.models import init_model

    # Hydra needs a config path RELATIVE to the caller module's file. Clear any
    # prior global Hydra state first.
    config_dir = os.path.join(ANY4D_ROOT, "configs")
    rel = os.path.relpath(config_dir, os.path.dirname(os.path.abspath(__file__)))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=rel)
    cfg = hydra.compose(
        config_name="train",
        overrides=[
            "machine=local",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ],
    )
    net = init_model(cfg.model.model_str, cfg.model.model_config)

    ckpt = torch.load(_checkpoint(ANY4D_CKPT, *ANY4D_HF, env="ANY4D_CKPT"),
                      map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        warnings.warn(
            f"Any4D state_dict load: {len(missing)} missing keys "
            f"(first few: {list(missing)[:3]}).", stacklevel=2)
    # `unexpected` = the unused scene-flow event encoder (`event_encoder.*`).
    return net


def _build_vggt() -> nn.Module:
    from vggt.models.vggt import VGGT
    return VGGT.from_pretrained("facebook/VGGT-1B")


def _build_vggt_omega() -> nn.Module:
    import warnings
    from vggt_omega.models.vggt_omega import VGGTOmega

    from vi3.paths import VGGT_OMEGA_CKPT, VGGT_OMEGA_HF
    path = _checkpoint(VGGT_OMEGA_CKPT, *VGGT_OMEGA_HF, env="VGGT_OMEGA_CKPT")
    model = VGGTOmega(patch_size=16, enable_camera=True,
                      enable_depth=True, enable_alignment=False)
    state_dict = torch.load(path, map_location="cpu")
    # Allow either a raw state_dict or a wrapped {"state_dict": ...} form.
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Don't crash, but warn loudly — partial loads are usually a sign
        # of a checkpoint / model-class mismatch.
        warnings.warn(
            f"VGGT-Omega state_dict load: {len(missing)} missing keys, "
            f"{len(unexpected)} unexpected. First few: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}",
            stacklevel=2,
        )
    return model


def _build_pi3() -> nn.Module:
    from pi3.models.pi3 import Pi3
    from vi3.models.pi3_adapter import _Pi3VGGTAdapter
    return _Pi3VGGTAdapter(Pi3.from_pretrained("yyfz233/Pi3"))


def _build_pi3x() -> nn.Module:
    from pi3.models.pi3x import Pi3X
    from vi3.models.pi3x_adapter import _Pi3XVGGTAdapter
    return _Pi3XVGGTAdapter(Pi3X.from_pretrained("yyfz233/Pi3X"))


def _build_da3() -> nn.Module:
    from vi3.models.da3_adapter import _DA3Adapter
    return _DA3Adapter(_build_da3_net())


def _build_any4d() -> nn.Module:
    from vi3.models.any4d_adapter import _Any4DAdapter
    return _Any4DAdapter(_build_any4d_net())


# Built-in foundation models. VGGT-Omega takes any HxW (multiple of 16) and has no
# pointmap head; the others share the 518/pad/patch-14 square input. Pi3/Pi3X
# re-normalize ImageNet-style inside forward, DA3/Any4D in their adapters; DA3
# has no native point map, so its adapter derives world_points from depth.
register_model("vggt",       lambda: square_spec("vggt"),  _build_vggt)
register_model("vggt_omega", _vggt_omega_spec,             _build_vggt_omega)
register_model("pi3",        lambda: square_spec("pi3"),   _build_pi3)
register_model("pi3x",       lambda: square_spec("pi3x"),  _build_pi3x)
register_model("da3",        lambda: square_spec("da3"),   _build_da3)
register_model("any4d",      lambda: square_spec("any4d"), _build_any4d)


def build_model(kind: str, *, device: Optional[str] = None) -> Tuple[nn.Module, ModelSpec]:
    """Construct the model + its preprocessing spec.

    Args:
        kind: a registered foundation model name (`available_models()`), or
              `pkg.module:name` for one registered by an external module.
        device: optional device string; if None, leaves the model on CPU
                (caller decides when to move it).

    Returns:
        (model, spec). Model is in eval() mode.
    """
    spec_fn, build_fn = _lookup(kind)
    model = build_fn()
    model.eval()
    if device is not None:
        model = model.to(device)
    return model, spec_fn()


__all__ = ["ModelSpec", "available_models", "build_model", "build_model_spec",
           "register_model", "square_spec"]
