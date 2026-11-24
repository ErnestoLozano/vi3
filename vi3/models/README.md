# Foundation models

Any multi-view 3D foundation model that maps a few frames to camera poses and
depth can be grounded. VI3 needs no change to the model: an **adapter** maps its
predictions to one output dict, and **one call** registers it.

Built in: `vggt`, `vggt_omega`, `pi3`, `pi3x`, `da3`, `any4d`
(`vi3.models.available_models()`).

## Adding a new 3D foundation model

Most models predict camera→world poses and per-frame depth (often intrinsics
too). Then the adapter is just this:

```python
# my_adapter.py
from vi3.models import register_model, square_spec
from vi3.models.foundation_model import FoundationModel, outputs_from_poses


class MyFoundationModel(FoundationModel):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images):                   # [1, N, 3, H, W] in [0, 1]
        pred = self.net(images)                  # your model's own forward
        return outputs_from_poses(
            pred["camera_poses"],                # [1, N, 4, 4] cam→world, OpenCV axes
            pred["depth"],                       # [1, N, H, W] z-depth
            pred.get("intrinsics", self.cached_K),   # its own K, or the dataset's
            self.cached_image_hw,
        )


def build():
    from my_model import MyModel                 # your package
    return MyFoundationModel(MyModel.from_pretrained("me/my-model"))


register_model("my_model", lambda: square_spec("my_model"), build)
```

Then run it either way:
- From outside the repo: `python scripts/run.py --model my_adapter:my_model ...`. This imports `my_adapter`, which registers `my_model`.
- From inside it: put the file in `vi3/models/` and add the `register_model(...)` line next to the built-in ones at the bottom of `vi3/models/__init__.py`. Then use `--model my_model`.

Everything else is shared, whatever the foundation model:
- the GT-free IMU init;
- the closed-form alignment (`cfa_gyro` / `cfa`);
- the datasets and the metrics.

### The contract (`vi3/models/foundation_model.py`)

`forward(images)` takes `[1, N, 3, H, W]` in `[0, 1]` and returns a dict:

| key | shape | meaning |
|---|---|---|
| `pose_enc` | `[1, N, 9]` | world→camera OpenCV extrinsic, with frame 0 as the world: `[t(3), q(4) xyzw, fov_h, fov_w]` |
| `depth` | `[1, N, H, W, 1]` | z-depth |
| `depth_conf` | `[1, N, H, W]` | optional |
| `world_points` | `[1, N, H, W, 3]` | optional, in the frame-0 world (derived from depth + pose when absent) |

`outputs_from_poses` builds this dict from:
- poses in any world frame (it rebases them to frame 0);
- depth;
- K (for the FoV channels);
- optionally, world points.

**Scale.**
- Up-to-scale models (e.g. VGGT) need nothing extra: the IMU supplies the metric scale.
- Metric models (e.g. DA3) are handled the same way. The closed-form scale then simply comes out close to 1.

**Intrinsics.** Before every forward, `cached_K` (`[N, 3, 3]`, at the loaded resolution) and `cached_image_hw` are set on any `FoundationModel`. Use them when the model takes intrinsics as input or has no intrinsics head.

**Preprocessing.** `square_spec(name, target_size=518, patch_size=14)` pads the images to a square with VGGT's loader. For other input sizes, build a `ModelSpec` yourself, as `_vggt_omega_spec` does.

### Pose conditioning (optional)

`--methods cond` feeds the IMU trajectory as the model's pose input. An adapter supports it by holding a `cond_poses` attribute (`None` by default), which the grounder sets before the forward:
- format: `[N, 4, 4]` camera→world, OpenCV axes, frame-0 relative, in metres;
- use it in your forward as the model's pose prior (see `any4d_adapter.py`).

### Test-time refinement (optional)

The default methods (`cfa_gyro`, `cfa`) only call `forward`. TTR (`--methods ttr`) also fine-tunes parts of the model, which it finds by attribute name. Expose them as aliases of the model's own submodules in `__init__`:

```python
self.camera_head = net.pose_head                   # always trained
self.aggregator = nn.Module()
self.aggregator.global_blocks = net.transformer.blocks   # last `unfreeze_n` trained
self.aggregator.frame_blocks = nn.ModuleList([])          # per-frame blocks, if any
self.dense_head = net.depth_head                   # trained by the default loss
```

`forward` must then stay differentiable with respect to those modules:
- no `torch.no_grad()`;
- no in-place writes on graph tensors.

`pi3_adapter.py`, `da3_adapter.py` and `any4d_adapter.py` are worked examples.
