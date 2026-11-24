# Datasets

The loaders read three datasets: **TartanAir** (synthetic, dense GT depth),
**EuRoC MAV** and **UZH-FPV** (real-sensor). `scripts/download_data.py` fetches
each from its official source straight into the layout below:

```bash
python scripts/download_data.py euroc       # ~11 GB: the 11 sequences (ASL zips, no ROS bags)
python scripts/download_data.py uzhfpv      # the 16 sequences with GT + calibration
python scripts/download_data.py tartanair   # official `tartanair` package (`pip install tartanair -c requirements.txt`)
python scripts/download_data.py tartanair --sequences House/easy   # one environment / difficulty
```

`--sequences` picks a subset and `--root` overrides the destination.

## Where the loaders look

Each dataset root defaults to `data/<name>` inside the repo (git ignores `data/`;
a symlink to an existing copy works too) and is overridable by an environment
variable (`vi3/paths.py`):

| Dataset | Env var | Default root |
|---|---|---|
| TartanAir | `TARTANAIR_ROOT` | `data/tartanair` |
| EuRoC MAV | `EUROC_ROOT` | `data/euroc-mav` |
| UZH-FPV | `UZHFPV_ROOT` | `data/uzh-fpv/extracted` |

```bash
export TARTANAIR_ROOT=/path/to/tartanair
export EUROC_ROOT=/path/to/euroc-mav
export UZHFPV_ROOT=/path/to/uzh-fpv/extracted
```

The sequence names used by `--conditions` / `--suite` are the folder names below
(e.g. `House/easy/P000`, `MH_01_easy`, `indoor_forward_5`).

---

## TartanAir

Source: [TartanAir V2](https://tartanair.org/), via its official `tartanair` package.
The loader uses the **left front** camera (`lcam_front`); intrinsics are fixed
(640×640, 90° FOV, centered principal point) — no calibration file needed. GT
depth is dense (one PNG per frame). IMU is synthetic, bias-free.

Sequence label `<Env>/<difficulty>/<traj>` → `<Env>/Data_<difficulty>/<traj>/`:

```
$TARTANAIR_ROOT/
  House/Data_easy/P000/
    image_lcam_front/   000000_lcam_front.png ...          # RGB frames
    depth_lcam_front/   000000_lcam_front_depth.png ...     # dense GT depth
    pose_lcam_front.txt                                     # cam->world poses (tx ty tz qx qy qz qw)
    imu/
      acc.npy  gyro.npy  imu_time.npy  cam_time.npy         # required
      acc_nograv.npy  vel_global.npy  pos_global.npy  ori_global.npy
```

The four evaluated environments are `House`, `Hospital`, `ModernCityDowntown`,
`ForestEnv` (each with `Data_easy` / `Data_hard`, trajectories `P000`, `P002`).

---

## EuRoC MAV

Source: the [EuRoC MAV](https://projects.asl.ethz.ch/datasets/euroc-mav/)
ASL dataset. Calibration + extrinsics come from the per-sequence `sensor.yaml`.
GT depth is derived per-frame and cached (Leica pointcloud for the Vicon rooms,
stereo SGBM for Machine Hall). If neither depth source is present the sequence
still loads (the depth metric is just NaN).

Sequence label → `<category>/<label>/`:

| Label prefix | Category dir | Group | GT depth |
|---|---|---|---|
| `MH_0*` | `machine_hall/` | MH | stereo SGBM (needs `cam1/`) |
| `V1_0*` | `vicon_room1/` | V1 | Leica (`pointcloud0/`) |
| `V2_0*` | `vicon_room2/` | V2 | Leica (`pointcloud0/`) |

Each sequence is either a pre-extracted `mav0/` tree or the ASL `.zip` (the
loader unzips it on first access):

```
$EUROC_ROOT/machine_hall/MH_01_easy/
  MH_01_easy.zip                     # OR the extracted mav0/ below
  mav0/
    cam0/ sensor.yaml  data.csv  data/<ts_ns>.png ...     # required
    cam1/ sensor.yaml            data/<ts_ns>.png ...     # required for stereo depth (MH)
    imu0/ data.csv                                        # required
    state_groundtruth_estimate0/ data.csv                # required (pose+vel+bias GT)
    pointcloud0/ data.ply                                # Vicon rooms only (Leica depth)
```

Full sequence set: `MH_01_easy … MH_05_difficult`, `V1_01_easy … V1_03_difficult`,
`V2_01_easy … V2_03_difficult`.

The loader writes caches next to the data (`mav0/depth_cache_<H>x<W>/`) and an
undistorted-RGB cache under `/tmp/euroc_undist_cache/` — both regenerate on
demand, don't ship them.

---

## UZH-FPV

Source: the [UZH-FPV](https://fpv.ifi.uzh.ch/datasets/) drone dataset. The loader uses the
**Snapdragon** camera (fisheye, 640×480); calibration comes from a per-family
Kalibr `camchain-imucam-*.yaml`. GT is pose-only — velocity and biases are
derived on load and cached. GT depth is Snapdragon stereo SGBM.

`UZHFPV_ROOT` points at the already-extracted tree. Sequence label → sequence dir
`<label>/`; the calibration is shared per family (`indoor_forward`, `indoor_45`,
`outdoor_forward`, `outdoor_45`):

```
$UZHFPV_ROOT/
  calib/
    indoor_forward/ camchain-imucam-<...>_imu.yaml         # required (cam0 + cam1 Kalibr)
    indoor_45/ ...    outdoor_forward/ ...    outdoor_45/ ...
  indoor_forward_5/
    snapdragon/
      groundtruth.txt                                      # required (pose-only GT)
      imu.txt                                              # required
      left_images.txt                                      # required (id  ts  img/name)
      img/
        image_0_<N>.png ...                                # required (left)
        image_1_<N>.png ...                                # required for GT depth (right)
```

The family a sequence belongs to (which `calib/<family>/` it uses) is set in
`UZHFPV_SEQUENCES` in `uzhfpv.py`. The loader writes caches under `snapdragon/`
(`biases_cache_*.npz`, `depth_cache_<H>x<W>/`) and `/tmp/uzhfpv_undist_cache/`;
all regenerate on demand. `race_*` sequences are out of scope (no GT).
