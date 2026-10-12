# sosl26

Semantic odor source localization (olfaction + vision fusion) on a TurtleBot4 running ROS 2 Jazzy.
This is the TB4 port of the AI2-THOR `sOSL_main.py` / `fusion_controller.py` pipeline.

## Files (`sosl26/`)

| File | Role |
| --- | --- |
| `sOSL_olfactionFunctions.py` | Gaussian plume, `BayesianAgent` (plume with `q_s`, `D`, `tau`), `map_entropy` |
| `sOSL_tb4Functions.py` | TB4 helpers: image decoding, depth-based 3D localisation, YOLO `detect_objects`, search grid (boundary polygon or `/map`), plots |
| `sOSL_dirichletFunctions.py` | Dirichlet object map (the vision branch) |
| `tb4_base_controller.py` | `ExperimentConfig` + shared node: pose, map, olfaction, logging, `navigate()` stub |
| `tb4_olfactory_controller.py` | `'O'`: olfactory map only (no camera, no YOLO) |
| `tb4_fusion_controller.py` | `'F'`: olfaction + Dirichlet vision |
| `sOSL_tb4_main.py` | `main()`: experiment parameters |

## Experiments (set in `main()`)

- `alg_choice = 'O'`: olfactory-only. Each step saves the Bayesian map. The estimate is its arg-max cell.
- `alg_choice = 'F'`: builds a per-cell Dirichlet class distribution over the YOLO classes plus
  Background.
  - It starts from a uniform prior (`prior_strength / K`).
  - Each detection adds `confidence · (confusion-matrix row of its class)` over a footprint of radius half the metric
    bbox width (`bbox_w_px * depth / fx / 2`). Same-class overlaps in one frame are max-merged.
  - Every visible cell without a detection gets Background evidence `(1 - eta) / (1 + lambda · dist)`.
  - Then `P(src | V) ∝ Σ_k p(o_k | z) · max(0, sim(class_k, goal))` (Background similarity 0), and the fused map is
    `P(src | C) · P(src | V)` normalised. The estimate is the fused arg-max.

## Search boundary

`search_boundary` in `main()` lists the corners `(x, y)` of the search area in the map frame, in order around it.
The grid covers its bounding box, and the olfactory, visual and fused source probabilities are 0 outside it.
`boundary_margin` (m) also counts cells that close to the boundary as inside. With `search_boundary = None` the grid
comes from `/map` and has no boundary.

## Build and run

```bash
pip install ultralytics sentence-transformers
cd ~/<ws> && colcon build --symlink-install --packages-select sosl26 && source install/setup.bash

# Terminal 1: localization in a saved map (publishes /map and map -> base_footprint)
# Terminal 2: keyboard control
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# Terminal 3
ros2 run sosl26 sosl_tb4
```

Stop with `Ctrl+C` (or `q` in the YOLO window). The run also ends at `step_threshold` or `run_time_limit`.

Each step is a window of `sample_period` seconds (`n`, default 5 s). Stop the robot while it samples.

- **Olfaction:** every `/olfaction` message in the window (`Vector3`: x = wind direction, y = wind speed, z = concentration)
  is averaged. The Bayesian map is updated once per step with that mean, at the pose at the end of the window.
- **Vision (`'F'`):** YOLO runs on every new frame, at most one per `vision_period` (default 0.5 s), throughout the window.
  Each frame is projected with the pose at that moment. Each frame adds its own evidence to the Dirichlet map. A 5 s window therefore adds up to about 10 frames of evidence.
- **End of the step:** maps are fused and saved, and `navigate()` is called. `navigate()` does nothing yet, and the node
  never publishes `/cmd_vel`, so teleop keeps control.

Sensor callbacks and processing run in separate threads (`MultiThreadedExecutor`), so `/olfaction` readings are not
dropped while YOLO runs.

## Olfactory readings

Before the Bayesian update, raw `/olfaction` concentrations are rescaled linearly, as in the offline mapping:

`rescaled = 100 · (raw − min) / (max − min)`

- Set `olfaction_min_conc` and `olfaction_max_conc` in `main()`. Each is a number or the path of an `olfaction_data.csv`, in which case its mean `chemicalConc` is used.
- Values are not clipped, so readings below `min` become negative.
- Leave both `None` to use raw readings.
- `trajectory_log.csv` logs the window mean `raw_concentration`, its `raw_concentration_std`, the number of readings
  (`olfaction_readings`), and the rescaled `concentration`. With `'F'` it also logs `vision_frames` and `vision_detections` per step.

## Outputs

`<sosl26>/save/save_F_{Odor}_dirichlet/...` or `save_O_{Odor}/...`, then `{entropy_frac}/{run}/` (with `--symlink-install`; otherwise `./save`):

- `trajectory_log.csv`: one row per step: pose, source estimate (`target_x`, `target_z`), olfaction, wind, the olfactory arg-max, entropies and time
- `maps_all_XXX_x_.._z_...png`: the olfactory, visual `P(src | V)` and fused maps (boundary dashed), plus `maps_XXX.npz` with the raw arrays and the `inside` mask
- `olfaction_raw.csv` (every `/olfaction` message), `trajectory_plot.png`, `run_config.json`
- Fusion:
  - `yolo_detections/yolo_{step}_{frame}.jpg` (every processed frame; `save_vision_frames`), `frames/frame_*.png` / `depth/depth_XXX.png` (mm) for the last frame of each step, `detections_log.csv` (every detection).
  - Each step: `object_map_XXX.png` (MLE class map; grey means never observed), plus `beta`, `observed` and `mle_class` in `maps_XXX.npz`.
  - At the end: `dirichlet_final.npz`, `dirichlet_classes.json`, `semantic_similarity_table.csv`, and `dirichlet_maps/` (per-class posteriors, MLE, entropies, semantic likelihood).
- Olfactory-only: `maps_all_*` has one panel and `maps_XXX.npz` holds only the olfactory map. No frames or detections are saved.

**Axis convention:** as in the AI2-THOR code, ROS map `y` is stored as the `z` axis, so `robot_z`, `target_z` and
`z_points` are map y.

**Handedness:** AI2-THOR (Unity) is left-handed and ROS (REP 103) is right-handed. The y/z swap above is the
conversion between the two, so positions and top-down plots match. Angles do not match:

- ROS yaw is counter-clockwise from +x.
- AI2-THOR yaw is clockwise from +z (map +y).

`trajectory_log.csv` stores `robot_yaw` in the AI2-THOR convention (`(90 - ros_deg) mod 360`) and `robot_yaw_ros_deg`
in the ROS convention. Headings sent to the robot must use the ROS `atan2(dy, dx)`, not the `atan2(dx, dz)` in `fusion_controller.py`.
