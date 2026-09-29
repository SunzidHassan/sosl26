# sosl26

Semantic odor source localization (olfaction + vision fusion) on a TurtleBot4 running ROS 2 Jazzy.
This is the TB4 port of the AI2-THOR `sOSL_main.py` / `fusion_controller.py` pipeline.

## Files (`sosl26/`)

| File | Role |
| --- | --- |
| `sOSL_olfactionFunctions.py`, `sOSL_visionFunctions.py`, `sOSL_utils.py`, `sOSL_loggerFunctions.py` | Unchanged copies of the AI2-THOR code base (the package `__init__` makes their flat imports work) |
| `sOSL_tb4Functions.py` | TB4 adapters: image decoding, depth-based 3D localisation, TB4 `visionBranch`, `TB4BayesianAgent` (plume with `q_s`, `D`, `tau`), search grid from `/map`, plots |
| `sOSL_dirichletFunctions.py` | Dirichlet object map (second vision approach) |
| `tb4_base_controller.py` | `ExperimentConfig` + shared node: pose, map, olfaction, logging, `navigate()` stub |
| `tb4_olfactory_controller.py` | `'O'`: olfactory map only (no camera, no YOLO) |
| `tb4_fusion_controller.py` | `'F'`: olfaction + vision, `vision_mode = 'navKnowledge'` or `'dirichlet'` |
| `sOSL_tb4_main.py` | `main()`: experiment parameters |

## Experiments (set in `main()`)

- `alg_choice = 'O'`: olfactory-only. Each step saves the Bayesian map. The estimate is its arg-max cell.
- `alg_choice = 'F'`, `vision_mode = 'navKnowledge'`: the AI2-THOR approach as is. It builds an object list, then
  `add_goal_similarity` scores each object as `goalSim = langSim * olfactionSim`.
- `alg_choice = 'F'`, `vision_mode = 'dirichlet'`: builds a per-cell Dirichlet class distribution over the YOLO classes plus
  Background.
  - It starts from a uniform prior (`prior_strength / K`).
  - Each detection adds +1 to its class over a footprint of radius half the metric bbox width (`bbox_w_px * depth / fx / 2`).
  - Cells between the camera and each detection, inside the FOV and map bounds, get +0.5 Background.
  - Then `P(src | V) ∝ Σ_k p(o_k | z) · max(0, sim(class_k, goal))` (Background similarity 0), and the fused map is
    `P(src | C) · P(src | V)` normalised. The estimate is the fused arg-max.

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
  Each frame is projected with the pose at that moment. Its detections are merged into `envKnowledge`, and in Dirichlet
  mode each frame adds its own evidence (+1 per detection footprint, +0.5 per background cell). A 5 s window therefore
  adds up to about 10 evidence per object.
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

`<sosl26>/save/save_F_{Odor}_{vision_mode}/...` or `save_O_{Odor}/...`, then `{entropy_frac}/{run}/` (with `--symlink-install`; otherwise `./save`):

- `trajectory_log.csv`: one row per step, with the same columns as the AI2-THOR runs plus wind, the olfactory arg-max and time
- `maps_all_XXX_x_.._z_...png`: the olfactory, visual (`langSim`) and fused (`goalSim`) maps, plus `maps_XXX.npz` with the raw arrays
- `navKnowledge_XXX_*.csv`, `detected_objects_map_XXX.png`, `yolo_{step}_{frame}.jpg` (every processed frame; `save_vision_frames`), and `frame_*.png` / `depth_XXX.png` (mm) for the last frame of each step
- `envKnowledge_final.csv`, `navKnowledge_final.csv`, `detections_log.csv` (every detection), `olfaction_raw.csv` (every `/olfaction` message), `trajectory_plot.png`, `run_config.json`
- Dirichlet mode:
  - Each step: `object_map_XXX.png` (MLE class map; grey means never observed), plus `beta`, `observed` and `mle_class` in `maps_XXX.npz`.
  - At the end: `dirichlet_final.npz`, `dirichlet_classes.json`, `semantic_similarity_table.csv`, and `dirichlet_maps/` (per-class posteriors, MLE, entropies, semantic likelihood).
- Olfactory-only: `maps_all_*` has one panel and `maps_XXX.npz` holds only the olfactory map. No frames or detections are saved.

**Axis convention:** to reuse the AI2-THOR functions unchanged, ROS map `y` is stored as the AI2-THOR `z` axis.
So `robot_z` is map y, and a `Position` string `"a, b, c"` means map (x=a, y=c, height=b).

**Handedness:** AI2-THOR (Unity) is left-handed and ROS (REP 103) is right-handed. The y/z swap above is the
conversion between the two, so positions and top-down plots match. Angles do not match:

- ROS yaw is counter-clockwise from +x.
- AI2-THOR yaw is clockwise from +z (map +y).

`trajectory_log.csv` stores `robot_yaw` in the AI2-THOR convention (`(90 - ros_deg) mod 360`) and `robot_yaw_ros_deg`
in the ROS convention. Headings sent to the robot must use `ros_heading_to()`, not the `atan2(dx, dz)` in `fusion_controller.py`.
