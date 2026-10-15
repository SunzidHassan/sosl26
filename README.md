# sosl26

Semantic odor source localization (olfaction + vision fusion) on a TurtleBot4 running ROS 2 Jazzy.
This is the TB4 port of the AI2-THOR `sOSL_main.py` / `fusion_controller.py` pipeline.

## Files (`sosl26/`)

| File | Role |
| --- | --- |
| `sOSL_tb4_main.py` | `main()`: the experiment parameters, run folder, ROS executor loop |
| `tb4_base_controller.py` | `ExperimentConfig` (every parameter with its default) and `TB4BaseController`: the olfactory-only (`'O'`) node (pose, olfaction, Bayesian map, step loop, logging, `navigate()` stub) |
| `tb4_fusion_controller.py` | `TB4FusionController` (`'F'`): adds the camera, YOLO and the Dirichlet map, and fuses vision with olfaction |
| `sOSL_olfactionFunctions.py` | Gaussian plume, `BayesianAgent`, `map_entropy` |
| `sOSL_navigationFunctions.py` | Navigation planning without ROS: headings, adjacent cells, next cell, scan order |
| `sOSL_dirichletFunctions.py` | Dirichlet object map (the vision branch) |
| `sOSL_tb4Functions.py` | Helpers without ROS: image decoding, camera projection, YOLO `detect_objects`, search grid and boundary, plots |

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
`boundary_margin` (m) also counts cells that close to the boundary as inside. `/map` is not used.

## Navigation (Nav2)

The robot only stops at `reachable_cells` (map x, y) facing one of `headings_wz` (quaternions `(w, z)`:
180°, 90°, 0°, -90°). It moves between them with Nav2 `NavigateToPose` goals:

1. **Initial scan (the first step):** the robot drives to `initial_position` (set in `main()`, now (-1, -2)) facing
   the first heading, then turns to each of the other headings there, in `headings_wz` order (180°, 90°, 0°, -90°).
   It samples and updates the maps at each one, so the scan is 4 steps with `behavior_flag = Initialization`.
   With `initial_position = None` it scans at the reachable cell nearest its start pose.
2. **Search:** after each step the next goal is the cell, among the current cell and its adjacent reachable cells
   (within `adjacency_radius` = 1.5 m, i.e. 8 neighbours including diagonals), closest to the source estimate (the
   fused arg-max; the olfactory arg-max for `'O'`). If no neighbour is closer, the robot stays and only turns.
   - `'F'`: the heading is the one whose camera view (field of view, up to the max depth, inside the boundary) covers
     the most visual entropy, i.e. the least certain cells of the Dirichlet object map.
   - `'O'`: the heading faces the source estimate.
3. **Goal navigation:** once the olfactory map is concentrated (entropy below `entropy_frac · H_max`), the robot
   drives to the reachable cell closest to the estimate, facing it, and the run ends.

The run also ends at `step_threshold` steps (stops). A goal that Nav2 aborts or rejects, or that takes longer than
`nav_timeout`, is logged in `nav_status`, and the robot samples where it is.

## Build and run

```bash
pip install ultralytics sentence-transformers
cd ~/<ws> && colcon build --symlink-install --packages-select sosl26 && source install/setup.bash

# Terminal 1: localization in a saved map (publishes the TF map -> base_footprint)
# Terminal 2: Nav2 (provides the navigate_to_pose action)
# Terminal 3
ros2 run sosl26 sosl_tb4
```

Stop with `Ctrl+C` (or `q` in the YOLO window); a running Nav2 goal is cancelled. The run also ends at
`step_threshold`, after the final goal navigation, or at `run_time_limit`.

Each step is one stop: the robot reaches its goal, then samples for `sample_period` seconds (default 5 s).

- **Olfaction:** every `/olfaction` message in the window (`Vector3`: x = wind direction, y = wind speed, z = concentration)
  is averaged. The Bayesian map is updated once per step with that mean, at the pose at the end of the window.
  Readings taken while driving are only logged in `olfaction_raw.csv`.
- **Vision (`'F'`):** YOLO runs on the newest frame, at most one per `vision_period`, only while sampling.
  Each frame is projected with the pose at that moment and adds its own evidence to the Dirichlet map.
- **End of the step:** maps are fused and saved, the next goal is planned and sent to Nav2.

Sensor callbacks and processing run in separate threads (`MultiThreadedExecutor`), so `/olfaction` readings are not
dropped while YOLO runs.

## Olfactory readings

Before the Bayesian update, raw `/olfaction` concentrations are rescaled linearly, as in the offline mapping:

`rescaled = 100 · (raw − min) / (max − min)`

- Set `olfaction_min_conc` and `olfaction_max_conc` in `main()`.
- Values are not clipped, so readings below `min` become negative.
- Leave both `None` to use raw readings.
- `trajectory_log.csv` logs the window mean `raw_concentration`, its `raw_concentration_std`, the number of readings
  (`olfaction_readings`), and the rescaled `concentration`. With `'F'` it also logs `vision_frames` and `vision_detections` per step.

## Outputs

`<sosl26>/save/save_F_{Odor}_dirichlet/...` or `save_O_{Odor}/...`, then `{entropy_frac}/{run}/` (with `--symlink-install`; otherwise `./save`):

- `trajectory_log.csv`: one row per step: pose, source estimate (`target_x`, `target_z`), olfaction, wind, the olfactory arg-max, entropies, time, and navigation (`goal_*` and `nav_status` / `nav_time` of the goal that led to this stop, `next_goal_*` planned from it)
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
