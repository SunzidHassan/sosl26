# sosl26

Semantic odor source localization (olfaction + vision fusion) on a TurtleBot4 running ROS 2 Jazzy.
This is the TB4 port of the AI2-THOR `sOSL_main.py` / `fusion_controller.py` pipeline.

## Files (`sosl26/`)

| File | Role |
| --- | --- |
| `sOSL_olfactionFunctions.py`, `sOSL_visionFunctions.py`, `sOSL_utils.py`, `sOSL_loggerFunctions.py` | Unchanged copies of the AI2-THOR code base (the package `__init__` makes their flat imports work) |
| `sOSL_tb4Functions.py` | TB4 adapters: image decoding, depth-based 3D localisation, TB4 `visionBranch`, `TB4BayesianAgent` (plume with `q_s`, `D`, `tau`), search grid from `/map`, plots |
| `tb4_fusion_controller.py` | ROS 2 node + `ExperimentConfig` |
| `sOSL_tb4_main.py` | `main()`: experiment parameters |

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

Every `sample_period` seconds the node reads the pose (TF `map -> base_footprint`) and the latest `/olfaction`
(`Vector3`: x = wind direction, y = wind speed, z = concentration), RGB (`/oakd/rgb/image_raw/compressed`) and
depth (`/oakd/stereo/image_raw/compressedDepth`) messages. It then updates the Bayesian map, runs YOLO, and builds
`navKnowledge` with `add_goal_similarity`. It calls `navigate()`, which does nothing yet, and it never publishes
`/cmd_vel`, so teleop keeps control.

## Outputs

`<sosl26>/save/save_{alg}_{Odor}/{entropy_frac}/{run}/` (with `--symlink-install`; otherwise `./save`):

- `trajectory_log.csv`: one row per step, with the same columns as the AI2-THOR runs plus wind, the olfactory arg-max and time
- `maps_all_XXX_x_.._z_...png`: the olfactory, visual (`langSim`) and fused (`goalSim`) maps, plus `maps_XXX.npz` with the raw arrays
- `navKnowledge_XXX_*.csv`, `detected_objects_map_XXX.png`, `yolo_stepN.jpg`, `frame_*.png`, `depth_XXX.png` (mm)
- `envKnowledge_final.csv`, `navKnowledge_final.csv`, `olfaction_raw.csv` (every `/olfaction` message), `trajectory_plot.png`, `run_config.json`

**Axis convention:** to reuse the AI2-THOR functions unchanged, ROS map `y` is stored as the AI2-THOR `z` axis.
So `robot_z` is map y, and a `Position` string `"a, b, c"` means map (x=a, y=c, height=b).
