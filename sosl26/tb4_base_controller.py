"""Experiment configuration and the olfactory TurtleBot4 controller (ROS 2 Jazzy).

TB4BaseController is the olfactory-only ('O') controller and the base class of
the fusion ('F') controller in tb4_fusion_controller.py.

The robot moves with Nav2 between fixed poses (cfg.reachable_cells x
cfg.headings_wz). The run is a loop driven by step_callback():

  navigating --(Nav2 result)--> arrived --> sampling (sample_period s, robot still)
       ^                                        |
       |                                        v
       +------------ send next goal <------ run_step():
                                              1. average the window's /olfaction readings and
                                                 update the Bayesian source map P(src | C),
                                              2. perceive(): source estimate (olfactory arg-max;
                                                 the fusion controller fuses in vision),
                                              3. plan_next_goal(),
                                              4. save the maps and one row of trajectory_log.csv.

The first stops are a scan: the robot turns to every heading at its start cell.
After that each goal is the adjacent reachable cell closest to the source
estimate, facing choose_heading(). Once the olfactory map is concentrated
(behavior flag 'goal_navigation') the robot drives to the reachable cell
closest to the estimate and the run ends; it also ends at step_threshold.

Source probabilities are 0 outside cfg.search_boundary.

Axis naming (see sOSL_tb4Functions.py): x = ROS map x, z = ROS map y.
"""

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Vector3
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sOSL_navigationFunctions import heading_towards, nearest_cell, next_cell, scan_order, wz_from_yaw, yaw_from_wz
from sOSL_olfactionFunctions import BayesianAgent, map_entropy
from sOSL_tb4Functions import (
    grid_from_polygon,
    plot_tb4_trajectory,
    ros_yaw_to_ai2thor_deg,
    save_belief_maps,
    yaw_from_quaternion,
)


# ==========================
# Experiment Configuration
# ==========================

@dataclass
class ExperimentConfig:
    """All experiment parameters, with defaults. The values used are set in sOSL_tb4_main.main()."""

    # --- Task ---
    alg_choice: str = 'F'                      # 'F' fusion (olfaction + Dirichlet vision), 'O' olfactory-only
    odor: str = "burnt"
    goal_phrase: Optional[str] = None          # default: f"Is emitting {odor} odor:"
    entropy_frac: float = 0.8                  # behavior flag: 'search' until H(P(src | C)) < entropy_frac * H_max

    # --- Sampling / run limits ---
    sample_period: float = 5.0                 # sampling time (s) at each stop: olfaction averaged, frames accumulate
    vision_period: float = 0.5                 # 'F': process a new camera frame at most every vision_period s
    save_vision_frames: bool = True            # 'F': save the annotated YOLO image of every processed frame
    step_threshold: int = 100                  # the run ends after this many steps
    run_time_limit: Optional[float] = None     # seconds, None = until Ctrl+C / 'q'

    # --- Olfaction (Gaussian plume model, see sOSL_olfactionFunctions.gaussian_plume) ---
    q_s: float = 2000.0                        # emission rate
    D: float = 10.0                            # diffusion coefficient
    tau: float = 1000.0                        # decay time
    U: float = 0.0                             # wind speed (0 = pure diffusion)
    psi_deg: float = 0.0                       # wind direction
    sigma_noise: float = 1.5                   # std of the sensor noise (rescaled units)
    # Linear rescaling of raw /olfaction readings before the Bayesian update:
    #   rescaled = olfaction_rescale_to * (raw - olfaction_min_conc) / (olfaction_max_conc - olfaction_min_conc)
    # (min -> 0, max -> olfaction_rescale_to; not clipped). None for both = use raw readings.
    olfaction_min_conc: Optional[float] = None
    olfaction_max_conc: Optional[float] = None
    olfaction_rescale_to: float = 100.0

    # --- Search area ---
    # Polygon corners [(x, y), ...] in the map frame, in order around the boundary. The grid
    # covers its bounding box with cells of grid_step m; source probabilities are 0 outside it.
    search_boundary: Sequence[Tuple[float, float]] = ((-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0))
    boundary_margin: float = 0.0               # cells within this distance (m) of the boundary count as inside
    grid_step: float = 0.25
    source_position: Optional[Tuple[float, float]] = None           # ground truth (x, y), for evaluation only

    # --- Navigation (Nav2) ---
    reachable_cells: Sequence[Tuple[float, float]] = ()   # robot stop positions (map x, y)
    # Headings as quaternions (w, z) about the vertical axis; (1, 0) faces map +x
    headings_wz: Sequence[Tuple[float, float]] = ((1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0), (0.7071, -0.7071))
    adjacency_radius: float = 1.5              # m; cells this close are adjacent (1.5 = 8-neighbourhood at 1 m)
    nav_action: str = "navigate_to_pose"       # Nav2 NavigateToPose action name
    nav_timeout: float = 120.0                 # s; a goal still running after this is cancelled

    # --- Vision ('F' only) ---
    yolo_model_path: str = ""
    yolo_conf: float = 0.3                     # minimum box confidence
    yolo_exclude_classes: Sequence[str] = field(default_factory=list)
    depth_percentile: float = 50.0             # object depth = this percentile of the depth pixels in its box
    depth_range: Tuple[float, float] = (0.2, 5.0)   # valid depth (m); the max is also the camera's view range
    camera_hfov_deg: float = 69.0              # only used until CameraInfo arrives
    camera_height: float = 0.25                # only used if the camera TF is unavailable
    sentence_model: str = 'all-mpnet-base-v2'  # embeds class names and the goal phrase for their similarity

    # --- Dirichlet object map ('F' only, see sOSL_dirichletFunctions.py) ---
    dirichlet_conf_matrix: Optional[Sequence[Sequence[float]]] = None  # normalized, rows predicted / cols true
    dirichlet_conf_labels: Optional[Sequence[str]] = None              # axis order, background last
    dirichlet_bg_fp_rate: float = 0.05            # P(any false detection | background region)
    dirichlet_conf_temper: float = 1.0            # evidence weight = YOLO confidence * conf_temper
    dirichlet_bg_false_neg_rate: Optional[float] = None  # eta; None = 1 - mean recall
    dirichlet_bg_dist_decay: float = 0.5          # lambda: free-space evidence decay with distance
    dirichlet_prior_strength: float = 1.0         # uniform prior: each class starts at prior_strength / K
    dirichlet_min_radius: Optional[float] = None  # footprint radius clip (m); None = grid_step / 2
    dirichlet_max_radius: float = 1.0
    dirichlet_background_similarity: float = 0.0  # sim(Background, goal) used in P(src | V)

    # --- ROS interfaces ---
    rgb_topic: str = "/oakd/rgb/image_raw/compressed"
    depth_topic: str = "/oakd/stereo/image_raw/compressedDepth"
    camera_info_topic: str = "/oakd/rgb/camera_info"
    olfaction_topic: str = "/olfaction"        # geometry_msgs/Vector3: x wind dir, y wind speed, z concentration
    map_frame: str = "map"
    base_frame: str = "base_footprint"
    camera_frame: str = ""                     # "" = use the RGB image header frame_id
    show_window: bool = True                   # 'F': show the annotated YOLO frames

    # --- Ground truth / plotting ---
    object_positions: Optional[Dict[str, Tuple[float, float]]] = None  # {class: (map x, map y)}
    object_colors: Optional[Dict[str, str]] = None                     # {class: matplotlib color}

    def resolved_goal_phrase(self):
        return self.goal_phrase or f"Is emitting {self.odor} odor:"


# ==========================
# Olfactory controller (base)
# ==========================

class TB4BaseController(Node):

    def __init__(self, cfg: ExperimentConfig, save_dir: str, node_name="sosl_tb4_olfactory_controller"):
        super().__init__(node_name)
        self.cfg = cfg
        self.save_dir = save_dir
        self.goal_phrase = cfg.resolved_goal_phrase()
        if (cfg.olfaction_min_conc is None) != (cfg.olfaction_max_conc is None):
            raise ValueError("Set both olfaction_min_conc and olfaction_max_conc, or neither.")
        if cfg.olfaction_min_conc is not None and cfg.olfaction_max_conc == cfg.olfaction_min_conc:
            raise ValueError("olfaction_max_conc must differ from olfaction_min_conc.")

        # Two callback groups: sensor callbacks keep running (in another executor
        # thread) while a step or a YOLO frame is being processed.
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.processing_group = MutuallyExclusiveCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subscribe(Vector3, cfg.olfaction_topic, self.olfactory_callback)
        self.nav_client = ActionClient(self, NavigateToPose, cfg.nav_action, callback_group=self.sensor_group)

        # --- Search grid and olfactory map (fixed by the boundary) ---
        self.grid = grid_from_polygon(cfg.search_boundary, cfg.grid_step, cfg.boundary_margin)
        self.bayesian_agent = BayesianAgent(
            self.grid.x_points, self.grid.z_points, cfg.sigma_noise, inside=self.grid.inside,
            q_s=cfg.q_s, D=cfg.D, tau=cfg.tau, U=cfg.U, psi_deg=cfg.psi_deg)
        self.H_max = map_entropy(self.bayesian_agent.prob_map)   # entropy of the uniform prior
        self.entropy_threshold = self.H_max * cfg.entropy_frac

        # --- Navigation poses ---
        if not cfg.reachable_cells:
            raise ValueError("Set reachable_cells: the positions the robot may navigate to.")
        self.cells = np.asarray(cfg.reachable_cells, dtype=float)
        self.headings = [yaw_from_wz(w, z) for w, z in cfg.headings_wz]   # ROS yaw (rad)
        outside = [tuple(c) for c in self.cells if not self.grid.contains(*c)]
        if outside:
            self.get_logger().warn(f"Reachable cells outside the search boundary: {outside}")

        # --- State ---
        # 'init' -> 'navigating' -> 'arrived' -> 'sampling' -> 'navigating' -> ... (see module docstring)
        self.state = 'init'
        self.goal = None             # current goal: dict(kind='scan'|'search'|'goal', x, y, yaw)
        self.goal_queue = []         # goals planned ahead (the rest of the initial scan)
        self.goal_sent_time = None
        self.nav_status = None       # how the last goal ended: 'succeeded', 'aborted', ...
        self.nav_time = np.nan       # how long it took (s)
        self._goal_handle = None
        self._cancel_requested = False
        self.olfaction_raw = []      # every /olfaction message, with the pose at that time
        self.window_start = None     # start of the current sampling window (s since start_time)
        self.window_olf_idx = 0      # index of the window's first reading in olfaction_raw
        self.step_count = 0
        self.trajectory_log_list = []
        self.last_target_xz = None
        self.start_time = time.time()
        self.done = False            # set when the run should end; main() then shuts down
        self._finalized = False

        # Polls every 0.1 s and advances the navigate / sample / update loop.
        self.step_timer = self.create_timer(0.1, self.step_callback, callback_group=self.processing_group)

        g = self.grid
        self.get_logger().info(
            f"sOSL TB4 {type(self).__name__} (alg '{cfg.alg_choice}'). Goal: '{self.goal_phrase}', "
            f"sample period {cfg.sample_period}s, plume q_s={cfg.q_s}, D={cfg.D}, tau={cfg.tau}. "
            f"Grid x [{g.x_points[0]:.2f}, {g.x_points[-1]:.2f}], y [{g.z_points[0]:.2f}, {g.z_points[-1]:.2f}], "
            f"{int(g.inside.sum())} cells inside the boundary. H_max {self.H_max:.2f} bits, "
            f"threshold {self.entropy_threshold:.2f} bits. Saving to {save_dir}")
        if cfg.source_position is not None and not g.contains(*cfg.source_position):
            self.get_logger().warn(f"Ground-truth source {cfg.source_position} lies outside the search boundary.")

    def subscribe(self, msg_type, topic, callback):
        """Sensor subscription; sensor QoS matches both reliable and best-effort publishers."""
        self.create_subscription(msg_type, topic, callback, qos_profile_sensor_data,
                                 callback_group=self.sensor_group)

    # ------------------------------------------------------------------
    # Hooks overridden by the fusion controller
    # ------------------------------------------------------------------

    def missing_inputs(self):
        """Names of sensors / servers not available yet; the run starts once this is empty."""
        missing = [] if self.olfaction_raw else ["olfaction"]
        if not self.nav_client.server_is_ready():
            missing.append(f"Nav2 action '{self.cfg.nav_action}'")
        return missing

    def choose_heading(self, cell_xy, target_xz):
        """Heading (ROS yaw) at the next cell: here the one facing the source estimate.
        The fusion controller picks the view with the most visual entropy instead."""
        return heading_towards(self.headings, cell_xy, target_xz)

    def perceive(self, pose, step_count, srcProbGivenOlfactory):
        """Source estimate for one step: here the olfactory arg-max.

        Returns a dict with
            panels : list[(map, title)]  extra maps drawn after the olfactory one in maps_all_*.png
            arrays : dict[str, np.ndarray]  extra arrays saved in maps_XXX.npz
            target_object, target_xz (map x, y), visual_entropy, fused_entropy
            log : dict  extra trajectory_log columns (optional)
        """
        target_xz, _ = self.grid.argmax_xz(srcProbGivenOlfactory)
        return dict(panels=[], arrays={}, target_object='N/A', target_xz=target_xz,
                    visual_entropy=np.nan, fused_entropy=np.nan)

    def save_perception_outputs(self, step_count, tag, pose):
        """Per-step files beyond the maps and the trajectory log."""

    def finalize_perception(self):
        """End-of-run files beyond the trajectory log."""

    def ui_update(self):
        """Called from the main thread about every 30 ms (the fusion controller shows the camera here)."""

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def olfactory_callback(self, msg: Vector3):
        # Every reading is kept with the pose at that moment: the step uses the
        # window mean, and the full list goes to olfaction_raw.csv.
        pose = self.get_robot_pose(warn=False)
        self.olfaction_raw.append({
            "time": round(self.now(), 3),
            "robot_x": pose[0] if pose else np.nan,
            "robot_z": pose[1] if pose else np.nan,
            "robot_yaw": ros_yaw_to_ai2thor_deg(pose[2]) if pose else np.nan,
            "robot_yaw_ros_deg": math.degrees(pose[2]) if pose else np.nan,
            "wind_direction": msg.x,
            "wind_speed": msg.y,
            "chemicalConc": msg.z,
        })

    def now(self):
        """Seconds since the node started."""
        return time.time() - self.start_time

    def get_robot_pose(self, warn=True):
        """Returns (x, y, yaw) of base_frame in map_frame (yaw: ROS, rad CCW from +x), or None."""
        try:
            t = self.tf_buffer.lookup_transform(self.cfg.map_frame, self.cfg.base_frame, rclpy.time.Time())
        except TransformException as ex:
            if warn:
                self.get_logger().warn(
                    f"Could not get {self.cfg.map_frame}->{self.cfg.base_frame}: {ex}", throttle_duration_sec=2.0)
            return None
        q = t.transform.rotation
        return (t.transform.translation.x, t.transform.translation.y,
                yaw_from_quaternion(q.x, q.y, q.z, q.w))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step_callback(self):
        """Advances the loop: navigating -> arrived -> sampling -> run_step -> next goal."""
        if self.done:
            return
        cfg = self.cfg
        if cfg.run_time_limit is not None and self.now() > cfg.run_time_limit:
            self.get_logger().info(f"Run time limit {cfg.run_time_limit}s reached. Ending run.")
            self.end_run()
            return

        if self.state == 'navigating':
            # Nav2 runs its own recoveries; cancel only if a goal takes far too long.
            if self.now() - self.goal_sent_time > cfg.nav_timeout and not self._cancel_requested:
                self.get_logger().warn(f"Goal not reached after {cfg.nav_timeout:.0f}s; cancelling it.")
                self._cancel_requested = True
                if self._goal_handle is not None:
                    self._goal_handle.cancel_goal_async()   # the result callback then reports 'canceled'
                else:
                    self.on_goal_done('timeout')
            return

        if self.state == 'arrived':
            if self.goal['kind'] == 'goal':
                self.get_logger().info(f"Reached the final goal ({self.nav_status}). Ending run.")
                self.done = True
                return
            # Robot stopped at a goal: collect sensor data for sample_period seconds
            self.start_window()
            self.state = 'sampling'
            self.get_logger().info(f"Sampling step {self.step_count + 1} for {cfg.sample_period:.1f}s...")
            return

        if self.state == 'sampling' and self.now() - self.window_start < cfg.sample_period:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        if self.state == 'init':
            missing = self.missing_inputs()
            if missing:
                self.get_logger().warn(f"Waiting for: {', '.join(missing)}", throttle_duration_sec=2.0)
                return
            # First step: turn to every heading at the start cell, sampling at each
            self.write_run_config(pose)
            cell = self.cells[nearest_cell(self.cells, pose[:2])]
            self.goal_queue = [dict(kind='scan', x=cell[0], y=cell[1], yaw=yaw)
                               for yaw in scan_order(self.headings, pose[2])]
            self.send_goal(self.goal_queue.pop(0))
            return

        # state == 'sampling' and the window is over: update the maps and plan the next goal
        try:
            next_goal = self.run_step(pose)
        except Exception:
            # Keep the run alive (and its data): sample again at the same place
            self.get_logger().error(f"Step {self.step_count} failed:\n{traceback.format_exc()}")
            self.start_window()
            return

        if self.step_count >= cfg.step_threshold:
            self.get_logger().info(f"Step threshold {cfg.step_threshold} reached. Ending run.")
            self.done = True
        else:
            self.send_goal(next_goal)

    def start_window(self):
        """Starts a sampling window: only /olfaction readings from now on are averaged."""
        self.window_start = self.now()
        self.window_olf_idx = len(self.olfaction_raw)

    # ------------------------------------------------------------------
    # Navigation (Nav2 NavigateToPose)
    # ------------------------------------------------------------------

    def plan_next_goal(self, behavior_flag, target_xz, pose):
        """Next goal: the rest of the scan, else an adjacent cell towards the source
        estimate, or (goal_navigation) the reachable cell closest to the estimate."""
        if self.goal_queue:
            return self.goal_queue.pop(0)
        if behavior_flag == "goal_navigation":
            # Final goal: the reachable cell closest to the estimate, facing it
            cell = self.cells[nearest_cell(self.cells, target_xz)]
            yaw = heading_towards(self.headings, cell, target_xz, default=pose[2])
            return dict(kind='goal', x=cell[0], y=cell[1], yaw=yaw)
        # The robot may not have reached its goal exactly: plan from the cell it is at
        idx = next_cell(self.cells, nearest_cell(self.cells, pose[:2]), target_xz, self.cfg.adjacency_radius)
        cell = self.cells[idx]
        return dict(kind='search', x=cell[0], y=cell[1], yaw=self.choose_heading(cell, target_xz))

    def send_goal(self, goal):
        """Sends a NavigateToPose goal; on_goal_done() is called when Nav2 finishes."""
        self.goal = goal
        msg = NavigateToPose.Goal()
        msg.pose.header.frame_id = self.cfg.map_frame
        msg.pose.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(goal['x'])
        msg.pose.pose.position.y = float(goal['y'])
        w, z = wz_from_yaw(goal['yaw'])
        msg.pose.pose.orientation.w = w
        msg.pose.pose.orientation.z = z
        self.state = 'navigating'
        self.goal_sent_time = self.now()
        self._goal_handle = None
        self._cancel_requested = False
        self.get_logger().info(f"Nav2 goal ({goal['kind']}): ({goal['x']:.2f}, {goal['y']:.2f}), "
                               f"yaw {math.degrees(goal['yaw']):.0f} deg")
        self.nav_client.send_goal_async(msg).add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 rejected the goal.")
            self.on_goal_done('rejected')
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        status = future.result().status
        names = {GoalStatus.STATUS_SUCCEEDED: 'succeeded', GoalStatus.STATUS_ABORTED: 'aborted',
                 GoalStatus.STATUS_CANCELED: 'canceled'}
        self.on_goal_done(names.get(status, f'status_{status}'))

    def on_goal_done(self, status):
        """Goal finished (reached or not): the next step_callback starts sampling."""
        self.nav_status = status
        self.nav_time = self.now() - self.goal_sent_time
        self._goal_handle = None
        if status != 'succeeded':
            self.get_logger().warn(f"Goal {status}; continuing from where the robot is.")
        self.state = 'arrived'

    def end_run(self):
        self.stop_navigation()
        self.done = True

    def stop_navigation(self):
        """Cancels a running Nav2 goal (called on shutdown)."""
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    def write_run_config(self, start_pose):
        g = self.grid
        run_info = {
            "controller": type(self).__name__,
            "config": asdict(self.cfg),
            "goal_phrase": self.goal_phrase,
            "start_pose": {"x": start_pose[0], "y": start_pose[1], "yaw_ros_deg": math.degrees(start_pose[2]),
                           "yaw_ai2thor_deg": ros_yaw_to_ai2thor_deg(start_pose[2])},
            "x_points": g.x_points.tolist(),
            "z_points(map_y)": g.z_points.tolist(),
            "H_max": float(self.H_max),
            "entropy_threshold": float(self.entropy_threshold),
        }
        with open(os.path.join(self.save_dir, "run_config.json"), "w") as f:
            json.dump(run_info, f, indent=2, default=str)

    def rescale_concentration(self, raw):
        """Maps a raw reading linearly so olfaction_min_conc -> 0 and olfaction_max_conc -> olfaction_rescale_to."""
        cfg = self.cfg
        if cfg.olfaction_min_conc is None:
            return raw
        a = cfg.olfaction_rescale_to / (cfg.olfaction_max_conc - cfg.olfaction_min_conc)
        return a * (raw - cfg.olfaction_min_conc)

    def read_olfaction(self):
        """Mean olfaction over the current step window (the latest reading if the window is empty)."""
        window = self.olfaction_raw[self.window_olf_idx:]
        if not window:
            self.get_logger().warn("No /olfaction readings in this step window; using the latest reading.")
        readings = window or self.olfaction_raw[-1:]
        conc = np.array([r["chemicalConc"] for r in readings], dtype=float)
        raw = float(conc.mean())
        return dict(raw=raw, concentration=self.rescale_concentration(raw), raw_std=float(conc.std()),
                    n_readings=len(window),
                    wind_direction=float(np.mean([r["wind_direction"] for r in readings])),
                    wind_speed=float(np.mean([r["wind_speed"] for r in readings])))

    def run_step(self, pose):
        cfg = self.cfg
        g = self.grid
        step_start_time = time.time()
        step_count = self.step_count
        robot_x, robot_z, robot_yaw = pose   # robot_z == map y; robot_yaw is ROS (rad, CCW from +x)

        self.get_logger().info(
            f"===== Step {step_count + 1}/{cfg.step_threshold} at ({robot_x:.2f}, {robot_z:.2f}) =====")

        # --- 1. Olfaction: Bayesian update with the window mean, at the current pose ---
        olf = self.read_olfaction()
        self.bayesian_agent.posterior(olf["concentration"], robot_x, robot_z)
        srcProbGivenOlfactory = self.bayesian_agent.prob_map
        olfactoryEntropy = map_entropy(srcProbGivenOlfactory)
        olfactory_max_xz, _ = g.argmax_xz(srcProbGivenOlfactory)

        # --- 2. Source estimate (olfactory arg-max here, fused with vision in 'F') ---
        result = self.perceive(pose, step_count, srcProbGivenOlfactory)
        target_xz = result["target_xz"]
        self.last_target_xz = target_xz

        # Behavior flag: initial scan, then keep searching while the olfactory map is
        # still uncertain, and go to the estimate once it is concentrated.
        if self.goal['kind'] == 'scan':
            behavior_flag = "Initialization"
        elif olfactoryEntropy > self.entropy_threshold:
            behavior_flag = "search"
        else:
            behavior_flag = "goal_navigation"

        # --- 3. Next Nav2 goal (sent by step_callback after this step) ---
        next_goal = self.plan_next_goal(behavior_flag, target_xz, pose)

        # --- 4. Evaluation against the ground truth (if known) ---
        gt_distance, target_error = np.nan, np.nan
        if cfg.source_position is not None:
            src = np.asarray(cfg.source_position, dtype=float)
            gt_distance = float(np.linalg.norm(np.array([robot_x, robot_z]) - src))
            target_error = float(np.linalg.norm(src - target_xz))

        step_time = time.time() - step_start_time

        # --- 5. Save the maps (png + npz), per-step perception files and the trajectory log ---
        tag = f"{step_count:03d}_x_{robot_x:.2f}_z_{robot_z:.2f}"
        panels = [(srcProbGivenOlfactory, rf'$H_C={olfactoryEntropy:.2f}$')] + result["panels"]
        path = [(r["robot_x"], r["robot_z"]) for r in self.olfaction_raw if np.isfinite(r["robot_x"])]
        try:
            save_belief_maps(panels, g, os.path.join(self.save_dir, f"maps_all_{tag}.png"),
                             trajectory=path + [(robot_x, robot_z)], boundary=cfg.search_boundary)
        except Exception as e:
            self.get_logger().error(f"Error saving belief map plot at step {step_count}: {e}")
        np.savez_compressed(os.path.join(self.save_dir, f"maps_{step_count:03d}.npz"),
                            olfactory=srcProbGivenOlfactory, x_points=g.x_points, z_points=g.z_points,
                            inside=g.inside, **result["arrays"])
        try:
            self.save_perception_outputs(step_count, tag, pose)
        except Exception as e:
            self.get_logger().error(f"Error saving perception outputs at step {step_count}: {e}")

        self.trajectory_log_list.append({
            "step": step_count,
            "time": round(self.now(), 3),
            "robot_x": robot_x,
            "robot_z": robot_z,
            "robot_yaw": ros_yaw_to_ai2thor_deg(robot_yaw),   # ai2thor convention (deg, CW from map +y)
            "robot_yaw_ros_deg": math.degrees(robot_yaw),     # ROS convention (deg, CCW from map +x)
            "step_time": step_time,
            "behavior_flag": behavior_flag,
            # The goal that brought the robot here, how it ended, and the goal planned next
            "goal_kind": self.goal['kind'],
            "goal_x": self.goal['x'],
            "goal_z": self.goal['y'],
            "goal_yaw_ros_deg": math.degrees(self.goal['yaw']),
            "nav_status": self.nav_status,
            "nav_time": self.nav_time,
            "next_goal_kind": next_goal['kind'],
            "next_goal_x": next_goal['x'],
            "next_goal_z": next_goal['y'],
            "next_goal_yaw_ros_deg": math.degrees(next_goal['yaw']),
            "target_object": result["target_object"],
            "target_x": float(target_xz[0]),
            "target_z": float(target_xz[1]),
            "target_coord_estimation_error": target_error,
            "olfactory_max_x": float(olfactory_max_xz[0]),
            "olfactory_max_z": float(olfactory_max_xz[1]),
            "concentration": olf["concentration"],         # rescaled, used by the Bayesian update
            "raw_concentration": olf["raw"],               # mean over the step window
            "raw_concentration_std": olf["raw_std"],
            "olfaction_readings": olf["n_readings"],
            "window_start": round(self.window_start, 3),
            "wind_direction": olf["wind_direction"],
            "wind_speed": olf["wind_speed"],
            "gt_distance_from_source": gt_distance,
            "Bayesian_entropy": olfactoryEntropy,
            "visual_entropy": result["visual_entropy"],
            "fused_entropy": result["fused_entropy"],
            "entropy_threshold": self.entropy_threshold,
            **result.get("log", {}),
        })
        # Rewritten every step so a crash never loses the log
        pd.DataFrame(self.trajectory_log_list).to_csv(os.path.join(self.save_dir, "trajectory_log.csv"), index=False)

        self.get_logger().info(
            f"conc={olf['concentration']:.2f} (raw mean {olf['raw']:.2f} of {olf['n_readings']}) "
            f"H_C={olfactoryEntropy:.2f} (thr {self.entropy_threshold:.2f}) flag={behavior_flag} "
            f"target={result['target_object']} @ ({target_xz[0]:.2f}, {target_xz[1]:.2f}) "
            f"olf_max=({olfactory_max_xz[0]:.2f}, {olfactory_max_xz[1]:.2f}) step_time={step_time:.2f}s")

        self.step_count += 1
        return next_goal

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def finalize(self):
        """Writes the end-of-run files. Safe to call more than once."""
        if self._finalized:
            return
        self._finalized = True
        print(f"Finalizing run after {self.step_count} steps. Saving to {self.save_dir}")

        try:
            pd.DataFrame(self.olfaction_raw).to_csv(os.path.join(self.save_dir, "olfaction_raw.csv"), index=False)
        except Exception as e:
            print(f"Error saving olfaction_raw.csv: {e}")

        try:
            self.finalize_perception()
        except Exception as e:
            print(f"Error saving final perception outputs: {e}")

        if self.trajectory_log_list:
            cfg = self.cfg
            try:
                plot_tb4_trajectory(
                    os.path.join(self.save_dir, "trajectory_log.csv"), self.grid,
                    os.path.join(self.save_dir, "trajectory_plot.png"),
                    source_xz=cfg.source_position, estimate_xz=self.last_target_xz,
                    plume_params=dict(q_s=cfg.q_s, D=cfg.D, U=cfg.U, tau=cfg.tau, psi_deg=cfg.psi_deg),
                    boundary=cfg.search_boundary, reachable_cells=self.cells)
            except Exception as e:
                print(f"Error saving trajectory plot: {e}")
