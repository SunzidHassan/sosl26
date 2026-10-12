"""Shared ROS 2 (Jazzy) TurtleBot4 controller for semantic odor source localization.

TB4BaseController does everything that the olfactory-only ('O') and fusion
('F') experiments have in common. Each step lasts `sample_period` seconds
(stop the robot while it samples). At the end of each step it:

  1. reads the robot pose (TF map -> base_footprint),
  2. averages all /olfaction readings of the step window and updates the
     Bayesian source map with that mean,
  3. calls `perceive()` (implemented by the subclass) for vision / fusion; the
     fusion controller processes camera frames continuously during the window,
  4. saves the maps and the trajectory log,
  5. calls navigate(), which is currently a no-op (drive with teleop).

Source probabilities are 0 outside `search_boundary` (if set).

Subclasses:
  tb4_olfactory_controller.TB4OlfactoryController  ('O')
  tb4_fusion_controller.TB4FusionController        ('F', Dirichlet object map)

See sOSL_tb4Functions.py for the axis convention (robot_z / z_points == ROS map y).
"""

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Vector3
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sOSL_olfactionFunctions import BayesianAgent, map_entropy
from sOSL_tb4Functions import (
    decode_compressed_depth,
    decode_compressed_rgb,
    grid_from_occupancy,
    grid_from_polygon,
    intrinsics_from_fov,
    optical_to_map_from_robot_pose,
    plot_tb4_trajectory,
    ros_yaw_to_ai2thor_deg,
    save_belief_maps,
    scale_intrinsics,
    transform_point,
    yaw_from_quaternion,
)


# ==========================
# Experiment Configuration
# ==========================

@dataclass
class ExperimentConfig:
    """All experiment parameters. Filled in sOSL_tb4_main.main()."""

    # --- Task ---
    alg_choice: str = 'F'                      # 'F' fusion (olfaction + Dirichlet vision), 'O' olfactory-only
    odor: str = "burnt"
    goal_phrase: Optional[str] = None          # default: f"Is emitting {odor} odor:"
    entropy_frac: float = 0.8

    # --- Sampling / run limits ---
    sample_period: float = 5.0                 # step window (s): olfaction is averaged, vision frames accumulate
    vision_period: float = 0.5                 # 'F': process a new camera frame at most every vision_period s
    save_vision_frames: bool = True            # 'F': save the annotated YOLO image of every processed frame
    step_threshold: int = 100
    run_time_limit: Optional[float] = None     # seconds, None = until Ctrl+C / 'q'

    # --- Olfaction (plume model) ---
    q_s: float = 2000.0
    D: float = 10.0
    tau: float = 1000.0
    U: float = 0.0
    psi_deg: float = 0.0
    sigma_noise: float = 1.5
    # Linear rescaling of raw /olfaction readings before the Bayesian update:
    #   rescaled = olfaction_rescale_to * (raw - olfaction_min_conc) / (olfaction_max_conc - olfaction_min_conc)
    # (min -> 0, max -> olfaction_rescale_to; not clipped). None for both = use raw readings.
    olfaction_min_conc: Optional[float] = None
    olfaction_max_conc: Optional[float] = None
    olfaction_rescale_to: float = 100.0

    # --- Map / grid ---
    grid_step: float = 0.25
    # Search area: polygon corners [(x, y), ...] in the map frame, in order around the boundary.
    # The grid covers its bounding box and source probabilities are 0 outside it.
    # None = grid from /map, no boundary.
    search_boundary: Optional[Sequence[Tuple[float, float]]] = None
    boundary_margin: float = 0.0               # cells within this distance (m) of the boundary count as inside
    source_position: Optional[Tuple[float, float]] = None           # ground truth (x, y) map frame, optional

    # --- Vision (F only) ---
    yolo_conf: float = 0.3
    yolo_exclude_classes: Sequence[str] = field(default_factory=list)
    depth_percentile: float = 50.0
    depth_range: Tuple[float, float] = (0.2, 5.0)
    camera_hfov_deg: float = 69.0              # only used until CameraInfo arrives
    camera_height: float = 0.25                # only used if the camera TF is unavailable
    sentence_model: str = 'all-mpnet-base-v2'  # class name <-> goal phrase similarity

    # --- Dirichlet object map ---
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
    olfaction_topic: str = "/olfaction"
    map_topic: str = "/map"
    map_frame: str = "map"
    base_frame: str = "base_footprint"
    camera_frame: str = ""                     # "" = use the RGB image header frame_id

    show_window: bool = True

    # --- Ground truth / plotting ---
    object_positions: Optional[Dict[str, Tuple[float, float]]] = None  # {class: (map x, map y)}
    object_colors: Optional[Dict[str, str]] = None                     # {class: matplotlib color}

    def resolved_goal_phrase(self):
        return self.goal_phrase or f"Is emitting {self.odor} odor:"


# ==========================
# Base Controller Node
# ==========================

class TB4BaseController(Node):

    uses_camera = False

    def __init__(self, cfg: ExperimentConfig, save_dir: str, node_name="sosl_tb4_controller"):
        super().__init__(node_name)
        self.cfg = cfg
        self.save_dir = save_dir
        self.goal_phrase = cfg.resolved_goal_phrase()
        if (cfg.olfaction_min_conc is None) != (cfg.olfaction_max_conc is None):
            raise ValueError("Set both olfaction_min_conc and olfaction_max_conc, or neither.")
        if cfg.olfaction_min_conc is not None and cfg.olfaction_max_conc == cfg.olfaction_min_conc:
            raise ValueError("olfaction_max_conc must differ from olfaction_min_conc.")

        # Sensor callbacks run in their own group so they keep up while YOLO / the
        # map updates (processing group) run; use a MultiThreadedExecutor (see main).
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.processing_group = MutuallyExclusiveCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Subscriptions (sensor QoS matches both reliable and best-effort publishers) ---
        def subscribe(msg_type, topic, callback, qos=qos_profile_sensor_data):
            self.create_subscription(msg_type, topic, callback, qos, callback_group=self.sensor_group)

        if self.uses_camera:
            subscribe(CompressedImage, cfg.rgb_topic, self.image_callback)
            subscribe(CompressedImage, cfg.depth_topic, self.depth_callback)
            subscribe(CameraInfo, cfg.camera_info_topic, self.camera_info_callback)
        subscribe(Vector3, cfg.olfaction_topic, self.olfactory_callback)
        if cfg.search_boundary is None:
            subscribe(OccupancyGrid, cfg.map_topic, self.map_callback,
                      QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # --- Sensor state ---
        self.latest_rgb = None
        self.rgb_seq = 0             # incremented per decoded RGB frame
        self.rgb_frame_id = ""
        self.latest_depth = None
        self.latest_annotated = None
        self.cam_K = None            # (fx, fy, cx, cy) from CameraInfo
        self.cam_info_size = (0, 0)  # (width, height)
        self.olfaction_raw = []      # every /olfaction message
        self.map_msg = None

        # --- Algorithm state ---
        self.grid = None
        self.bayesian_agent = None
        self.entropy_threshold = None
        self.step_count = 0
        self.trajectory_log_list = []
        self.last_target_xz = None
        self.window_start = None     # start of the current step window (s since start_time)
        self.window_olf_idx = 0      # index of the window's first reading in olfaction_raw

        self.start_time = time.time()
        self.done = False
        self._finalized = False

        # Polls; a step runs once its window has lasted sample_period seconds.
        self.step_timer = self.create_timer(0.1, self.step_callback, callback_group=self.processing_group)
        self.show_window = cfg.show_window and self.uses_camera

        self.get_logger().info(
            f"sOSL TB4 {type(self).__name__} started (alg '{cfg.alg_choice}'). Goal: '{self.goal_phrase}', "
            f"sample period {cfg.sample_period}s, plume q_s={cfg.q_s}, D={cfg.D}, tau={cfg.tau}. "
            f"Saving to {save_dir}")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def setup_perception(self):
        """Called once, after the search grid and Bayesian agent exist."""

    def perceive(self, pose, step_count, srcProbGivenOlfactory):
        """Vision + fusion for one step.

        Returns
        -------
        dict with keys
            panels : list[(map, title)]  maps drawn in maps_all_*.png after the olfactory one
            arrays : dict[str, np.ndarray]  saved in maps_XXX.npz
            target_object, target_xz (map x, y)
            visual_entropy, fused_entropy : float
            log : dict  extra trajectory_log columns (optional)
        """
        raise NotImplementedError

    def save_perception_outputs(self, step_count, tag, pose):
        """Per-step files beyond the maps and the trajectory log."""

    def finalize_perception(self):
        """End-of-run files beyond the trajectory log."""

    def run_info_extra(self):
        return {}

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        img = decode_compressed_rgb(msg.data)
        if img is None:
            self.get_logger().error("Failed to decode RGB image", throttle_duration_sec=5.0)
            return
        self.latest_rgb = img
        self.rgb_frame_id = msg.header.frame_id
        self.rgb_seq += 1

    def depth_callback(self, msg: CompressedImage):
        depth = decode_compressed_depth(msg.data, msg.format)
        if depth is None:
            self.get_logger().error("Failed to decode depth image", throttle_duration_sec=5.0)
            return
        self.latest_depth = depth

    def camera_info_callback(self, msg: CameraInfo):
        k = msg.k
        if k[0] > 0:
            self.cam_K = (k[0], k[4], k[2], k[5])
            self.cam_info_size = (msg.width, msg.height)

    def olfactory_callback(self, msg: Vector3):
        # x: wind direction, y: wind speed, z: chemical concentration
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

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg

    def ui_update(self):
        """Shows the latest (annotated) frame; call from the main thread (OpenCV GUI is not thread-safe)."""
        if not self.show_window:
            return
        frame = self.latest_annotated if self.latest_annotated is not None else self.latest_rgb
        if frame is None:
            return
        try:
            cv2.imshow("sOSL TB4 - YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("'q' pressed. Ending run.")
                self.done = True
        except cv2.error as e:
            self.get_logger().warn(f"Disabling OpenCV window: {e}")
            self.show_window = False

    def now(self):
        return time.time() - self.start_time

    # ------------------------------------------------------------------
    # Pose / camera helpers
    # ------------------------------------------------------------------

    def get_robot_pose(self, warn=True):
        """Returns (x, y, yaw) of base_frame in map_frame, or None."""
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

    def camera_intrinsics(self, rgb_shape):
        h, w = rgb_shape[:2]
        if self.cam_K is not None:
            return scale_intrinsics(self.cam_K, self.cam_info_size, (w, h))
        self.get_logger().warn("No CameraInfo yet, using HFOV intrinsics.", throttle_duration_sec=10.0)
        return intrinsics_from_fov(w, h, self.cfg.camera_hfov_deg)

    def optical_to_map_fn(self, pose):
        """Returns a function mapping camera optical-frame points into the map frame."""
        cam_frame = self.cfg.camera_frame or self.rgb_frame_id
        if cam_frame:
            try:
                t = self.tf_buffer.lookup_transform(self.cfg.map_frame, cam_frame, rclpy.time.Time())
                tr = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
                q = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
                return lambda p: transform_point(p, tr, q)
            except TransformException as ex:
                self.get_logger().warn(
                    f"No TF {self.cfg.map_frame}->{cam_frame} ({ex}); using robot pose + camera_height.",
                    throttle_duration_sec=10.0)
        rx, ry, ryaw = pose
        return lambda p: optical_to_map_from_robot_pose(p, rx, ry, ryaw, self.cfg.camera_height)

    # ------------------------------------------------------------------
    # Setup (grid + Bayesian agent) once pose and map are known
    # ------------------------------------------------------------------

    def setup_search(self, pose):
        cfg = self.cfg
        if cfg.search_boundary is not None:
            self.grid = grid_from_polygon(cfg.search_boundary, cfg.grid_step, cfg.boundary_margin)
            grid_src = "search_boundary"
        elif self.map_msg is not None:
            info = self.map_msg.info
            occ = np.array(self.map_msg.data, dtype=np.int8).reshape(info.height, info.width)
            self.grid = grid_from_occupancy(occ, info.resolution, info.origin.position.x,
                                            info.origin.position.y, cfg.grid_step)
            grid_src = cfg.map_topic
        else:
            self.get_logger().info(f"Waiting for {cfg.map_topic}...", throttle_duration_sec=2.0)
            return False

        g = self.grid
        self.bayesian_agent = BayesianAgent(
            g.x_points, g.z_points, cfg.sigma_noise, inside=g.inside,
            q_s=cfg.q_s, D=cfg.D, tau=cfg.tau, U=cfg.U, psi_deg=cfg.psi_deg)
        H_max = map_entropy(self.bayesian_agent.prob_map)
        self.entropy_threshold = H_max * cfg.entropy_frac

        self.setup_perception()

        self.get_logger().info(
            f"Search grid from {grid_src}: x [{g.x_points[0]:.2f}, {g.x_points[-1]:.2f}], "
            f"y [{g.z_points[0]:.2f}, {g.z_points[-1]:.2f}], {len(g.z_points)}x{len(g.x_points)} cells, "
            f"{int(g.inside.sum())} inside the boundary. Start pose ({pose[0]:.2f}, {pose[1]:.2f}). "
            f"H_max {H_max:.2f} bits, threshold {self.entropy_threshold:.2f} bits ({cfg.entropy_frac * 100:.0f}%).")
        if cfg.source_position is not None and self.cell_is_outside(*cfg.source_position):
            self.get_logger().warn(f"Ground-truth source {cfg.source_position} lies outside the search boundary.")

        run_info = {
            "controller": type(self).__name__,
            "config": asdict(cfg),
            "goal_phrase": self.goal_phrase,
            "start_pose": {"x": pose[0], "y": pose[1], "yaw_ros_deg": math.degrees(pose[2]),
                           "yaw_ai2thor_deg": ros_yaw_to_ai2thor_deg(pose[2])},
            "grid_source": grid_src,
            "grid_bounds": list(map(float, g.bounds)),
            "x_points": g.x_points.tolist(),
            "z_points(map_y)": g.z_points.tolist(),
            "H_max": float(H_max),
            "entropy_threshold": float(self.entropy_threshold),
        }
        run_info.update(self.run_info_extra())
        with open(os.path.join(self.save_dir, "run_config.json"), "w") as f:
            json.dump(run_info, f, indent=2, default=str)
        return True

    def cell_is_outside(self, x, y):
        """True if the grid cell nearest to map (x, y) is outside the search boundary."""
        g = self.grid
        return not g.inside[np.abs(g.z_points - y).argmin(), np.abs(g.x_points - x).argmin()]

    def argmax_xz(self, prob_map):
        """Map (x, y) of the most likely cell, and its (row, col)."""
        row, col = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        return np.array([self.grid.x_points[col], self.grid.z_points[row]]), (row, col)

    # ------------------------------------------------------------------
    # Main loop (every sample_period seconds)
    # ------------------------------------------------------------------

    def step_callback(self):
        if self.done:
            return
        cfg = self.cfg
        if self.window_start is not None and self.now() - self.window_start < cfg.sample_period:
            return

        if cfg.run_time_limit is not None and self.now() > cfg.run_time_limit:
            self.get_logger().info(f"Run time limit {cfg.run_time_limit}s reached. Ending run.")
            self.done = True
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        needed = [("olfaction", bool(self.olfaction_raw))]
        if self.uses_camera:
            needed += [("RGB", self.latest_rgb is not None), ("depth", self.latest_depth is not None)]
        missing = [name for name, ok in needed if not ok]
        if missing:
            self.get_logger().warn(f"Waiting for: {', '.join(missing)}", throttle_duration_sec=2.0)
            return

        if self.grid is None:
            if self.setup_search(pose):
                # Step 0 starts now: it averages olfaction / accumulates vision over a full window.
                self.start_window()
                self.get_logger().info(f"Sampling step 1 for {cfg.sample_period:.1f}s...")
            return

        try:
            self.run_step(pose)
        except Exception:
            # Keep the run alive (and its data) if a single step fails
            self.get_logger().error(f"Step {self.step_count} failed:\n{traceback.format_exc()}")
        self.start_window()

        if self.step_count >= cfg.step_threshold:
            self.get_logger().info(f"Step threshold {cfg.step_threshold} reached. Ending run.")
            self.done = True

    def start_window(self):
        """Starts the next step window: later /olfaction readings belong to the next step."""
        self.window_start = self.now()
        self.window_olf_idx = len(self.olfaction_raw)

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

    def path_so_far(self, pose):
        """Robot path: every pose recorded with an /olfaction reading, plus the current pose."""
        pts = [(r["robot_x"], r["robot_z"]) for r in self.olfaction_raw if np.isfinite(r["robot_x"])]
        pts.append((pose[0], pose[1]))
        return pts

    def run_step(self, pose):
        cfg = self.cfg
        g = self.grid
        step_start_time = time.time()
        step_count = self.step_count
        robot_x, robot_z, robot_yaw = pose   # robot_z == map y; robot_yaw is ROS (rad, CCW from +x)

        self.get_logger().info(
            f"===== Step {step_count + 1}/{cfg.step_threshold} at ({robot_x:.2f}, {robot_z:.2f}) =====")

        # --- 1. Olfaction: Bayesian update ---
        olf = self.read_olfaction()
        self.bayesian_agent.posterior(olf["concentration"], robot_x, robot_z)
        srcProbGivenOlfactory = self.bayesian_agent.prob_map
        olfactoryEntropy = map_entropy(srcProbGivenOlfactory)
        olfactory_max_xz, _ = self.argmax_xz(srcProbGivenOlfactory)

        # --- 2. Vision / fusion (subclass) ---
        result = self.perceive(pose, step_count, srcProbGivenOlfactory)
        target_xz = result["target_xz"]
        self.last_target_xz = target_xz

        if step_count == 0:
            behavior_flag = "Initialization"
        elif olfactoryEntropy > self.entropy_threshold:
            behavior_flag = "search"
        else:
            behavior_flag = "goal_navigation"

        # --- 3. Navigation (no-op for now) ---
        self.navigate(behavior_flag, target_xz, olfactory_max_xz, pose)

        # --- 4. Evaluation against ground truth (if known) ---
        gt_distance, target_error = np.nan, np.nan
        if cfg.source_position is not None:
            src = np.asarray(cfg.source_position, dtype=float)
            gt_distance = float(np.linalg.norm(np.array([robot_x, robot_z]) - src))
            target_error = float(np.linalg.norm(src - target_xz))

        step_time = time.time() - step_start_time

        # --- 5. Save everything ---
        tag = f"{step_count:03d}_x_{robot_x:.2f}_z_{robot_z:.2f}"
        panels = [(srcProbGivenOlfactory, rf'$H_C={olfactoryEntropy:.2f}$')] + result["panels"]
        try:
            save_belief_maps(panels, g.x_points, g.z_points, os.path.join(self.save_dir, f"maps_all_{tag}.png"),
                             trajectory=self.path_so_far(pose), boundary=cfg.search_boundary)
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
        pd.DataFrame(self.trajectory_log_list).to_csv(os.path.join(self.save_dir, "trajectory_log.csv"), index=False)

        self.get_logger().info(
            f"conc={olf['concentration']:.2f} (raw mean {olf['raw']:.2f} of {olf['n_readings']}) "
            f"H_C={olfactoryEntropy:.2f} (thr {self.entropy_threshold:.2f}) flag={behavior_flag} "
            f"target={result['target_object']} @ ({target_xz[0]:.2f}, {target_xz[1]:.2f}) "
            f"olf_max=({olfactory_max_xz[0]:.2f}, {olfactory_max_xz[1]:.2f}) step_time={step_time:.2f}s")

        self.step_count += 1

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, behavior_flag, target_xz, olfactory_max_xz, pose):
        """Moves the robot toward the current source estimate.

        Not implemented yet: the robot is driven manually (keyboard teleop),
        so this deliberately publishes nothing.

        behavior_flag : "Initialization", "search" or "goal_navigation".
        target_xz : map (x, y) of the source estimate (fused or olfactory arg-max).
        olfactory_max_xz : map (x, y) of the most likely cell of the Bayesian map.
        pose : robot (x, y, yaw) in the map frame. Headings sent to the TB4 must
            use the ROS convention (atan2(dy, dx)), not AI2-THOR's atan2(dx, dz).
        """
        return None

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
                    boundary=cfg.search_boundary)
            except Exception as e:
                print(f"Error saving trajectory plot: {e}")
