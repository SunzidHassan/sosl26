"""Shared ROS 2 (Jazzy) TurtleBot4 controller for semantic odor source localization.

TB4BaseController does everything that the olfactory-only ('O') and fusion
('F') experiments have in common. Every `sample_period` seconds it:

  1. reads the robot pose (TF map -> base_footprint),
  2. takes the latest /olfaction reading and updates the Bayesian source map,
  3. calls `perceive()` (implemented by the subclass) for vision / fusion,
  4. saves the maps and the trajectory log (same files as the AI2-THOR runs),
  5. calls navigate(), which is currently a no-op (drive with teleop).

Subclasses:
  tb4_olfactory_controller.TB4OlfactoryController  ('O')
  tb4_fusion_controller.TB4FusionController        ('F', vision_mode 'navKnowledge' | 'dirichlet')

See sOSL_tb4Functions.py for the ai2thor <-> ROS axis convention
(robot_z / z_points == ROS map y).
"""

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sOSL_loggerFunctions import map_entropy
from sOSL_olfactionFunctions import gaussian_plume
from sOSL_tb4Functions import (
    TB4BayesianAgent,
    decode_compressed_depth,
    decode_compressed_rgb,
    grid_from_bounds,
    grid_from_occupancy,
    intrinsics_from_fov,
    optical_to_map_from_robot_pose,
    plot_tb4_trajectory,
    ros_yaw_to_ai2thor_deg,
    save_belief_maps,
    scale_intrinsics,
    transform_point,
    yaw_from_quaternion,
)
from sOSL_utils import grid_to_world, parse_position_string, world_to_grid


# ==========================
# Experiment Configuration
# ==========================

@dataclass
class ExperimentConfig:
    """All experiment parameters. Filled in sOSL_tb4_main.main()."""

    # --- Task ---
    alg_choice: str = 'F'                      # 'F' fusion, 'O' olfactory-only
    vision_mode: str = 'navKnowledge'          # 'F' only: 'navKnowledge' (object list) | 'dirichlet' (object map)
    odor: str = "burnt"
    goal_phrase: Optional[str] = None          # default: f"Is emitting {odor} odor:"
    entropy_frac: float = 0.8

    # --- Sampling / run limits ---
    sample_period: float = 2.0                 # seconds between olfactory + visual readings
    step_threshold: int = 100
    run_time_limit: Optional[float] = None     # seconds, None = until Ctrl+C / 'q'

    # --- Olfaction (plume model) ---
    q_s: float = 2000.0
    D: float = 10.0
    tau: float = 1000.0
    U: float = 0.0
    psi_deg: float = 0.0
    sigma_noise: float = 1.5
    use_simulated_olfaction: bool = False      # replace /olfaction by gaussian_plume(source_position)

    # --- Map / grid ---
    grid_step: float = 0.25
    map_bounds: Optional[Tuple[float, float, float, float]] = None  # (x_min, x_max, y_min, y_max) map frame
    map_wait_timeout: float = 10.0
    fallback_map_size: float = 10.0            # square around the start pose if no /map and no bounds
    source_position: Optional[Tuple[float, float]] = None           # ground truth (x, y) map frame, optional

    # --- Vision (F only) ---
    yolo_conf: float = 0.3
    yolo_target_classes: Optional[Sequence[str]] = None             # None = all model classes
    yolo_exclude_classes: Sequence[str] = field(default_factory=list)
    depth_percentile: float = 50.0
    depth_range: Tuple[float, float] = (0.2, 5.0)
    merge_dist: float = 0.1
    camera_hfov_deg: float = 69.0              # only used until CameraInfo arrives
    camera_height: float = 0.25                # only used if the camera TF is unavailable

    # --- Dirichlet object map (vision_mode 'dirichlet') ---
    dirichlet_object_evidence: float = 1.0     # added to a class per detection footprint cell
    dirichlet_background_evidence: float = 0.5 # added to Background per observed-empty cell
    dirichlet_prior_strength: float = 1.0      # uniform prior: each class starts at prior_strength / K
    dirichlet_min_radius: Optional[float] = None  # footprint radius clip (m); None = grid_step / 2
    dirichlet_max_radius: float = 1.0
    dirichlet_background_similarity: float = 0.0  # sim(Background, goal) used in P(src | V)

    # --- ROS interfaces ---
    rgb_topic: str = "/oakd/rgb/image_raw/compressed"
    depth_topic: str = "/oakd/stereo/image_raw/compressedDepth"
    camera_info_topic: str = "/oakd/rgb/camera_info"
    olfaction_topic: str = "/olfaction"
    map_topic: str = "/map"
    cmd_vel_topic: str = "/cmd_vel"
    map_frame: str = "map"
    base_frame: str = "base_footprint"
    camera_frame: str = ""                     # "" = use the RGB image header frame_id

    show_window: bool = True

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

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Subscriptions (sensor QoS matches both reliable and best-effort publishers) ---
        if self.uses_camera:
            self.create_subscription(CompressedImage, cfg.rgb_topic, self.image_callback, qos_profile_sensor_data)
            self.create_subscription(CompressedImage, cfg.depth_topic, self.depth_callback, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, cfg.camera_info_topic, self.camera_info_callback,
                                     qos_profile_sensor_data)
        self.create_subscription(Vector3, cfg.olfaction_topic, self.olfactory_callback, qos_profile_sensor_data)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, cfg.map_topic, self.map_callback, map_qos)

        # Reserved for navigate(). Nothing is published yet, so keyboard
        # teleop on /cmd_vel is not overridden.
        self.cmd_vel_pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)

        # --- Sensor state ---
        self.latest_rgb = None
        self.rgb_frame_id = ""
        self.latest_depth = None
        self.cam_K = None            # (fx, fy, cx, cy) from CameraInfo
        self.cam_info_size = (0, 0)  # (width, height)
        self.olfactionWindDirection = 0.0
        self.olfactionWindSpeed = 0.0
        self.olfactionChemicalConc = 0.0
        self.olfaction_received = False
        self.olfaction_raw = []
        self.map_msg = None
        self.latest_annotated = None

        # --- Algorithm state ---
        self.grid = None
        self.bayesian_agent = None
        self.H_max = None
        self.entropy_threshold = None
        self.start_pose = None
        self.step_count = 0
        self.trajectory_log_list = []

        self.start_time = time.time()
        self.done = False
        self._finalized = False

        self.step_timer = self.create_timer(cfg.sample_period, self.step_callback)
        if cfg.show_window and self.uses_camera:
            self.ui_timer = self.create_timer(0.05, self.ui_callback)

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
            panels : list[(map, title)]  maps drawn in maps_all_*.png
            arrays : dict[str, np.ndarray]  saved in maps_XXX.npz
            target_object, target_coordinate (ai2thor "x, y, z" string), target_xz (map x, y) or None
            visual_entropy, fused_entropy : float
        """
        raise NotImplementedError

    def save_perception_outputs(self, step_count, tag):
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
        self.olfactionWindDirection = msg.x
        self.olfactionWindSpeed = msg.y
        self.olfactionChemicalConc = msg.z
        self.olfaction_received = True

        pose = self.get_robot_pose(warn=False)
        self.olfaction_raw.append({
            "time": round(time.time() - self.start_time, 3),
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

    def ui_callback(self):
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
            self.ui_timer.cancel()

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
        if cfg.map_bounds is not None:
            x_min, x_max, y_min, y_max = cfg.map_bounds
            self.grid = grid_from_bounds(x_min, x_max, y_min, y_max, cfg.grid_step)
            grid_src = "map_bounds"
        elif self.map_msg is not None:
            info = self.map_msg.info
            occ = np.array(self.map_msg.data, dtype=np.int8).reshape(info.height, info.width)
            self.grid = grid_from_occupancy(occ, info.resolution, info.origin.position.x,
                                            info.origin.position.y, cfg.grid_step)
            grid_src = cfg.map_topic
        elif time.time() - self.start_time > cfg.map_wait_timeout:
            half = cfg.fallback_map_size / 2.0
            self.get_logger().warn(
                f"No {cfg.map_topic} received and no map_bounds set. Using a {cfg.fallback_map_size} m "
                f"square around the start pose.")
            self.grid = grid_from_bounds(pose[0] - half, pose[0] + half, pose[1] - half, pose[1] + half,
                                         cfg.grid_step)
            grid_src = "fallback"
        else:
            self.get_logger().info(f"Waiting for {cfg.map_topic}...", throttle_duration_sec=2.0)
            return False

        g = self.grid
        self.start_pose = pose
        start_grid_pos = world_to_grid(pose[0], pose[1], g.x_points, g.z_points)
        src_grid_pos = (world_to_grid(cfg.source_position[0], cfg.source_position[1], g.x_points, g.z_points)
                        if cfg.source_position is not None else None)

        self.bayesian_agent = TB4BayesianAgent(
            pos=start_grid_pos,
            src_pos=src_grid_pos,
            x_points=g.x_points,
            z_points=g.z_points,
            sigma_noise=cfg.sigma_noise,
            reachable_positions=g.reachable_positions,
            q_s=cfg.q_s, D=cfg.D, tau=cfg.tau, U=cfg.U, psi_deg=cfg.psi_deg,
        )
        self.H_max = map_entropy(self.bayesian_agent.prob_map)
        self.entropy_threshold = self.H_max * cfg.entropy_frac

        self.setup_perception()

        self.get_logger().info(
            f"Search grid from {grid_src}: x [{g.x_points[0]:.2f}, {g.x_points[-1]:.2f}], "
            f"y [{g.z_points[0]:.2f}, {g.z_points[-1]:.2f}], {len(g.z_points)}x{len(g.x_points)} cells. "
            f"Start pose ({pose[0]:.2f}, {pose[1]:.2f}). H_max {self.H_max:.2f} bits, "
            f"threshold {self.entropy_threshold:.2f} bits ({cfg.entropy_frac * 100:.0f}%).")

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
            "H_max": float(self.H_max),
            "entropy_threshold": float(self.entropy_threshold),
        }
        run_info.update(self.run_info_extra())
        with open(os.path.join(self.save_dir, "run_config.json"), "w") as f:
            json.dump(run_info, f, indent=2, default=str)
        return True

    # ------------------------------------------------------------------
    # Main loop (every sample_period seconds)
    # ------------------------------------------------------------------

    def step_callback(self):
        if self.done:
            return
        cfg = self.cfg

        if cfg.run_time_limit is not None and time.time() - self.start_time > cfg.run_time_limit:
            self.get_logger().info(f"Run time limit {cfg.run_time_limit}s reached. Ending run.")
            self.done = True
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        needed = [("olfaction", self.olfaction_received or cfg.use_simulated_olfaction)]
        if self.uses_camera:
            needed += [("RGB", self.latest_rgb is not None), ("depth", self.latest_depth is not None)]
        missing = [name for name, ok in needed if not ok]
        if missing:
            self.get_logger().warn(f"Waiting for: {', '.join(missing)}", throttle_duration_sec=2.0)
            return

        if self.grid is None and not self.setup_search(pose):
            return

        try:
            self.run_step(pose)
        except Exception:
            # Keep the run alive (and its data) if a single step fails
            self.get_logger().error(f"Step {self.step_count} failed:\n{traceback.format_exc()}")

        if self.step_count >= cfg.step_threshold:
            self.get_logger().info(f"Step threshold {cfg.step_threshold} reached. Ending run.")
            self.done = True

    def read_olfaction(self, robot_x, robot_z):
        """(concentration, wind_direction, wind_speed) for this step."""
        cfg = self.cfg
        if cfg.use_simulated_olfaction and cfg.source_position is not None:
            conc = gaussian_plume(robot_x, robot_z, tuple(cfg.source_position),
                                  q_s=cfg.q_s, D=cfg.D, U=cfg.U, tau=cfg.tau, psi_deg=cfg.psi_deg)
            return conc + np.random.normal(0, cfg.sigma_noise), 0.0, 0.0
        return self.olfactionChemicalConc, self.olfactionWindDirection, self.olfactionWindSpeed

    def run_step(self, pose):
        cfg = self.cfg
        g = self.grid
        step_start_time = time.time()
        step_count = self.step_count
        robot_x, robot_z, robot_yaw = pose   # robot_z == map y (ai2thor naming); robot_yaw is ROS (rad, CCW from +x)

        self.get_logger().info(f"===== Step {step_count + 1}/{cfg.step_threshold} at ({robot_x:.2f}, {robot_z:.2f}) =====")

        # --- 1. Olfaction: Bayesian update ---
        current_odor_concentration, wind_direction, wind_speed = self.read_olfaction(robot_x, robot_z)
        self.bayesian_agent.posterior(current_odor_concentration, robot_x, robot_z)
        srcProbGivenOlfactory = self.bayesian_agent.prob_map
        olfactoryEntropy = map_entropy(srcProbGivenOlfactory)

        max_idx = np.unravel_index(np.argmax(srcProbGivenOlfactory, axis=None), srcProbGivenOlfactory.shape)
        olfactory_max_xz = grid_to_world(max_idx, g.x_points, g.z_points)

        # --- 2. Vision / fusion (subclass) ---
        result = self.perceive(pose, step_count, srcProbGivenOlfactory)

        if step_count == 0:
            behavior_flag = "Initialization"
        elif olfactoryEntropy > self.entropy_threshold:
            behavior_flag = "search"
        else:
            behavior_flag = "goal_navigation"

        # --- 3. Navigation (no-op for now) ---
        target_xz = result["target_xz"]
        self.navigate(behavior_flag, target_xz, olfactory_max_xz, pose)

        # --- 4. Evaluation against ground truth (if known) ---
        gt_distance, target_error = np.nan, np.nan
        if cfg.source_position is not None:
            src = np.asarray(cfg.source_position, dtype=float)
            gt_distance = float(np.linalg.norm(np.array([robot_x, robot_z]) - src))
            if target_xz is not None:
                target_error = float(np.linalg.norm(src - np.asarray(target_xz, dtype=float)))

        step_time = time.time() - step_start_time

        # --- 5. Save everything ---
        tag = f"{step_count:03d}_x_{robot_x:.2f}_z_{robot_z:.2f}"
        panels = [(srcProbGivenOlfactory, rf'$H_C={olfactoryEntropy:.2f}$')] + result["panels"]
        try:
            save_belief_maps(panels, g.x_points, g.z_points, os.path.join(self.save_dir, f"maps_all_{tag}.png"))
        except Exception as e:
            self.get_logger().error(f"Error saving belief map plot at step {step_count}: {e}")
        np.savez_compressed(os.path.join(self.save_dir, f"maps_{step_count:03d}.npz"),
                            olfactory=srcProbGivenOlfactory, x_points=g.x_points, z_points=g.z_points,
                            **result["arrays"])
        try:
            self.save_perception_outputs(step_count, tag)
        except Exception as e:
            self.get_logger().error(f"Error saving perception outputs at step {step_count}: {e}")

        self.trajectory_log_list.append({
            "step": step_count,
            "time": round(time.time() - self.start_time, 3),
            "robot_x": robot_x,
            "robot_z": robot_z,
            "robot_yaw": ros_yaw_to_ai2thor_deg(robot_yaw),   # ai2thor convention (deg, CW from map +y)
            "robot_yaw_ros_deg": math.degrees(robot_yaw),     # ROS convention (deg, CCW from map +x)
            "step_time": step_time,
            "behavior_flag": behavior_flag,
            "is_random": False,
            "target_object": result["target_object"],
            "target_coordinate": result["target_coordinate"],
            "target_coord_estimation_error": target_error,
            "olfactory_max_x": float(olfactory_max_xz[0]),
            "olfactory_max_z": float(olfactory_max_xz[1]),
            "concentration": current_odor_concentration,
            "wind_direction": wind_direction,
            "wind_speed": wind_speed,
            "gt_distance_from_source": gt_distance,
            "Bayesian_entropy": olfactoryEntropy,
            "visual_entropy": result["visual_entropy"],
            "fused_entropy": result["fused_entropy"],
            "entropy_threshold": self.entropy_threshold,
        })
        pd.DataFrame(self.trajectory_log_list).to_csv(os.path.join(self.save_dir, "trajectory_log.csv"), index=False)

        self.get_logger().info(
            f"conc={current_odor_concentration:.2f} H_C={olfactoryEntropy:.2f} (thr {self.entropy_threshold:.2f}) "
            f"flag={behavior_flag} target={result['target_object']} @ {result['target_coordinate']} "
            f"olf_max=({olfactory_max_xz[0]:.2f}, {olfactory_max_xz[1]:.2f}) step_time={step_time:.2f}s")

        self.step_count += 1

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, behavior_flag, target_xz, olfactory_max_xz, pose):
        """Moves the robot toward the current source estimate.

        Not implemented yet: the robot is driven manually (keyboard teleop),
        so this deliberately publishes nothing.

        Parameters
        ----------
        behavior_flag : str
            "Initialization", "search" or "goal_navigation" (fusion_controller semantics).
        target_xz : np.ndarray | None
            Map (x, y) of the current source estimate: the top navKnowledge
            object, the fused-map arg-max (dirichlet), or the olfactory
            arg-max (olfactory-only).
        olfactory_max_xz : np.ndarray
            Map (x, y) of the most likely source cell of the Bayesian map.
        pose : tuple
            Current robot (x, y, yaw) in the map frame, yaw in the ROS
            convention (rad, CCW from +x).

        Note: fusion_controller computes headings as atan2(dx, dz) in Unity's
        left-handed convention (CW from +z). Headings sent to the TB4 must use
        the ROS convention instead: sOSL_tb4Functions.ros_heading_to().
        """
        # TODO: send a Nav2 goal / publish self.cmd_vel_pub toward target_xz
        # (search: next waypoint toward the target, goal_navigation: go to it).
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
            if self.trajectory_log_list:
                pd.DataFrame(self.trajectory_log_list).to_csv(os.path.join(self.save_dir, "trajectory_log.csv"),
                                                              index=False)
            pd.DataFrame(self.olfaction_raw).to_csv(os.path.join(self.save_dir, "olfaction_raw.csv"), index=False)
        except Exception as e:
            print(f"Error saving final CSVs: {e}")

        try:
            self.finalize_perception()
        except Exception as e:
            print(f"Error saving final perception outputs: {e}")

        if self.grid is not None and self.trajectory_log_list:
            last = self.trajectory_log_list[-1]
            estimate_xz = None
            if last["target_coordinate"] not in ('N/A', None):
                p = parse_position_string(last["target_coordinate"])
                estimate_xz = (p[0], p[2])
            plot_tb4_trajectory(
                os.path.join(self.save_dir, "trajectory_log.csv"), self.grid,
                os.path.join(self.save_dir, "trajectory_plot.png"),
                source_xz=self.cfg.source_position, estimate_xz=estimate_xz,
                plume_params=dict(q_s=self.cfg.q_s, D=self.cfg.D, U=self.cfg.U,
                                  tau=self.cfg.tau, psi_deg=self.cfg.psi_deg),
            )
