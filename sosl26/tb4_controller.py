"""ROS 2 (Jazzy) TurtleBot4 node for semantic odor source localization (sOSL).

One node runs both experiments:
  'O' olfactory-only : olfaction -> Bayesian source map
  'F' fusion         : the same, plus the camera branch in tb4_vision.Vision

Timeline of a run
-----------------
  1. Wait until the robot pose (TF map -> base_footprint), /olfaction (and for
     'F' the camera) are available, then build the search grid from /map and
     the uniform Bayesian map. The first step window starts.
  2. Each step is a window of `sample_period` seconds (stop the robot):
       - every /olfaction message is stored (olfactory_callback);
       - 'F': every new camera frame, at most one per vision_period, goes through
         YOLO and adds evidence (vision_callback -> Vision.process_frame).
  3. When the window is over (run_step):
       - the mean concentration of the window updates the Bayesian map;
       - 'F': Vision.fuse() combines vision with that map;
       - navigate() is called (empty for now: drive with keyboard teleop);
       - maps, frames and the trajectory log are saved; the next window starts.
  4. Ctrl+C, 'q' in the YOLO window, step_threshold or run_time_limit end the
     run; finalize() writes the end-of-run files.

Threads: sensor callbacks and processing (steps, YOLO) are in two callback
groups, so with a MultiThreadedExecutor /olfaction keeps being recorded while
YOLO runs. spin_with_ui() sets this up and drives the OpenCV window from the
main thread (OpenCV windows are not thread-safe).

Coordinates: positions are in the map frame; to reuse the AI2-THOR code, map y
is called "z" (robot_z, z_points). See sOSL_tb4Functions.py.
"""

import json
import math
import os
import threading
import time
import traceback
from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sOSL_olfactionFunctions import gaussian_plume
from sOSL_tb4Functions import (
    TB4BayesianAgent,
    decode_compressed_depth,
    decode_compressed_rgb,
    format_position,
    grid_from_bounds,
    grid_from_occupancy,
    grid_to_world,
    intrinsics_from_fov,
    map_entropy,
    optical_to_map_from_robot_pose,
    parse_position_string,
    plot_tb4_trajectory,
    ros_yaw_to_ai2thor_deg,
    save_belief_maps,
    scale_intrinsics,
    transform_point,
    world_to_grid,
    yaw_from_quaternion,
)
from sosl26.tb4_config import ExperimentConfig
from sosl26.tb4_vision import StepEstimate, Vision


class TB4Controller(Node):

    def __init__(self, cfg: ExperimentConfig, save_dir: str, yolo_model=None):
        super().__init__("sosl_tb4_controller")
        cfg.validate()
        self.cfg = cfg
        self.save_dir = save_dir
        self.goal_phrase = cfg.resolved_goal_phrase()

        # Camera branch: only for fusion runs
        self.uses_camera = cfg.alg_choice == 'F'
        self.vision = (Vision(cfg, yolo_model, self.goal_phrase, save_dir, self.get_logger())
                       if self.uses_camera else None)

        # Sensor callbacks and processing get separate callback groups so that
        # sensors keep being read while a step or a YOLO frame is processed.
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.processing_group = MutuallyExclusiveCallbackGroup()

        # Robot and camera poses come from TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- Subscriptions ----------------
        # Sensor QoS (best effort) matches both reliable and best-effort publishers.
        sub = lambda msg_type, topic, cb, qos=qos_profile_sensor_data: self.create_subscription(  # noqa: E731
            msg_type, topic, cb, qos, callback_group=self.sensor_group)
        if self.uses_camera:
            sub(CompressedImage, cfg.rgb_topic, self.image_callback)
            sub(CompressedImage, cfg.depth_topic, self.depth_callback)
            sub(CameraInfo, cfg.camera_info_topic, self.camera_info_callback)
        sub(Vector3, cfg.olfaction_topic, self.olfactory_callback)
        # /map is published once ("latched"), so it needs a transient-local subscription
        sub(OccupancyGrid, cfg.map_topic, self.map_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # Reserved for navigate(). Nothing is published, so keyboard teleop keeps control.
        self.cmd_vel_pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)

        # ---------------- Latest sensor data ----------------
        self.latest_rgb = None
        self.rgb_seq = 0                 # counts RGB frames, so each frame is processed only once
        self.rgb_frame_id = ""
        self.latest_depth = None
        self.cam_K = None                # (fx, fy, cx, cy) from CameraInfo
        self.cam_info_size = (0, 0)      # (width, height) the intrinsics refer to
        self.latest_annotated = None     # last YOLO image, shown in the window
        self.map_msg = None
        self.olfaction_raw = []          # every /olfaction message of the run -> olfaction_raw.csv
        self.olfaction_received = False

        # ---------------- Search state ----------------
        self.grid = None                 # SearchGrid, built once /map is available
        self.bayesian_agent = None       # olfactory source map
        self.entropy_threshold = None
        self.step_count = 0
        self.trajectory_log_list = []
        self.window_start = None         # start time of the current step window (None = not started)
        self.window_olf_idx = 0          # first olfaction_raw entry of the current window
        self._last_processed_seq = 0     # rgb_seq of the last frame given to YOLO

        self.start_time = time.time()
        self.done = False                # set to end the run
        self._finalized = False

        # ---------------- Timers ----------------
        # The step timer only polls: a step runs once its window has lasted sample_period.
        self.create_timer(0.1, self.step_callback, callback_group=self.processing_group)
        if self.uses_camera:
            self.create_timer(cfg.vision_period, self.vision_callback, callback_group=self.processing_group)
        self.show_window = cfg.show_window and self.uses_camera

        mode = "fusion (olfaction + Dirichlet vision)" if self.uses_camera else "olfactory-only"
        self.get_logger().info(
            f"sOSL TB4 controller started: {mode}. Goal: '{self.goal_phrase}', step window {cfg.sample_period}s, "
            f"plume q_s={cfg.q_s}, D={cfg.D}, tau={cfg.tau}. Saving to {save_dir}")

    def now(self):
        """Seconds since the node started."""
        return time.time() - self.start_time

    # ==================================================================
    # Sensor callbacks: only store the latest data
    # ==================================================================

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
        if msg.k[0] > 0:
            self.cam_K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.cam_info_size = (msg.width, msg.height)

    def olfactory_callback(self, msg: Vector3):
        """Stores every reading with the robot pose; run_step averages the readings of each window."""
        # Vector3 fields: x = wind direction, y = wind speed, z = chemical concentration
        self.olfaction_received = True
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

    # ==================================================================
    # Pose and camera geometry
    # ==================================================================

    def get_robot_pose(self, warn=True):
        """(x, y, yaw) of the robot in the map frame (yaw: ROS convention, rad), or None."""
        try:
            t = self.tf_buffer.lookup_transform(self.cfg.map_frame, self.cfg.base_frame, rclpy.time.Time())
        except TransformException as ex:
            if warn:
                self.get_logger().warn(f"Could not get {self.cfg.map_frame}->{self.cfg.base_frame}: {ex}",
                                       throttle_duration_sec=2.0)
            return None
        q = t.transform.rotation
        return t.transform.translation.x, t.transform.translation.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def camera_intrinsics(self, rgb_shape):
        """(fx, fy, cx, cy) for an image of this shape (from CameraInfo, else from the configured FOV)."""
        h, w = rgb_shape[:2]
        if self.cam_K is not None:
            return scale_intrinsics(self.cam_K, self.cam_info_size, (w, h))
        self.get_logger().warn("No CameraInfo yet, using HFOV intrinsics.", throttle_duration_sec=10.0)
        return intrinsics_from_fov(w, h, self.cfg.camera_hfov_deg)

    def optical_to_map_fn(self, pose):
        """Function mapping a point in the camera optical frame to the map frame.

        Uses the TF of the camera frame (exact: includes the camera's mounting
        position and tilt). If that TF is missing, falls back to a level camera
        at camera_height looking along the robot heading.
        """
        cam_frame = self.cfg.camera_frame or self.rgb_frame_id
        if cam_frame:
            try:
                t = self.tf_buffer.lookup_transform(self.cfg.map_frame, cam_frame, rclpy.time.Time())
                tr = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
                q = (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
                return lambda p: transform_point(p, tr, q)
            except TransformException as ex:
                self.get_logger().warn(f"No TF {self.cfg.map_frame}->{cam_frame} ({ex}); "
                                       f"using robot pose + camera_height.", throttle_duration_sec=10.0)
        rx, ry, ryaw = pose
        return lambda p: optical_to_map_from_robot_pose(p, rx, ry, ryaw, self.cfg.camera_height)

    # ==================================================================
    # Setup: search grid + uniform Bayesian map (once)
    # ==================================================================

    def setup_search(self, pose):
        """Builds the search grid and the Bayesian map. Returns False while still waiting for /map."""
        cfg = self.cfg

        # Grid bounds: fixed bounds, else the /map occupancy grid, else a square around the robot
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
        elif self.now() > cfg.map_wait_timeout:
            half = cfg.fallback_map_size / 2.0
            self.get_logger().warn(f"No {cfg.map_topic} received and no map_bounds set. "
                                   f"Using a {cfg.fallback_map_size} m square around the start pose.")
            self.grid = grid_from_bounds(pose[0] - half, pose[0] + half, pose[1] - half, pose[1] + half,
                                         cfg.grid_step)
            grid_src = "fallback"
        else:
            self.get_logger().info(f"Waiting for {cfg.map_topic}...", throttle_duration_sec=2.0)
            return False

        # Restrict the search to the configured polygon (cells outside get probability 0)
        g = self.grid
        g.set_search_area(cfg.search_area)
        if not g.area_mask.any():
            raise ValueError(f"search_area {cfg.search_area} contains no grid cell centre "
                             f"(grid x {g.x_points[0]}..{g.x_points[-1]}, y {g.z_points[0]}..{g.z_points[-1]})")

        # Bayesian map over the grid, uniform over the search area (the start pose needs no config)
        src_cell = (world_to_grid(cfg.source_position[0], cfg.source_position[1], g.x_points, g.z_points)
                    if cfg.source_position is not None else None)
        self.bayesian_agent = TB4BayesianAgent(
            pos=world_to_grid(pose[0], pose[1], g.x_points, g.z_points), src_pos=src_cell,
            x_points=g.x_points, z_points=g.z_points, sigma_noise=cfg.sigma_noise,
            reachable_positions=g.reachable_positions,
            q_s=cfg.q_s, D=cfg.D, tau=cfg.tau, U=cfg.U, psi_deg=cfg.psi_deg, area_mask=g.area_mask)

        # Entropy of the uniform map is the maximum; the behaviour flag switches at entropy_frac of it
        H_max = map_entropy(self.bayesian_agent.prob_map)
        self.entropy_threshold = H_max * cfg.entropy_frac

        if self.vision is not None:
            self.vision.setup(g)

        self.get_logger().info(
            f"Search grid from {grid_src}: x [{g.x_points[0]:.2f}, {g.x_points[-1]:.2f}], "
            f"y [{g.z_points[0]:.2f}, {g.z_points[-1]:.2f}], {len(g.z_points)}x{len(g.x_points)} cells, "
            f"{int(g.area_mask.sum())} inside the search area. "
            f"Start pose ({pose[0]:.2f}, {pose[1]:.2f}). H_max {H_max:.2f} bits, "
            f"threshold {self.entropy_threshold:.2f} bits ({cfg.entropy_frac * 100:.0f}%).")

        # Everything needed to interpret the run later
        run_info = {
            "config": asdict(cfg),
            "goal_phrase": self.goal_phrase,
            "start_pose": {"x": pose[0], "y": pose[1], "yaw_ros_deg": math.degrees(pose[2]),
                           "yaw_ai2thor_deg": ros_yaw_to_ai2thor_deg(pose[2])},
            "grid_source": grid_src,
            "grid_bounds": list(map(float, g.bounds)),
            "x_points": g.x_points.tolist(),
            "z_points(map_y)": g.z_points.tolist(),
            "search_area": g.area_polygon,
            "search_area_cells": int(g.area_mask.sum()),
            "H_max": float(H_max),
            "entropy_threshold": float(self.entropy_threshold),
            **(self.vision.run_info() if self.vision is not None else {}),
        }
        with open(os.path.join(self.save_dir, "run_config.json"), "w") as f:
            json.dump(run_info, f, indent=2, default=str)
        return True

    # ==================================================================
    # Step loop
    # ==================================================================

    def step_callback(self):
        """Polled every 0.1 s: sets up the search, then runs a step at the end of each window."""
        cfg = self.cfg
        if self.done:
            return
        if self.window_start is not None and self.now() - self.window_start < cfg.sample_period:
            return    # window still running
        if cfg.run_time_limit is not None and self.now() > cfg.run_time_limit:
            self.get_logger().info(f"Run time limit {cfg.run_time_limit}s reached. Ending run.")
            self.done = True
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        # Before the first step: wait for the sensors, then set up and open the first window
        if self.grid is None:
            needed = [("olfaction", self.olfaction_received or cfg.use_simulated_olfaction)]
            if self.uses_camera:
                needed += [("RGB", self.latest_rgb is not None), ("depth", self.latest_depth is not None)]
            missing = [name for name, ok in needed if not ok]
            if missing:
                self.get_logger().warn(f"Waiting for: {', '.join(missing)}", throttle_duration_sec=2.0)
            elif self.setup_search(pose):
                self.start_window()
                self.get_logger().info(f"Sampling step 1 for {cfg.sample_period:.1f}s...")
            return

        try:
            self.run_step(pose)
        except Exception:
            # Keep the run (and its data) alive if one step fails
            self.get_logger().error(f"Step {self.step_count} failed:\n{traceback.format_exc()}")
        self.start_window()

        if self.step_count >= cfg.step_threshold:
            self.get_logger().info(f"Step threshold {cfg.step_threshold} reached. Ending run.")
            self.done = True

    def start_window(self):
        """Starts the next step window: only /olfaction readings from now on count for the next step."""
        self.window_start = self.now()
        self.window_olf_idx = len(self.olfaction_raw)

    def read_olfaction(self, robot_x, robot_y):
        """Mean of the /olfaction readings of the current window, rescaled with the configured min / max."""
        cfg = self.cfg
        if cfg.use_simulated_olfaction and cfg.source_position is not None:
            # Testing only: noisy reading from the plume model
            conc = gaussian_plume(robot_x, robot_y, tuple(cfg.source_position),
                                  q_s=cfg.q_s, D=cfg.D, U=cfg.U, tau=cfg.tau, psi_deg=cfg.psi_deg)
            conc += np.random.normal(0, cfg.sigma_noise)
            return dict(raw=conc, concentration=conc, raw_std=0.0, n_readings=1,
                        wind_direction=0.0, wind_speed=0.0)

        window = self.olfaction_raw[self.window_olf_idx:]
        if window:
            conc = np.array([r["chemicalConc"] for r in window], dtype=float)
            raw, raw_std = float(conc.mean()), float(conc.std())
            wind_dir = float(np.mean([r["wind_direction"] for r in window]))
            wind_speed = float(np.mean([r["wind_speed"] for r in window]))
        else:
            self.get_logger().warn("No /olfaction readings in this step window; using the latest reading.")
            last = self.olfaction_raw[-1]
            raw, raw_std, wind_dir, wind_speed = last["chemicalConc"], 0.0, last["wind_direction"], last["wind_speed"]

        # Same linear rescaling as the offline mapping: min -> 0, max -> olfaction_rescale_to
        concentration = raw
        if cfg.olfaction_min_conc is not None:
            scale = cfg.olfaction_rescale_to / (cfg.olfaction_max_conc - cfg.olfaction_min_conc)
            concentration = scale * (raw - cfg.olfaction_min_conc)

        return dict(raw=raw, concentration=concentration, raw_std=raw_std, n_readings=len(window),
                    wind_direction=wind_dir, wind_speed=wind_speed)

    def run_step(self, pose):
        """End of a window: olfactory update, fusion, navigation, saving."""
        cfg, g = self.cfg, self.grid
        step = self.step_count
        step_start_time = time.time()
        robot_x, robot_z, robot_yaw = pose     # robot_z = map y (ai2thor naming); yaw: ROS, rad
        self.get_logger().info(f"===== Step {step + 1}/{cfg.step_threshold} at ({robot_x:.2f}, {robot_z:.2f}) =====")

        # --- 1. Olfaction: update the Bayesian map with the window's mean reading ---
        olf = self.read_olfaction(robot_x, robot_z)
        self.bayesian_agent.posterior(olf["concentration"], robot_x, robot_z)
        srcProbGivenOlfactory = self.bayesian_agent.prob_map
        olfactoryEntropy = map_entropy(srcProbGivenOlfactory)
        olf_idx = np.unravel_index(np.argmax(srcProbGivenOlfactory), srcProbGivenOlfactory.shape)
        olfactory_max_xz = grid_to_world(olf_idx, g.x_points, g.z_points)   # most likely source cell

        # --- 2. Source estimate: fused with vision ('F') or the olfactory maximum ('O') ---
        if self.vision is not None:
            estimate = self.vision.fuse(srcProbGivenOlfactory)
        else:
            x, y = olfactory_max_xz
            estimate = StepEstimate(target_object='N/A', target_coordinate=format_position(x, y, 0.0),
                                    target_xz=np.array([x, y]))

        # Behaviour flag, as in fusion_controller.py: search until the olfactory map is confident enough
        if step == 0:
            behavior_flag = "Initialization"
        elif olfactoryEntropy > self.entropy_threshold:
            behavior_flag = "search"
        else:
            behavior_flag = "goal_navigation"

        # --- 3. Navigation (does nothing yet) ---
        self.navigate(behavior_flag, estimate.target_xz, olfactory_max_xz, pose)

        # --- 4. Evaluation, only if the true source position is given ---
        gt_distance = target_error = np.nan
        if cfg.source_position is not None:
            src = np.asarray(cfg.source_position, dtype=float)
            gt_distance = float(np.linalg.norm(np.array([robot_x, robot_z]) - src))
            if estimate.target_xz is not None:
                target_error = float(np.linalg.norm(src - np.asarray(estimate.target_xz, dtype=float)))
        step_time = time.time() - step_start_time

        # --- 5. Save maps and per-step files ---
        tag = f"{step:03d}_x_{robot_x:.2f}_z_{robot_z:.2f}"
        panels = [(srcProbGivenOlfactory, rf'$H_C={olfactoryEntropy:.2f}$')] + estimate.panels
        try:
            trajectory = [(r["robot_x"], r["robot_z"]) for r in self.trajectory_log_list] + [(robot_x, robot_z)]
            save_belief_maps(panels, g.x_points, g.z_points, os.path.join(self.save_dir, f"maps_all_{tag}.png"),
                             trajectory=trajectory, outline=g.area_polygon)
        except Exception as e:
            self.get_logger().error(f"Error saving belief map plot at step {step}: {e}")
        np.savez_compressed(os.path.join(self.save_dir, f"maps_{step:03d}.npz"), olfactory=srcProbGivenOlfactory,
                            x_points=g.x_points, z_points=g.z_points, **estimate.arrays)
        if self.vision is not None:
            try:
                self.vision.save_step(step, tag, (robot_x, robot_z))
            except Exception as e:
                self.get_logger().error(f"Error saving vision outputs at step {step}: {e}")

        # --- 6. Trajectory log (same columns as the AI2-THOR runs, plus TB4 extras) ---
        self.trajectory_log_list.append({
            "step": step,
            "time": round(self.now(), 3),
            "robot_x": robot_x,
            "robot_z": robot_z,
            "robot_yaw": ros_yaw_to_ai2thor_deg(robot_yaw),   # ai2thor convention (deg, CW from map +y)
            "robot_yaw_ros_deg": math.degrees(robot_yaw),     # ROS convention (deg, CCW from map +x)
            "step_time": step_time,
            "behavior_flag": behavior_flag,
            "is_random": False,
            "target_object": estimate.target_object,
            "target_coordinate": estimate.target_coordinate,
            "target_coord_estimation_error": target_error,
            "olfactory_max_x": float(olfactory_max_xz[0]),
            "olfactory_max_z": float(olfactory_max_xz[1]),
            "concentration": olf["concentration"],            # rescaled window mean, used by the update
            "raw_concentration": olf["raw"],                  # raw window mean
            "raw_concentration_std": olf["raw_std"],
            "olfaction_readings": olf["n_readings"],
            "window_start": round(self.window_start, 3),
            "wind_direction": olf["wind_direction"],
            "wind_speed": olf["wind_speed"],
            "gt_distance_from_source": gt_distance,
            "Bayesian_entropy": olfactoryEntropy,
            "visual_entropy": estimate.visual_entropy,
            "fused_entropy": estimate.fused_entropy,
            "entropy_threshold": self.entropy_threshold,
            **estimate.log,
        })
        # Rewritten every step, so the data survives a crash
        pd.DataFrame(self.trajectory_log_list).to_csv(os.path.join(self.save_dir, "trajectory_log.csv"), index=False)

        self.get_logger().info(
            f"conc={olf['concentration']:.2f} (raw mean {olf['raw']:.2f} of {olf['n_readings']}) "
            f"H_C={olfactoryEntropy:.2f} (thr {self.entropy_threshold:.2f}) flag={behavior_flag} "
            f"target={estimate.target_object} @ {estimate.target_coordinate} "
            f"olf_max=({olfactory_max_xz[0]:.2f}, {olfactory_max_xz[1]:.2f}) step_time={step_time:.2f}s")
        self.step_count += 1

    # ==================================================================
    # Vision: one camera frame at a time during the window ('F' only)
    # ==================================================================

    def vision_callback(self):
        """Gives the newest camera frame (if not processed yet) to Vision.process_frame."""
        if self.done or self.window_start is None:
            return    # not set up yet, or run over
        if self.latest_rgb is None or self.latest_depth is None or self.rgb_seq == self._last_processed_seq:
            return    # no new frame
        pose = self.get_robot_pose(warn=False)
        if pose is None:
            return
        self._last_processed_seq = self.rgb_seq
        try:
            rgb, depth = self.latest_rgb.copy(), self.latest_depth.copy()
            self.latest_annotated = self.vision.process_frame(
                rgb, depth, K=self.camera_intrinsics(rgb.shape), optical_to_map=self.optical_to_map_fn(pose),
                step=self.step_count, robot_xy=pose[:2], t=round(self.now(), 3))
        except Exception as e:
            self.get_logger().error(f"Vision frame failed: {e}")

    # ==================================================================
    # Navigation (to be implemented)
    # ==================================================================

    def navigate(self, behavior_flag, target_xz, olfactory_max_xz, pose):
        """Moves the robot toward the current source estimate. Not implemented: drive with teleop.

        Parameters
        ----------
        behavior_flag : "Initialization", "search" or "goal_navigation" (fusion_controller.py semantics)
        target_xz : map (x, y) of the current estimate (fused maximum for 'F', olfactory maximum for 'O')
        olfactory_max_xz : map (x, y) of the most likely cell of the olfactory map
        pose : robot (x, y, yaw) in the map frame, yaw in the ROS convention (rad, CCW from +x)

        Headings sent to the robot must use the ROS convention, i.e.
        sOSL_tb4Functions.ros_heading_to(), not fusion_controller's atan2(dx, dz)
        (Unity, clockwise from +z).
        """
        # TODO: send a Nav2 goal / publish on self.cmd_vel_pub toward target_xz
        return None

    # ==================================================================
    # Shutdown
    # ==================================================================

    def ui_update(self):
        """Shows the latest YOLO image; 'q' ends the run. Call from the main thread."""
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
        except cv2.error as e:     # e.g. no display: keep running without a window
            self.get_logger().warn(f"Disabling OpenCV window: {e}")
            self.show_window = False

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

        if self.vision is not None:
            try:
                self.vision.save_final()
            except Exception as e:
                print(f"Error saving final vision outputs: {e}")

        # Trajectory over the map, with the final estimate (and the true source if known)
        if self.grid is not None and self.trajectory_log_list:
            last = self.trajectory_log_list[-1]
            estimate_xz = None
            if last["target_coordinate"] not in ('N/A', None):
                p = parse_position_string(last["target_coordinate"])
                estimate_xz = (p[0], p[2])
            cfg = self.cfg
            plot_tb4_trajectory(
                os.path.join(self.save_dir, "trajectory_log.csv"), self.grid,
                os.path.join(self.save_dir, "trajectory_plot.png"),
                source_xz=cfg.source_position, estimate_xz=estimate_xz,
                plume_params=dict(q_s=cfg.q_s, D=cfg.D, U=cfg.U, tau=cfg.tau, psi_deg=cfg.psi_deg))


def spin_with_ui(node):
    """Runs the node until the run ends, then writes the final files.

    The node is spun by a MultiThreadedExecutor in a background thread (so sensor
    callbacks and processing run in parallel); this thread only refreshes the
    OpenCV window. Ctrl+C, 'q', step_threshold and run_time_limit all end here.
    """
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:   # Ctrl+C shuts the ROS context down
            pass

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok() and not node.done and spin_thread.is_alive():
            node.ui_update()
            time.sleep(0.03)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown(timeout_sec=30.0)   # lets a running step / frame finish
        node.finalize()
        node.destroy_node()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if rclpy.ok():
            rclpy.shutdown()
