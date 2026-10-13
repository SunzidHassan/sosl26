"""Fusion ('F') TurtleBot4 controller: olfaction + Dirichlet object map.

On top of the olfactory controller (TB4BaseController), YOLO runs throughout
each step window: every new RGB + depth frame (at most one per
`vision_period` s) is detected, projected into the map with the robot pose at
that moment, and added to the Dirichlet object map
(sOSL_dirichletFunctions.py). At the end of the step:

    P(src | V)     ∝  sum_k p(o_k | z) * max(0, sim(class_k, goal))
    P(src | C, V)  ∝  P(src | C) * P(src | V)

both 0 outside the search boundary. The source estimate is the fused arg-max.
"""

import math
import os

import cv2
import numpy as np
import pandas as pd
import rclpy
from sensor_msgs.msg import CameraInfo, CompressedImage
from tf2_ros import TransformException

from sOSL_dirichletFunctions import DirichletObjectMap, confusion_matrix_from_normalized
from sOSL_olfactionFunctions import map_entropy
from sOSL_tb4Functions import (
    camera_pose_in_map,
    decode_compressed_depth,
    decode_compressed_rgb,
    detect_objects,
    intrinsics_from_fov,
    optical_to_map_from_robot_pose,
    scale_intrinsics,
    transform_point,
)
from tb4_base_controller import TB4BaseController


class TB4FusionController(TB4BaseController):

    def __init__(self, cfg, save_dir):
        # Heavy imports here, so olfactory-only runs don't need them
        from sentence_transformers import SentenceTransformer
        from ultralytics import YOLO

        super().__init__(cfg, save_dir, node_name="sosl_tb4_fusion_controller")
        if cfg.dirichlet_conf_matrix is None or cfg.dirichlet_conf_labels is None:
            raise ValueError("The fusion controller needs dirichlet_conf_matrix and dirichlet_conf_labels.")

        # --- Detector and object map ---
        self.yolo_model = YOLO(cfg.yolo_model_path)
        classes = [n for n in self.yolo_model.names.values() if n not in cfg.yolo_exclude_classes]
        self.object_map = DirichletObjectMap(
            self.grid, classes,
            confusion_matrix=confusion_matrix_from_normalized(
                classes, cfg.dirichlet_conf_matrix, cfg.dirichlet_conf_labels, bg_fp_rate=cfg.dirichlet_bg_fp_rate),
            conf_temper=cfg.dirichlet_conf_temper,
            bg_false_neg_rate=cfg.dirichlet_bg_false_neg_rate,
            bg_dist_decay=cfg.dirichlet_bg_dist_decay,
            prior_strength=cfg.dirichlet_prior_strength,
            min_radius=cfg.dirichlet_min_radius,
            max_radius=cfg.dirichlet_max_radius,
            class_colors=cfg.object_colors,
            gt_positions=cfg.object_positions,
        )
        # Class <-> goal similarity is fixed for the run, so the sentence model is only used here.
        table = self.object_map.set_class_similarity(
            SentenceTransformer(cfg.sentence_model), self.goal_phrase,
            background_similarity=cfg.dirichlet_background_similarity)
        self.get_logger().info(
            f"Dirichlet object map: {self.object_map.K} classes (incl. Background), "
            f"eta={self.object_map.bg_false_neg_rate:.3f}. Class similarities to '{self.goal_phrase}':\n"
            f"{table.to_string(index=False)}")

        # --- Camera state (written by the sensor callbacks) ---
        # The callbacks only keep the newest *compressed* message (RGB 10 Hz, depth 30 Hz);
        # process_frame() decodes the pair it uses, so ~2 frames/s are decoded instead of ~40.
        self.latest_rgb_msg = None
        self.rgb_seq = 0             # incremented per RGB message, to skip frames already processed
        self.latest_depth_msg = None
        self.cam_K = None            # (fx, fy, cx, cy) from CameraInfo
        self.cam_info_size = (0, 0)  # (width, height) of CameraInfo
        self.subscribe(CompressedImage, cfg.rgb_topic, self.image_callback)
        self.subscribe(CompressedImage, cfg.depth_topic, self.depth_callback)
        self.subscribe(CameraInfo, cfg.camera_info_topic, self.camera_info_callback)

        # --- Vision state ---
        self.latest_annotated = None      # shown by ui_update()
        self.rgb_frame_id = ""            # camera frame of the last processed image (for the TF lookup)
        self.show_window = cfg.show_window
        self.detection_log = []           # every detection, for detections_log.csv
        self._last_frame = None           # (rgb, depth) of the last processed frame
        self._last_seq = 0
        self.frame_count = 0              # frames processed in the whole run
        self.step_frames = 0              # frames / detections in the current step window
        self.step_detections = 0
        # Same callback group as the step timer, so a frame never runs during a step.
        self.vision_timer = self.create_timer(cfg.vision_period, self.vision_callback,
                                              callback_group=self.processing_group)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        self.latest_rgb_msg = msg     # decoded later, only if process_frame() uses it
        self.rgb_seq += 1

    def depth_callback(self, msg: CompressedImage):
        self.latest_depth_msg = msg

    def camera_info_callback(self, msg: CameraInfo):
        k = msg.k   # row-major 3x3: fx at 0, cx at 2, fy at 4, cy at 5
        if k[0] > 0:
            self.cam_K = (k[0], k[4], k[2], k[5])
            self.cam_info_size = (msg.width, msg.height)

    def missing_inputs(self):
        return super().missing_inputs() + [name for name, msg in (("RGB", self.latest_rgb_msg),
                                                                  ("depth", self.latest_depth_msg)) if msg is None]

    def camera_intrinsics(self, rgb_shape):
        """(fx, fy, cx, cy) for the RGB image: from CameraInfo, else from camera_hfov_deg."""
        h, w = rgb_shape[:2]
        if self.cam_K is not None:
            return scale_intrinsics(self.cam_K, self.cam_info_size, (w, h))
        self.get_logger().warn("No CameraInfo yet, using HFOV intrinsics.", throttle_duration_sec=10.0)
        return intrinsics_from_fov(w, h, self.cfg.camera_hfov_deg)

    def optical_to_map_fn(self, pose):
        """Function mapping camera optical-frame points into the map frame: the TF
        map -> camera if available, else the robot pose + camera_height."""
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

    def ui_update(self):
        """Shows the latest annotated frame; runs in the main thread (OpenCV GUI is not thread-safe)."""
        if not self.show_window:
            return
        frame = self.latest_annotated   # blank until the first frame is processed
        if frame is None:
            return
        try:
            cv2.imshow("sOSL TB4 - YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("'q' pressed. Ending run.")
                self.done = True
        except cv2.error as e:   # e.g. no display
            self.get_logger().warn(f"Disabling OpenCV window: {e}")
            self.show_window = False

    # ------------------------------------------------------------------
    # Per-frame vision (runs during the step window)
    # ------------------------------------------------------------------

    def vision_callback(self):
        """Processes the newest camera frame (if not processed yet) once the first window has started."""
        if self.done or self.window_start is None:
            return
        if self.latest_rgb_msg is None or self.latest_depth_msg is None or self.rgb_seq == self._last_seq:
            return
        pose = self.get_robot_pose(warn=False)
        if pose is None:
            return
        self._last_seq = self.rgb_seq
        try:
            self.process_frame(pose)
        except Exception as e:
            self.get_logger().error(f"Vision frame {self.frame_count} failed: {e}")

    def process_frame(self, pose):
        """YOLO -> map-frame detections -> Dirichlet evidence for one frame."""
        cfg = self.cfg
        # Take the newest messages (the callbacks may replace them meanwhile) and decode only these two
        rgb_msg, depth_msg = self.latest_rgb_msg, self.latest_depth_msg
        rgb = decode_compressed_rgb(rgb_msg.data)
        depth = decode_compressed_depth(depth_msg.data, depth_msg.format)
        if rgb is None or depth is None:
            self.get_logger().error(f"Failed to decode the {'RGB' if rgb is None else 'depth'} image "
                                    f"(format '{(rgb_msg if rgb is None else depth_msg).format}')",
                                    throttle_duration_sec=5.0)
            return
        self.rgb_frame_id = rgb_msg.header.frame_id
        self._last_frame = (rgb, depth)
        K = self.camera_intrinsics(rgb.shape)
        optical_to_map = self.optical_to_map_fn(pose)

        annotated, detections = detect_objects(
            self.yolo_model, rgb, depth, K, optical_to_map,
            conf_thr=cfg.yolo_conf,
            exclude_names=cfg.yolo_exclude_classes,
            depth_percentile=cfg.depth_percentile,
            depth_range=cfg.depth_range,
            logger=self.get_logger().debug,
        )
        self.latest_annotated = annotated

        # Camera position / heading and field of view decide which cells were seen
        cam_xz, heading = camera_pose_in_map(optical_to_map)
        hfov = 2.0 * math.atan(rgb.shape[1] / (2.0 * K[0]))
        self.object_map.update(detections, cam_xz, heading, hfov, max_range=cfg.depth_range[1])

        t = round(self.now(), 3)
        self.detection_log += [dict(step=self.step_count, frame=self.frame_count, time=t,
                                    robot_x=pose[0], robot_z=pose[1], **d) for d in detections]
        if cfg.save_vision_frames:
            cv2.imwrite(self.out_path("yolo_detections", f"yolo_{self.step_count:03d}_{self.frame_count:04d}.jpg"),
                        annotated)
        self.frame_count += 1
        self.step_frames += 1
        self.step_detections += len(detections)

    # ------------------------------------------------------------------
    # Per-step fusion
    # ------------------------------------------------------------------

    def perceive(self, pose, step_count, srcProbGivenOlfactory):
        frames, dets = self.step_frames, self.step_detections
        self.step_frames = self.step_detections = 0
        self.get_logger().info(f"Vision this step: {frames} frames, {dets} detections")
        if frames == 0:
            self.get_logger().warn("No camera frames were processed in this step window.")

        # P(src | C, V) ∝ P(src | C) * P(src | V); both are already 0 outside the boundary
        om, inside = self.object_map, self.grid.inside
        srcProbGivenVision = om.source_prob_given_vision()
        fused = srcProbGivenOlfactory * srcProbGivenVision
        total = fused.sum()
        fused = fused / total if total > 1e-300 else inside / inside.sum()

        target_xz, (row, col) = self.grid.argmax_xz(fused)
        hv, hf = map_entropy(srcProbGivenVision), map_entropy(fused)
        return dict(
            panels=[(srcProbGivenVision, rf'$H_V={hv:.2f}$'), (fused, rf'$H_F={hf:.2f}$')],
            arrays=dict(visual=srcProbGivenVision, fused=fused, beta=om.beta.astype(np.float32),
                        observed=om.observed, mle_class=om.mle_class()),
            target_object=om.top_object_at(row, col),   # most likely object class in the estimated cell
            target_xz=target_xz,
            visual_entropy=hv,
            fused_entropy=hf,
            log=dict(vision_frames=frames, vision_detections=dets),
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_perception_outputs(self, step_count, tag, pose):
        if self._last_frame is not None:   # last processed frame of the step
            rgb, depth = self._last_frame
            cv2.imwrite(self.out_path("frames", f"frame_{tag}.png"), rgb)
            cv2.imwrite(self.out_path("depth", f"depth_{step_count:03d}.png"),
                        np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))   # 16-bit PNG in mm
        self.object_map.plot_mle(self.out_path("object_map", f"object_map_{step_count:03d}.png"),
                                 title=f"Object map (MLE), step {step_count}", robot_xz=pose[:2])

    def finalize_perception(self):
        pd.DataFrame(self.detection_log).to_csv(os.path.join(self.save_dir, "detections_log.csv"), index=False)
        self.object_map.save_final(self.save_dir)

    def out_path(self, folder, filename):
        """save_dir/folder/filename, creating the folder if needed."""
        d = os.path.join(self.save_dir, folder)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, filename)
