"""Fusion ('F') TurtleBot4 controller: olfaction + Dirichlet object map.

On top of the olfactory update in TB4BaseController, YOLO runs continuously
during each step window: every new RGB + depth frame (at most one per
`vision_period` s) is detected, projected with the robot pose at that moment
and added to the Dirichlet object map (sOSL_dirichletFunctions.py):
each detection adds confidence * (confusion-matrix row of its class) over a
footprint set by the bounding box (same-class overlaps are max-merged), and
every visible cell without a detection adds distance-decayed Background evidence.

At the end of the step:

    P(src | V)  ∝  sum_k p(o_k | z) * max(0, sim(class_k, goal))
    P(src | C, V)  ∝  P(src | C) * P(src | V)

both 0 outside the search boundary. The source estimate is the fused arg-max.
"""

import math
import os

import cv2
import numpy as np
import pandas as pd

from sOSL_dirichletFunctions import DirichletObjectMap, camera_pose_in_map, confusion_matrix_from_normalized
from sOSL_olfactionFunctions import map_entropy
from sOSL_tb4Functions import detect_objects
from tb4_base_controller import TB4BaseController


class TB4FusionController(TB4BaseController):

    uses_camera = True

    def __init__(self, cfg, yolo_model, save_dir):
        if cfg.dirichlet_conf_matrix is None or cfg.dirichlet_conf_labels is None:
            raise ValueError("The fusion controller needs dirichlet_conf_matrix and dirichlet_conf_labels.")
        self.yolo_model = yolo_model
        self.classes = [n for n in yolo_model.names.values() if n not in cfg.yolo_exclude_classes]
        self.confusion_matrix = confusion_matrix_from_normalized(
            self.classes, cfg.dirichlet_conf_matrix, cfg.dirichlet_conf_labels, bg_fp_rate=cfg.dirichlet_bg_fp_rate)
        self.eta = cfg.dirichlet_bg_false_neg_rate
        if self.eta is None:
            self.eta = float(self.confusion_matrix[-1, :-1].mean())   # mean miss rate from the matrix
        self.object_map = None
        self.detection_log = []
        self._last_frame = None           # (rgb, depth) of the last processed frame
        self._last_seq = 0
        self.frame_count = 0              # frames processed in the whole run
        self.step_frames = 0              # frames processed in the current step window
        self.step_detections = 0
        super().__init__(cfg, save_dir, node_name="sosl_tb4_fusion_controller")
        self.vision_timer = self.create_timer(cfg.vision_period, self.vision_callback,
                                              callback_group=self.processing_group)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_perception(self):
        from sentence_transformers import SentenceTransformer
        cfg, g = self.cfg, self.grid
        self.object_map = DirichletObjectMap(
            g.x_points, g.z_points, self.classes,
            confusion_matrix=self.confusion_matrix,
            conf_temper=cfg.dirichlet_conf_temper,
            bg_false_neg_rate=self.eta,
            bg_dist_decay=cfg.dirichlet_bg_dist_decay,
            prior_strength=cfg.dirichlet_prior_strength,
            min_radius=cfg.dirichlet_min_radius,
            max_radius=cfg.dirichlet_max_radius,
            class_colors=cfg.object_colors,
            gt_positions=cfg.object_positions,
        )
        table = self.object_map.set_class_similarity(
            SentenceTransformer(cfg.sentence_model), self.goal_phrase,
            background_similarity=cfg.dirichlet_background_similarity)
        self.get_logger().info(
            f"Dirichlet object map: {self.object_map.K} classes (incl. Background), "
            f"eta={self.eta:.3f}. Class similarities to '{self.goal_phrase}':\n{table.to_string(index=False)}")

    def run_info_extra(self):
        return {"yolo_classes": dict(self.yolo_model.names),
                "dirichlet_classes": self.object_map.classes,
                "dirichlet_confusion_matrix": self.confusion_matrix.tolist(),
                "dirichlet_bg_false_neg_rate": self.eta}

    # ------------------------------------------------------------------
    # Per-frame vision (runs during the step window)
    # ------------------------------------------------------------------

    def vision_callback(self):
        """Processes the newest camera frame and adds its evidence to the object map."""
        if self.done or self.object_map is None or self.window_start is None:
            return
        if self.latest_rgb is None or self.latest_depth is None or self.rgb_seq == self._last_seq:
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
        cfg = self.cfg
        rgb, depth = self.latest_rgb.copy(), self.latest_depth.copy()
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

        om, inside = self.object_map, self.grid.inside
        srcProbGivenVision = om.source_prob_given_vision(inside)
        fused = srcProbGivenOlfactory * srcProbGivenVision
        total = fused.sum()
        fused = fused / total if total > 1e-300 else inside / inside.sum()

        target_xz, (row, col) = self.argmax_xz(fused)
        hv, hf = map_entropy(srcProbGivenVision), map_entropy(fused)
        return dict(
            panels=[(srcProbGivenVision, rf'$H_V={hv:.2f}$'), (fused, rf'$H_F={hf:.2f}$')],
            arrays=dict(visual=srcProbGivenVision, fused=fused, beta=om.beta.astype(np.float32),
                        observed=om.observed, mle_class=om.mle_class()),
            target_object=om.top_object_at(row, col),
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
                        np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
        self.object_map.plot_mle(self.out_path("object_map", f"object_map_{step_count:03d}.png"),
                                 title=f"Object map (MLE), step {step_count}", robot_xz=pose[:2])

    def finalize_perception(self):
        pd.DataFrame(self.detection_log).to_csv(os.path.join(self.save_dir, "detections_log.csv"), index=False)
        if self.object_map is not None:
            self.object_map.save_final(self.save_dir)

    def out_path(self, folder, filename):
        d = os.path.join(self.save_dir, folder)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, filename)
