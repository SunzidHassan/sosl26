"""Fusion ('F') TurtleBot4 controller: olfaction + vision.

TB4 counterpart of ControlAlgorithms/Fusion/fusion_controller.py. On top of
the olfactory update in TB4BaseController, YOLO runs continuously during each
step window: every new RGB + depth frame (at most one per `vision_period` s)
is detected, projected with the robot pose at that moment and accumulated,
so a step collects evidence fromfrom sOSL_dirichletFunctions import DirichletObjectMap, camera_pose_in_map, confusion_matrix_from_normalized many frames instead of one. At the end of the
step, vision is fused with olfaction using one of two vision approaches
(ExperimentConfig.vision_mode):

'navKnowledge' (AI2-THOR approach, as is)
    Detections are merged into an object list (envKnowledge);
    add_goal_similarity scores each object with langSim * olfactionSim
    (navKnowledge). Visual map = langSim heatmap, fused map = goalSim heatmap,
    target = top navKnowledge object.

'dirichlet' (object distribution, sOSL_dirichletFunctions.py)
    Every processed frame adds Dirichlet evidence to a per-cell class distribution:
    each detection adds confidence * (confusion-matrix row of its class) over a
    footprint set by the bounding box (same-class overlaps are max-merged), and
    every visible cell without a detection adds distance-decayed Background evidence.

Both modes use the same detections, and envKnowledge is kept in both so the
detected-objects map is always available.
"""

import math
import os

import cv2
import numpy as np
import pandas as pd

from sOSL_dirichletFunctions import DirichletObjectMap, camera_pose_in_map, confusion_matrix_from_normalized
from sOSL_loggerFunctions import map_entropy, plot_detected_objects
from sOSL_tb4Functions import compute_maps, format_position, visionBranch
from sOSL_utils import grid_to_world, parse_position_string
from sosl26.tb4_base_controller import ExperimentConfig, TB4BaseController  # noqa: F401 (re-export)

VISION_MODES = ('navKnowledge', 'dirichlet')


class TB4FusionController(TB4BaseController):

    uses_camera = True

    def __init__(self, cfg, yolo_model, save_dir):
        if cfg.vision_mode not in VISION_MODES:
            raise ValueError(f"vision_mode must be one of {VISION_MODES}, got '{cfg.vision_mode}'")
        # Imported here so olfactory-only runs don't load the sentence transformer.
        import sOSL_visionFunctions
        self._vision = sOSL_visionFunctions

        self.yolo_model = yolo_model
        self.envKnowledge = pd.DataFrame()
        self.navKnowledge = pd.DataFrame()
        self.object_map = None
        self.detection_log = []
        self._frames = (None, None)
        self._last_seq = 0
        self.frame_count = 0              # frames processed in the whole run
        self.step_frames = 0              # frames processed in the current step window
        self.step_detections = 0
        super().__init__(cfg, save_dir, node_name="sosl_tb4_fusion_controller")
        if cfg.vision_mode == 'dirichlet':
            self.confusion_matrix, self.dirichlet_eta = self.build_confusion_matrix()
        self.vision_timer = self.create_timer(cfg.vision_period, self.vision_callback,
                                              callback_group=self.processing_group)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def dirichlet_classes(self):
        """Object classes of the Dirichlet map: model classes after target / exclude filters."""
        names = list(self.yolo_model.names.values())
        if self.cfg.yolo_target_classes:
            names = [n for n in names if n in self.cfg.yolo_target_classes]
        return [n for n in names if n not in self.cfg.yolo_exclude_classes]

    def build_confusion_matrix(self):
        cfg = self.cfg
        if cfg.dirichlet_conf_matrix is None or cfg.dirichlet_conf_labels is None:
            raise ValueError("vision_mode 'dirichlet' needs dirichlet_conf_matrix and dirichlet_conf_labels.")
        C = confusion_matrix_from_normalized(self.dirichlet_classes(), cfg.dirichlet_conf_matrix,
                                             cfg.dirichlet_conf_labels, bg_fp_rate=cfg.dirichlet_bg_fp_rate)
        eta = cfg.dirichlet_bg_false_neg_rate
        if eta is None:
            eta = float(C[-1, :-1].mean())      # mean miss rate from the matrix (~0.03)
        return C, eta

    def setup_perception(self):
        if self.cfg.vision_mode != 'dirichlet':
            return
        cfg, g = self.cfg, self.grid
        self.object_map = DirichletObjectMap(
            g.x_points, g.z_points, self.dirichlet_classes(),
            confusion_matrix=self.confusion_matrix,
            conf_temper=cfg.dirichlet_conf_temper,
            bg_false_neg_rate=self.dirichlet_eta,
            bg_dist_decay=cfg.dirichlet_bg_dist_decay,
            prior_strength=cfg.dirichlet_prior_strength,
            min_radius=cfg.dirichlet_min_radius,
            max_radius=cfg.dirichlet_max_radius,
            class_colors=cfg.object_colors,
            gt_positions=cfg.object_positions,
        )
        table = self.object_map.set_class_similarity(
            self._vision.model, self.goal_phrase,
            background_similarity=cfg.dirichlet_background_similarity)
        self.get_logger().info(
            f"Dirichlet object map: {self.object_map.K} classes (incl. Background), "
            f"eta={self.dirichlet_eta:.3f}. Top class similarities to '{self.goal_phrase}':\n"
            f"{table.head(8).to_string(index=False)}")

    def run_info_extra(self):
        info = {"yolo_classes": dict(getattr(self.yolo_model, "names", {}))}
        if self.object_map is not None:
            info["dirichlet_classes"] = self.object_map.classes
            info["dirichlet_confusion_matrix"] = self.object_map.confusion_matrix.tolist()
            info["dirichlet_bg_false_neg_rate"] = self.dirichlet_eta
        return info

    # ------------------------------------------------------------------
    # Per-step perception
    # ------------------------------------------------------------------

    def vision_callback(self):
        """Processes the newest camera frame and accumulates its evidence (runs during the step window)."""
        if self.done or self.grid is None or self.window_start is None:
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
        rgb = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        self._frames = (rgb, depth)
        K = self.camera_intrinsics(rgb.shape)
        optical_to_map = self.optical_to_map_fn(pose)

        self.envKnowledge, annotated, detections = visionBranch(
            self.yolo_model, self.envKnowledge, rgb, depth,
            K=K,
            optical_to_map=optical_to_map,
            save_dir=None,
            confThr=cfg.yolo_conf,
            target_names=cfg.yolo_target_classes,
            exclude_names=cfg.yolo_exclude_classes,
            depth_percentile=cfg.depth_percentile,
            depth_range=cfg.depth_range,
            merge_dist=cfg.merge_dist,
            logger=self.get_logger().debug,
        )
        self.latest_annotated = annotated

        if self.object_map is not None:
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

    def perceive(self, pose, step_count, srcProbGivenOlfactory):
        frames, dets = self.step_frames, self.step_detections
        self.step_frames = self.step_detections = 0
        self.get_logger().info(f"Vision this step: {frames} frames, {dets} detections "
                               f"({len(self.envKnowledge)} objects in envKnowledge)")
        if frames == 0:
            self.get_logger().warn("No camera frames were processed in this step window.")

        if self.cfg.vision_mode == 'navKnowledge':
            result = self._perceive_navknowledge(srcProbGivenOlfactory)
        else:
            result = self._perceive_dirichlet(srcProbGivenOlfactory)
        result["log"] = dict(vision_frames=frames, vision_detections=dets)
        return result

    def _perceive_navknowledge(self, srcProbGivenOlfactory):
        g = self.grid
        self.navKnowledge = self._vision.add_goal_similarity(
            self.envKnowledge.copy(), self.goal_phrase, srcProbGivenOlfactory,
            g.x_points, g.z_points, alg_choice='F')
        if not self.navKnowledge.empty:
            self.get_logger().info(f"NavKnowledge:\n{self.navKnowledge.head(4).to_string()}")

        _, vision_raw, goal_raw = compute_maps(srcProbGivenOlfactory, self.navKnowledge, g.x_points, g.z_points)

        target_object, target_coordinate, target_xz = 'N/A', 'N/A', None
        if not self.navKnowledge.empty:
            top = self.navKnowledge.iloc[0]
            target_object, target_coordinate = top["objectType"], top["Position"]
            pred_pos = parse_position_string(target_coordinate)
            target_xz = np.array([pred_pos[0], pred_pos[2]])

        hv, hf = map_entropy(vision_raw), map_entropy(goal_raw)
        return dict(
            panels=[(vision_raw, rf'$H_V={hv:.2f}$'), (goal_raw, rf'$H_F={hf:.2f}$')],
            arrays=dict(visual=vision_raw, fused=goal_raw),
            target_object=target_object,
            target_coordinate=target_coordinate,
            target_xz=target_xz,
            visual_entropy=hv,
            fused_entropy=hf,
        )

    def _perceive_dirichlet(self, srcProbGivenOlfactory):
        g, om = self.grid, self.object_map
        srcProbGivenVision = om.source_prob_given_vision()
        fused = srcProbGivenOlfactory * srcProbGivenVision
        total = fused.sum()
        fused = fused / total if total > 1e-300 else np.full_like(fused, 1.0 / fused.size)

        row, col = np.unravel_index(np.argmax(fused), fused.shape)
        x, z = grid_to_world((row, col), g.x_points, g.z_points)

        hv, hf = map_entropy(srcProbGivenVision), map_entropy(fused)
        return dict(
            panels=[(srcProbGivenVision, rf'$H_V={hv:.2f}$'), (fused, rf'$H_F={hf:.2f}$')],
            arrays=dict(visual=srcProbGivenVision, fused=fused, beta=om.beta.astype(np.float32),
                        observed=om.observed, mle_class=om.mle_class()),
            target_object=om.top_object_at(row, col),
            target_coordinate=format_position(x, z, 0.0),
            target_xz=np.array([x, z]),
            visual_entropy=hv,
            fused_entropy=hf,
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_perception_outputs(self, step_count, tag):
        g, cfg = self.grid, self.cfg
        rgb, depth = self._frames   # last processed frame of the step
        if rgb is not None:
            cv2.imwrite(self.out_path("frames", f"frame_{tag}.png"), rgb)
            cv2.imwrite(self.out_path("depth", f"depth_{step_count:03d}.png"),
                        np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))

        plot_detected_objects(itemDF=self.envKnowledge, mask_closed=g.mask, scene_bounds_tuple=g.bounds,
                              save_path=self.out_path("detected_objects_map",
                                                      f"detected_objects_map_{step_count:03d}.png"),
                              gt_objects=cfg.object_positions, class_colors=cfg.object_colors)

        if cfg.vision_mode == 'navKnowledge':
            self.navKnowledge.to_csv(os.path.join(self.save_dir, f"navKnowledge_{tag}.csv"), index=False)
        else:
            pose = self.get_robot_pose(warn=False)
            self.object_map.plot_mle(self.out_path("object_map", f"object_map_{step_count:03d}.png"),
                                     title=f"Object map (MLE), step {step_count}",
                                     robot_xz=pose[:2] if pose else None)

    def finalize_perception(self):
        self.envKnowledge.to_csv(os.path.join(self.save_dir, "envKnowledge_final.csv"), index=False)
        pd.DataFrame(self.detection_log).to_csv(os.path.join(self.save_dir, "detections_log.csv"), index=False)
        if self.cfg.vision_mode == 'navKnowledge':
            self.navKnowledge.to_csv(os.path.join(self.save_dir, "navKnowledge_final.csv"), index=False)
        elif self.object_map is not None:
            self.object_map.save_final(self.save_dir)

    def out_path(self, folder, filename):
        d = os.path.join(self.save_dir, folder)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, filename)