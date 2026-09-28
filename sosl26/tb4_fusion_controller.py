"""Fusion ('F') TurtleBot4 controller: olfaction + vision.

TB4 counterpart of ControlAlgorithms/Fusion/fusion_controller.py. On top of
the olfactory update in TB4BaseController, each step runs YOLO on the latest
RGB + depth frame and fuses vision with olfaction using one of two vision
approaches (ExperimentConfig.vision_mode):

'navKnowledge' (AI2-THOR approach, as is)
    Detections are merged into an object list (envKnowledge);
    add_goal_similarity scores each object with langSim * olfactionSim
    (navKnowledge). Visual map = langSim heatmap, fused map = goalSim heatmap,
    target = top navKnowledge object.

'dirichlet' (object distribution, sOSL_dirichletFunctions.py)
    Detections add Dirichlet evidence to a per-cell class distribution
    (objects +1, observed empty space between camera and objects -> Background
    +0.5, footprint radius from the bounding box). Visual map
    P(src | V) = normalised sum_k p(o_k | z) * sim(class_k, goal);
    fused map = normalised P(src | C) * P(src | V); target = fused arg-max.

Both modes use the same detections, and envKnowledge is kept in both so the
detected-objects map is always available.
"""

import math
import os

import cv2
import numpy as np
import pandas as pd

from sOSL_dirichletFunctions import DirichletObjectMap, camera_pose_in_map
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
        super().__init__(cfg, save_dir, node_name="sosl_tb4_fusion_controller")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def dirichlet_classes(self):
        """Object classes of the Dirichlet map: model classes after target / exclude filters."""
        names = list(self.yolo_model.names.values())
        if self.cfg.yolo_target_classes:
            names = [n for n in names if n in self.cfg.yolo_target_classes]
        return [n for n in names if n not in self.cfg.yolo_exclude_classes]

    def setup_perception(self):
        if self.cfg.vision_mode != 'dirichlet':
            return
        cfg, g = self.cfg, self.grid
        self.object_map = DirichletObjectMap(
            g.x_points, g.z_points, self.dirichlet_classes(),
            object_evidence=cfg.dirichlet_object_evidence,
            background_evidence=cfg.dirichlet_background_evidence,
            prior_strength=cfg.dirichlet_prior_strength,
            min_radius=cfg.dirichlet_min_radius,
            max_radius=cfg.dirichlet_max_radius,
        )
        table = self.object_map.set_class_similarity(
            self._vision.model, self.goal_phrase,
            background_similarity=cfg.dirichlet_background_similarity)
        self.get_logger().info(
            f"Dirichlet object map: {self.object_map.K} classes (incl. Background). "
            f"Top class similarities to '{self.goal_phrase}':\n{table.head(8).to_string(index=False)}")

    def run_info_extra(self):
        info = {"yolo_classes": dict(getattr(self.yolo_model, "names", {}))}
        if self.object_map is not None:
            info["dirichlet_classes"] = self.object_map.classes
        return info

    # ------------------------------------------------------------------
    # Per-step perception
    # ------------------------------------------------------------------

    def perceive(self, pose, step_count, srcProbGivenOlfactory):
        cfg, g = self.cfg, self.grid
        rgb = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        self._frames = (rgb, depth)

        K = self.camera_intrinsics(rgb.shape)
        optical_to_map = self.optical_to_map_fn(pose)

        self.envKnowledge, annotated, detections = visionBranch(
            self.yolo_model, self.envKnowledge, rgb, depth,
            K=K,
            optical_to_map=optical_to_map,
            save_dir=self.save_dir, step_count=step_count,
            confThr=cfg.yolo_conf,
            target_names=cfg.yolo_target_classes,
            exclude_names=cfg.yolo_exclude_classes,
            depth_percentile=cfg.depth_percentile,
            depth_range=cfg.depth_range,
            merge_dist=cfg.merge_dist,
            logger=self.get_logger().info,
        )
        self.latest_annotated = annotated
        self.detection_log += [dict(step=step_count, **d) for d in detections]

        if cfg.vision_mode == 'navKnowledge':
            return self._perceive_navknowledge(srcProbGivenOlfactory)

        cam_xz, heading = camera_pose_in_map(optical_to_map)
        hfov = 2.0 * math.atan(rgb.shape[1] / (2.0 * K[0]))
        return self._perceive_dirichlet(srcProbGivenOlfactory, detections, cam_xz, heading, hfov)

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

        return dict(
            panels=[(vision_raw, 'V'), (goal_raw, 'F')],
            arrays=dict(visual=vision_raw, fused=goal_raw),
            target_object=target_object,
            target_coordinate=target_coordinate,
            target_xz=target_xz,
            visual_entropy=map_entropy(vision_raw),
            fused_entropy=map_entropy(goal_raw),
        )

    def _perceive_dirichlet(self, srcProbGivenOlfactory, detections, cam_xz, heading, hfov):
        g, om = self.grid, self.object_map
        stats = om.update(detections, cam_xz, heading, hfov, max_range=self.cfg.depth_range[1])
        self.get_logger().info(
            f"Dirichlet update: {stats['object_cells']} object cells, {stats['background_cells']} background cells"
            + (f", skipped {stats['skipped']}" if stats['skipped'] else ""))

        srcProbGivenVision = om.source_prob_given_vision()
        fused = srcProbGivenOlfactory * srcProbGivenVision
        total = fused.sum()
        fused = fused / total if total > 1e-300 else np.full_like(fused, 1.0 / fused.size)

        row, col = np.unravel_index(np.argmax(fused), fused.shape)
        x, z = grid_to_world((row, col), g.x_points, g.z_points)

        return dict(
            panels=[(srcProbGivenVision, 'V'), (fused, 'F')],
            arrays=dict(visual=srcProbGivenVision, fused=fused, beta=om.beta.astype(np.float32),
                        observed=om.observed, mle_class=om.mle_class()),
            target_object=om.top_object_at(row, col),
            target_coordinate=format_position(x, z, 0.0),
            target_xz=np.array([x, z]),
            visual_entropy=map_entropy(srcProbGivenVision),
            fused_entropy=map_entropy(fused),
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_perception_outputs(self, step_count, tag):
        g = self.grid
        rgb, depth = self._frames
        cv2.imwrite(os.path.join(self.save_dir, f"frame_{tag}.png"), rgb)
        cv2.imwrite(os.path.join(self.save_dir, f"depth_{step_count:03d}.png"),
                    np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))

        plot_detected_objects(itemDF=self.envKnowledge, mask_closed=g.mask, scene_bounds_tuple=g.bounds,
                              save_path=os.path.join(self.save_dir, f"detected_objects_map_{step_count:03d}.png"))

        if self.cfg.vision_mode == 'navKnowledge':
            self.navKnowledge.to_csv(os.path.join(self.save_dir, f"navKnowledge_{tag}.csv"), index=False)
        else:
            pose = self.get_robot_pose(warn=False)
            self.object_map.plot_mle(os.path.join(self.save_dir, f"object_map_{step_count:03d}.png"),
                                     title=f"Object map (MLE), step {step_count}",
                                     robot_xz=pose[:2] if pose else None)

    def finalize_perception(self):
        self.envKnowledge.to_csv(os.path.join(self.save_dir, "envKnowledge_final.csv"), index=False)
        pd.DataFrame(self.detection_log).to_csv(os.path.join(self.save_dir, "detections_log.csv"), index=False)
        if self.cfg.vision_mode == 'navKnowledge':
            self.navKnowledge.to_csv(os.path.join(self.save_dir, "navKnowledge_final.csv"), index=False)
        elif self.object_map is not None:
            self.object_map.save_final(self.save_dir)
