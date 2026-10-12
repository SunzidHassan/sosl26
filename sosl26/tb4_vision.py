"""Vision branch of the fusion ('F') experiment: YOLO -> objects in the map -> fusion with olfaction.

How it is used by the controller (tb4_controller.py):

  during a step  : process_frame() is called for every new camera frame (at most
                   one per vision_period s). Each frame is detected with YOLO, the
                   detections are placed in the map, and their evidence is stored.
  end of a step  : fuse(olfactory_map) turns the accumulated vision evidence into a
                   visual source map, combines it with the olfactory map and picks
                   the source estimate.

Two ways to turn detections into a visual map (ExperimentConfig.vision_mode):

  'navKnowledge'  AI2-THOR approach, unchanged. Detections are merged into an object
                  list (envKnowledge); add_goal_similarity scores each object with
                  langSim (name vs. goal phrase) * olfactionSim (olfactory map at the
                  object). Visual map = langSim heatmap, fused map = goalSim heatmap,
                  estimate = best-scoring object.

  'dirichlet'     Object distribution (sOSL_dirichletFunctions.py). Every frame adds
                  evidence to a per-cell class distribution (objects +1, empty space
                  seen between the camera and an object +0.5 Background). Visual map
                  P(src | V) = sum_k p(class_k | cell) * similarity(class_k, goal);
                  fused map = P(src | olfaction) * P(src | V), normalised;
                  estimate = best fused cell.

envKnowledge is kept in both modes so the detected-objects map is always saved.
Nothing here imports rclpy; the controller passes in images, intrinsics and transforms.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from sOSL_dirichletFunctions import DirichletObjectMap
from sOSL_loggerFunctions import generate_heatmap, map_entropy, plot_detected_objects
from sOSL_tb4Functions import boxDepth, camera_pose_in_map, format_position, pixel_to_optical
from sOSL_utils import grid_to_world, parse_position_string


@dataclass
class StepEstimate:
    """What one step produced, besides the olfactory map (also used by olfactory-only runs)."""

    target_object: str                    # class of the estimated source object ('N/A' if none)
    target_coordinate: str                # ai2thor style "x, height, map_y" string, or 'N/A'
    target_xz: Optional[np.ndarray]       # map (x, y) of the estimate, or None
    panels: list = field(default_factory=list)   # extra (map, title) panels drawn after the olfactory map
    arrays: dict = field(default_factory=dict)   # extra arrays saved in maps_XXX.npz
    visual_entropy: float = np.nan
    fused_entropy: float = np.nan
    log: dict = field(default_factory=dict)      # extra trajectory_log.csv columns


# ======================================================================
# Detection: YOLO + depth -> objects in the map frame
# ======================================================================

def detect_objects(model, rgb_bgr, depth_m, K, optical_to_map, confThr=0.3,
                   target_names=None, exclude_names=(), depth_percentile=50,
                   depth_range=(0.2, 5.0), logger=print):
    """Runs YOLO on one frame and places every detection in the map frame.

    For each box: depth = percentile of the depth pixels inside it (boxDepth),
    then the box centre is back-projected with the camera intrinsics K and
    transformed to the map with `optical_to_map`.

    Returns
    -------
    (detections, annotated_image)
        detections: list of dicts with objectType, Conf, map_x, map_y, map_z,
        depth (m) and radius (m) = half the box width in metres, which the
        Dirichlet map uses as the object's footprint.
    """
    # Restrict YOLO to the requested classes (None = all model classes)
    target_classes = None
    if target_names:
        target_classes = [idx for idx, name in model.names.items() if name in target_names]

    results = model(rgb_bgr, verbose=False, conf=confThr, classes=target_classes)
    annotated_img = results[0].plot()

    detections = []
    for box in results[0].boxes:
        confidence = box.conf[0].item()
        if confidence <= confThr:
            continue
        className = model.names[int(box.cls[0].item())]
        if className in exclude_names:
            continue

        # Box centre and size in pixels
        x, y, w, h = box.xywh[0]
        x_pix, y_pix, w_pix, h_pix = round(x.item()), round(y.item()), round(w.item()), round(h.item())

        d = boxDepth(x_pix, y_pix, w_pix, h_pix, depth_m, rgb_bgr.shape,
                     percentile=depth_percentile, min_depth=depth_range[0], max_depth=depth_range[1])
        if d <= 0:
            continue    # no valid depth inside the box

        p_map = optical_to_map(pixel_to_optical(x_pix, y_pix, d, K))
        if p_map is None:
            continue
        map_x, map_y, map_z = (float(v) for v in p_map)

        # Half the box width converted to metres at that depth (pinhole model)
        radius = (w_pix / 2.0) * d / K[0]
        detections.append(dict(objectType=className, Conf=confidence, map_x=map_x, map_y=map_y,
                               map_z=map_z, depth=d, radius=radius))
        logger(f"Detected {className} ({confidence:.2f}) at depth {d:.2f} m, radius {radius:.2f} m "
               f"-> map ({map_x:.2f}, {map_y:.2f})")

        # Write depth + map position under the YOLO label
        cv2.putText(annotated_img, f"{d:.2f}m ({map_x:.2f}, {map_y:.2f})",
                    (max(0, x_pix - w_pix // 2), min(annotated_img.shape[0] - 5, y_pix + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return detections, annotated_img


def merge_into_envKnowledge(envKnowledge, detections, merge_dist=0.1, logger=print):
    """Adds detections to the object list, with the AI2-THOR visionBranch merge logic.

    A detection whose position is within `merge_dist` of an existing object of
    the same class is averaged into it (keeping the higher confidence);
    otherwise it becomes a new row. Positions are "x, height, map_y" strings.
    """
    updated = envKnowledge.copy()
    for det in detections:
        className, confidence = det['objectType'], det['Conf']
        new_position = np.array([det['map_x'], det['map_z'], det['map_y']])   # ai2thor order

        merged = False
        if not updated.empty and 'objectType' in updated.columns:
            for idx in updated.index[updated['objectType'] == className].tolist():
                try:
                    existing = np.array([float(v.strip()) for v in updated.loc[idx, 'Position'].split(',')])
                    if np.linalg.norm(new_position - existing) < merge_dist:
                        avg = (new_position + existing) / 2.0
                        updated.loc[idx, 'Position'] = f"{avg[0]:.2f}, {avg[1]:.2f}, {avg[2]:.2f}"
                        updated.loc[idx, 'Conf'] = max(confidence, updated.loc[idx, 'Conf'])
                        merged = True
                        # (no break: like the AI2-THOR version, every close match is updated)
                except Exception as e:
                    logger(f"Error processing existing position for {className} at index {idx}: {e}")

        if not merged:
            row = pd.DataFrame({"objectType": [className], "Conf": [confidence],
                                "Position": [format_position(det['map_x'], det['map_y'], det['map_z'])]})
            updated = pd.concat([updated, row], ignore_index=True)

    # Keep the expected columns even when nothing has been detected yet
    for col in ['objectType', 'Conf', 'Position']:
        if col not in updated.columns:
            updated[col] = pd.Series(dtype='object' if col != 'Conf' else 'float')
    return updated


# ======================================================================
# Vision branch: accumulate frames, fuse with olfaction, save
# ======================================================================

class Vision:

    def __init__(self, cfg, yolo_model, goal_phrase, save_dir, logger):
        # Imported here (not at module level) so olfactory-only runs never load
        # the sentence transformer that sOSL_visionFunctions loads on import.
        import sOSL_visionFunctions
        self.add_goal_similarity = sOSL_visionFunctions.add_goal_similarity
        self.sentence_model = sOSL_visionFunctions.model

        self.cfg = cfg
        self.model = yolo_model
        self.goal_phrase = goal_phrase
        self.save_dir = save_dir
        self.log = logger                       # rclpy logger (info / warn / error / debug)

        self.grid = None
        self.envKnowledge = pd.DataFrame()      # object list, built in both modes
        self.navKnowledge = pd.DataFrame()      # 'navKnowledge': envKnowledge + similarity scores
        self.object_map = None                  # 'dirichlet': DirichletObjectMap
        self.detection_log = []                 # every detection of the run -> detections_log.csv

        self.last_frame = (None, None)          # (rgb, depth) of the last processed frame
        self.frame_count = 0                    # frames processed in the whole run
        self.step_frames = 0                    # frames / detections in the current step window
        self.step_detections = 0

    def setup(self, grid):
        """Called once the search grid is known."""
        self.grid = grid
        if self.cfg.vision_mode != 'dirichlet':
            return
        cfg = self.cfg
        self.object_map = DirichletObjectMap(
            grid.x_points, grid.z_points, self.dirichlet_classes(),
            object_evidence=cfg.dirichlet_object_evidence,
            background_evidence=cfg.dirichlet_background_evidence,
            prior_strength=cfg.dirichlet_prior_strength,
            min_radius=cfg.dirichlet_min_radius,
            max_radius=cfg.dirichlet_max_radius,
        )
        # How related each class name is to the goal phrase; computed once per run
        table = self.object_map.set_class_similarity(
            self.sentence_model, self.goal_phrase, background_similarity=cfg.dirichlet_background_similarity)
        self.log.info(f"Dirichlet object map: {self.object_map.K} classes (incl. Background). "
                      f"Top class similarities to '{self.goal_phrase}':\n{table.head(8).to_string(index=False)}")

    def dirichlet_classes(self):
        """Classes of the object map: the YOLO classes left after the target / exclude filters."""
        names = list(self.model.names.values())
        if self.cfg.yolo_target_classes:
            names = [n for n in names if n in self.cfg.yolo_target_classes]
        return [n for n in names if n not in self.cfg.yolo_exclude_classes]

    def run_info(self):
        info = {"yolo_classes": dict(getattr(self.model, "names", {}))}
        if self.object_map is not None:
            info["dirichlet_classes"] = self.object_map.classes
        return info

    # ------------------------------------------------------------------
    # During the step: one call per camera frame
    # ------------------------------------------------------------------

    def process_frame(self, rgb, depth, K, optical_to_map, step, robot_xy, t):
        """Detects objects in one frame and adds them to envKnowledge and the object map.

        Returns the annotated image (shown in the YOLO window).
        """
        cfg = self.cfg
        self.last_frame = (rgb, depth)

        detections, annotated = detect_objects(
            self.model, rgb, depth, K, optical_to_map,
            confThr=cfg.yolo_conf, target_names=cfg.yolo_target_classes,
            exclude_names=cfg.yolo_exclude_classes, depth_percentile=cfg.depth_percentile,
            depth_range=cfg.depth_range, logger=self.log.debug)

        self.envKnowledge = merge_into_envKnowledge(self.envKnowledge, detections, cfg.merge_dist, self.log.debug)

        if self.object_map is not None:
            # Each frame is one more observation: objects add evidence in their footprint,
            # the empty space the camera saw in front of them adds Background evidence.
            cam_xy, heading = camera_pose_in_map(optical_to_map)
            hfov = 2.0 * math.atan(rgb.shape[1] / (2.0 * K[0]))
            self.object_map.update(detections, cam_xy, heading, hfov, max_range=cfg.depth_range[1])

        self.detection_log += [dict(step=step, frame=self.frame_count, time=t,
                                    robot_x=robot_xy[0], robot_z=robot_xy[1], **d) for d in detections]
        if cfg.save_vision_frames:
            cv2.imwrite(os.path.join(self.save_dir, f"yolo_{step:03d}_{self.frame_count:04d}.jpg"), annotated)

        self.frame_count += 1
        self.step_frames += 1
        self.step_detections += len(detections)
        return annotated

    # ------------------------------------------------------------------
    # End of the step: fuse with the olfactory map
    # ------------------------------------------------------------------

    def fuse(self, srcProbGivenOlfactory):
        """Returns the StepEstimate (visual + fused maps, source estimate) for this step."""
        frames, dets = self.step_frames, self.step_detections
        self.step_frames = self.step_detections = 0          # next window starts from zero
        self.log.info(f"Vision this step: {frames} frames, {dets} detections "
                      f"({len(self.envKnowledge)} objects in envKnowledge)")
        if frames == 0:
            self.log.warn("No camera frames were processed in this step window.")

        if self.cfg.vision_mode == 'navKnowledge':
            estimate = self._fuse_navknowledge(srcProbGivenOlfactory)
        else:
            estimate = self._fuse_dirichlet(srcProbGivenOlfactory)
        estimate.log = dict(vision_frames=frames, vision_detections=dets)
        return estimate

    def _fuse_navknowledge(self, srcProbGivenOlfactory):
        g = self.grid
        # Score every object: goalSim = langSim (name vs. goal) * olfactionSim (olfactory map at the object)
        self.navKnowledge = self.add_goal_similarity(
            self.envKnowledge.copy(), self.goal_phrase, srcProbGivenOlfactory,
            g.x_points, g.z_points, alg_choice='F')
        if not self.navKnowledge.empty:
            self.log.info(f"NavKnowledge:\n{self.navKnowledge.head(4).to_string()}")

        # Maps for the figure: each object's score painted into its grid cell
        if not self.navKnowledge.empty:
            vision_raw = generate_heatmap(self.navKnowledge, g.x_points, g.z_points, weight_key='langSim', sigma=1)
            goal_raw = generate_heatmap(self.navKnowledge, g.x_points, g.z_points, weight_key='goalSim', sigma=1)
        else:
            vision_raw = np.zeros_like(srcProbGivenOlfactory)
            goal_raw = np.zeros_like(srcProbGivenOlfactory)

        # Estimate = the best-scoring object (navKnowledge is sorted by goalSim)
        estimate = StepEstimate(target_object='N/A', target_coordinate='N/A', target_xz=None)
        if not self.navKnowledge.empty:
            top = self.navKnowledge.iloc[0]
            pos = parse_position_string(top["Position"])
            estimate = StepEstimate(top["objectType"], top["Position"], np.array([pos[0], pos[2]]))

        estimate.panels = [(vision_raw, 'V'), (goal_raw, 'F')]
        estimate.arrays = dict(visual=vision_raw, fused=goal_raw)
        estimate.visual_entropy = map_entropy(vision_raw)
        estimate.fused_entropy = map_entropy(goal_raw)
        return estimate

    def _fuse_dirichlet(self, srcProbGivenOlfactory):
        g, om = self.grid, self.object_map
        # P(src | V): per cell, how goal-related the objects believed to be there are
        srcProbGivenVision = om.source_prob_given_vision()

        # Fused map: product of the two independent beliefs, renormalised
        fused = srcProbGivenOlfactory * srcProbGivenVision
        total = fused.sum()
        fused = fused / total if total > 1e-300 else np.full_like(fused, 1.0 / fused.size)

        # Estimate = the most likely fused cell, labelled with the most likely object there
        row, col = np.unravel_index(np.argmax(fused), fused.shape)
        x, y = grid_to_world((row, col), g.x_points, g.z_points)
        return StepEstimate(
            target_object=om.top_object_at(row, col),
            target_coordinate=format_position(x, y, 0.0),
            target_xz=np.array([x, y]),
            panels=[(srcProbGivenVision, 'V'), (fused, 'F')],
            arrays=dict(visual=srcProbGivenVision, fused=fused, beta=om.beta.astype(np.float32),
                        observed=om.observed, mle_class=om.mle_class()),
            visual_entropy=map_entropy(srcProbGivenVision),
            fused_entropy=map_entropy(fused),
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save_step(self, step, tag, robot_xy):
        """Per-step vision files: last frame, depth, detected objects, navKnowledge / object map."""
        g = self.grid
        rgb, depth = self.last_frame
        if rgb is not None:
            cv2.imwrite(os.path.join(self.save_dir, f"frame_{tag}.png"), rgb)
            cv2.imwrite(os.path.join(self.save_dir, f"depth_{step:03d}.png"),
                        np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))   # millimetres, 16 bit

        plot_detected_objects(itemDF=self.envKnowledge, mask_closed=g.mask, scene_bounds_tuple=g.bounds,
                              save_path=os.path.join(self.save_dir, f"detected_objects_map_{step:03d}.png"))

        if self.cfg.vision_mode == 'navKnowledge':
            self.navKnowledge.to_csv(os.path.join(self.save_dir, f"navKnowledge_{tag}.csv"), index=False)
        else:
            self.object_map.plot_mle(os.path.join(self.save_dir, f"object_map_{step:03d}.png"),
                                     title=f"Object map (MLE), step {step}", robot_xz=robot_xy)

    def save_final(self):
        """End-of-run vision files."""
        self.envKnowledge.to_csv(os.path.join(self.save_dir, "envKnowledge_final.csv"), index=False)
        pd.DataFrame(self.detection_log).to_csv(os.path.join(self.save_dir, "detections_log.csv"), index=False)
        if self.cfg.vision_mode == 'navKnowledge':
            self.navKnowledge.to_csv(os.path.join(self.save_dir, "navKnowledge_final.csv"), index=False)
        elif self.object_map is not None:
            self.object_map.save_final(self.save_dir)
