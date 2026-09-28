"""TurtleBot4 adapters for the sOSL olfaction / vision pipeline.

The AI2-THOR functions in sOSL_visionFunctions.py read everything from an
ai2thor `controller`. This module provides the TB4 equivalents that work on
plain sensor data (decoded images, camera intrinsics, TF poses), so the rest
of the pipeline (BayesianAgent, add_goal_similarity, generate_heatmap,
map_entropy, plot_detected_objects, ...) is reused unchanged.

Nothing in this module imports rclpy, so it can be tested offline.

Coordinate convention
---------------------
The AI2-THOR pipeline works on the (x, z) ground plane with y pointing up and
stores object positions as the string "x, y, z". To reuse those functions
as-is, ROS `map` frame coordinates are mapped as:

    ai2thor x  <-  map x
    ai2thor y  <-  map z   (height)
    ai2thor z  <-  map y

So `robot_z`, `z_points` and the third entry of a "Position" string are all
the ROS map *y* coordinate.

Handedness: AI2-THOR/Unity is left-handed (x right, y up, z forward), ROS
(REP 103) is right-handed (x forward, y left, z up). Swapping the y and z
axes as above is exactly the right-to-left-handed conversion, so positions
map 1:1 and a top-down plot (x right, z/map-y up, origin='lower') shows the
same picture in both. Only *angles* differ:

    ROS yaw      : counter-clockwise from +x      (atan2(dy, dx))
    ai2thor yaw  : clockwise from +z (= map +y)   (atan2(dx, dz), rotation.y)
    ai2thor_yaw_deg = (90 - ros_yaw_deg) mod 360

Use the ROS convention for anything sent to the robot (Nav2 goals,
cmd_vel); use ros_yaw_to_ai2thor_deg() only for logs compared against the
AI2-THOR runs. The camera projection here uses the ROS optical frame + TF
and does not reuse the Unity-camera math of coord23D_focal.
"""

import math
import os
import struct
from dataclasses import dataclass, field

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from sOSL_olfactionFunctions import BayesianAgent, gaussian_plume  # noqa: E402
from sOSL_loggerFunctions import generate_heatmap  # noqa: E402

# Same font sizes as ControlAlgorithms/Fusion/fusion_controller.py
TITLE_FONTSIZE = 40
LABEL_FONTSIZE = 32


# ==========================
# Pose / Frame Helpers
# ==========================

def yaw_from_quaternion(qx, qy, qz, qw):
    """Returns the yaw (rad) of a quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def ros_yaw_to_ai2thor_deg(yaw_rad):
    """ROS yaw (rad, CCW from +x) -> ai2thor yaw (deg, CW from +z = map +y)."""
    return (90.0 - math.degrees(yaw_rad)) % 360.0


def ai2thor_deg_to_ros_yaw(yaw_deg):
    """ai2thor yaw (deg, CW from +z = map +y) -> ROS yaw (rad, CCW from +x, in [-pi, pi))."""
    return (math.radians(90.0 - yaw_deg) + math.pi) % (2.0 * math.pi) - math.pi


def ros_heading_to(robot_xy, target_xy):
    """ROS yaw (rad) pointing from robot_xy to target_xy in the map frame.

    ROS counterpart of fusion_controller's `atan2(delta_x, delta_z)`.
    """
    return math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])


def format_position(map_x, map_y, map_z):
    """Formats a ROS map-frame point as an ai2thor style "x, y, z" string."""
    return f"{map_x:.2f}, {map_z:.2f}, {map_y:.2f}"


# ==========================
# Image Decoding
# ==========================

def decode_compressed_rgb(data):
    """Decodes a sensor_msgs/CompressedImage payload into a BGR image."""
    np_arr = np.frombuffer(bytes(data), dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def decode_compressed_depth(data, fmt=""):
    """Decodes a compressedDepth payload into a float32 depth image in meters.

    compressed_depth_image_transport prepends a 12 byte header
    (int32 format + float32 depthParam[2]) to a PNG. 16UC1 images are in
    millimeters; 32FC1 images are stored as quantized inverse depth.
    Invalid pixels are returned as 0.
    """
    raw = bytes(data)
    png_start = raw.find(b'\x89PNG')
    if png_start < 0:
        png_start = 12
    img = cv2.imdecode(np.frombuffer(raw[png_start:], np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if '32FC1' in fmt and png_start >= 12:
        _, quant_a, quant_b = struct.unpack('<iff', raw[:12])
        img = img.astype(np.float32)
        depth = np.zeros_like(img)
        valid = img > 0
        depth[valid] = quant_a / (img[valid] - quant_b)
        return depth

    # 16UC1 (OAK-D default): millimeters
    return img.astype(np.float32) / 1000.0


# ==========================
# Camera Model
# ==========================

def intrinsics_from_fov(width, height, hfov_deg):
    """Pinhole intrinsics (fx, fy, cx, cy) from a horizontal field of view.

    Same model as coord23D_focal: f = W / (2 * tan(FOV / 2)).
    """
    f = width / (2.0 * np.tan(np.deg2rad(hfov_deg / 2.0)))
    return f, f, width / 2.0, height / 2.0


def scale_intrinsics(K, info_size, image_size):
    """Rescales (fx, fy, cx, cy) from the CameraInfo size to the image size."""
    fx, fy, cx, cy = K
    info_w, info_h = info_size
    img_w, img_h = image_size
    if info_w <= 0 or info_h <= 0 or (info_w == img_w and info_h == img_h):
        return K
    sx, sy = img_w / info_w, img_h / info_h
    return fx * sx, fy * sy, cx * sx, cy * sy


def boxDepth(x, y, w, h, depth_m, rgb_shape, percentile=50, shrink=0.5,
             min_depth=0.2, max_depth=5.0):
    """Estimates an object's depth (m) from its bounding box.

    TB4 version of sOSL_visionFunctions.boxDepth. The RGB bounding box is
    rescaled to the depth image resolution, shrunk around its center (real
    boxes contain a lot of background), and the requested percentile of the
    valid depth pixels is returned. Returns 0.0 when no valid depth exists.
    """
    if depth_m is None:
        return 0.0
    rgb_h, rgb_w = rgb_shape[:2]
    frame_h, frame_w = depth_m.shape[:2]
    sx, sy = frame_w / rgb_w, frame_h / rgb_h

    cx, cy = x * sx, y * sy
    half_w, half_h = max(1.0, w * sx * shrink / 2.0), max(1.0, h * sy * shrink / 2.0)
    vMin = max(0, int(round(cy - half_h)))
    vMax = min(frame_h, int(round(cy + half_h)))
    hMin = max(0, int(round(cx - half_w)))
    hMax = min(frame_w, int(round(cx + half_w)))
    if vMin >= vMax or hMin >= hMax:
        return 0.0

    depth_values = depth_m[vMin:vMax, hMin:hMax]
    depth_values = depth_values[np.isfinite(depth_values)
                                & (depth_values >= min_depth)
                                & (depth_values <= max_depth)]
    if depth_values.size == 0:
        return 0.0
    return round(float(np.percentile(depth_values, percentile)), 2)


def pixel_to_optical(u, v, d, K):
    """Back-projects pixel (u, v) at depth d into the camera optical frame.

    Optical frame (ROS REP 103): +X right, +Y down, +Z forward.
    """
    fx, fy, cx, cy = K
    return np.array([(u - cx) * d / fx, (v - cy) * d / fy, d])


def transform_point(p, translation, quaternion_xyzw):
    """Applies a TF transform (translation, quaternion x,y,z,w) to a point."""
    return Rotation.from_quat(quaternion_xyzw).apply(p) + np.asarray(translation)


def optical_to_map_from_robot_pose(p_opt, robot_x, robot_y, robot_yaw,
                                   camera_height=0.25, camera_forward=0.0):
    """Fallback optical -> map transform when the camera TF is unavailable.

    Assumes a level camera looking along the robot heading, like the yaw-only
    rotation in coord23D_focal.
    """
    forward = p_opt[2] + camera_forward
    left = -p_opt[0]
    up = camera_height - p_opt[1]
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    return np.array([robot_x + forward * c - left * s,
                     robot_y + forward * s + left * c,
                     up])


# ==========================
# Vision Branch (TB4)
# ==========================

def visionBranch(model, itemDF, rgb_bgr, depth_m, K, optical_to_map,
                 save_dir=None, step_count=0, confThr=0.3,
                 target_names=None, exclude_names=(), depth_percentile=50,
                 depth_range=(0.2, 5.0), merge_dist=0.1, logger=print):
    """TB4 version of sOSL_visionFunctions.visionBranch.

    Runs YOLO on the latest RGB frame, projects every detection to the map
    frame, and merges it into `itemDF` with the same logic as the AI2-THOR
    version (average positions of same-class detections closer than
    `merge_dist`, otherwise append a new row).

    Parameters
    ----------
    optical_to_map : callable
        Maps an optical-frame point (np.ndarray of 3) to a map-frame point
        (map_x, map_y, map_z), or returns None when it cannot.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray | None, list[dict]]
        Updated itemDF, the annotated frame (BGR), and this frame's
        detections: dicts with objectType, Conf, map_x, map_y, map_z,
        depth (m) and radius (m, half the metric bounding box width, used as
        the Dirichlet footprint radius).
    """
    updated_itemDF = itemDF.copy()
    detections = []
    if rgb_bgr is None:
        return updated_itemDF, None, detections

    target_classes = None
    if target_names:
        target_classes = [idx for idx, name in model.names.items() if name in target_names]

    results = model(rgb_bgr, verbose=False, conf=confThr, classes=target_classes)
    annotated_img = results[0].plot()

    for box in results[0].boxes:
        confidence = box.conf[0].item()
        if confidence <= confThr:
            continue
        class_id = int(box.cls[0].item())
        className = model.names[class_id]
        if className in exclude_names:
            continue

        x, y, w, h = box.xywh[0]
        x_pix, y_pix, w_pix, h_pix = round(x.item()), round(y.item()), round(w.item()), round(h.item())

        d = boxDepth(x_pix, y_pix, w_pix, h_pix, depth_m, rgb_bgr.shape,
                     percentile=depth_percentile,
                     min_depth=depth_range[0], max_depth=depth_range[1])
        if d <= 0:
            continue

        p_map = optical_to_map(pixel_to_optical(x_pix, y_pix, d, K))
        if p_map is None:
            continue
        map_x, map_y, map_z = (float(v) for v in p_map)

        # Stored in ai2thor order: x, up, z(=map y)
        new_position = np.array([map_x, map_z, map_y])
        position_str = format_position(map_x, map_y, map_z)
        radius = (w_pix / 2.0) * d / K[0]
        detections.append(dict(objectType=className, Conf=confidence, map_x=map_x, map_y=map_y,
                               map_z=map_z, depth=d, radius=radius))
        logger(f"Detected {className} ({confidence:.2f}) at depth {d:.2f} m, radius {radius:.2f} m "
               f"-> map ({map_x:.2f}, {map_y:.2f})")

        cv2.putText(annotated_img, f"{d:.2f}m ({map_x:.2f}, {map_y:.2f})",
                    (max(0, x_pix - w_pix // 2), min(annotated_img.shape[0] - 5, y_pix + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # --- Same merge logic as the AI2-THOR visionBranch ---
        updated = False
        if not updated_itemDF.empty and 'objectType' in updated_itemDF.columns:
            match_indices = updated_itemDF.index[updated_itemDF['objectType'] == className].tolist()
            for idx in match_indices:
                try:
                    existing_position_str = updated_itemDF.loc[idx, 'Position']
                    existing_position = np.array([float(val.strip()) for val in existing_position_str.split(',')])
                    dist = np.linalg.norm(new_position - existing_position)
                    if dist < merge_dist:
                        avg_position = (new_position + existing_position) / 2.0
                        updated_itemDF.loc[idx, 'Position'] = f"{avg_position[0]:.2f}, {avg_position[1]:.2f}, {avg_position[2]:.2f}"
                        updated_itemDF.loc[idx, 'Conf'] = max(confidence, updated_itemDF.loc[idx, 'Conf'])
                        updated = True
                except Exception as e:
                    logger(f"Error processing existing position for {className} at index {idx}: {e}")
                    continue

        if not updated:
            new_row_df = pd.DataFrame({
                "objectType": [className],
                "Conf": [confidence],
                "Position": [position_str],
            })
            updated_itemDF = pd.concat([updated_itemDF, new_row_df], ignore_index=True)

    for col in ['objectType', 'Conf', 'Position']:
        if col not in updated_itemDF.columns:
            updated_itemDF[col] = pd.Series(dtype='object' if col != 'Conf' else 'float')

    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, f"yolo_step{step_count}.jpg"), annotated_img)

    return updated_itemDF, annotated_img, detections


# ==========================
# Olfaction (parameterised plume)
# ==========================

class TB4BayesianAgent(BayesianAgent):
    """BayesianAgent whose plume model uses experiment-specific q_s, D, tau.

    sOSL_olfactionFunctions.BayesianAgent.posterior calls gaussian_plume with
    its defaults. This subclass keeps the exact same update but forwards the
    plume parameters set in main.
    """

    def __init__(self, pos, src_pos, x_points, z_points, sigma_noise,
                 reachable_positions, q_s=2000, D=10, tau=1000, U=0, psi_deg=0):
        super().__init__(pos, src_pos, x_points, z_points, sigma_noise, reachable_positions)
        self.plume_params = dict(q_s=q_s, D=D, U=U, tau=tau, psi_deg=psi_deg)

    def expected_concentration(self, x, z, source):
        return gaussian_plume(x, z, source, **self.plume_params)

    def posterior(self, current_odor_concentration, robot_x, robot_z, smooth_sigma=2.0):
        likelihood_map = np.zeros_like(self.prob_map)
        for iz, z_source in enumerate(self.z_points):
            for ix, x_source in enumerate(self.x_points):
                expected = self.expected_concentration(robot_x, robot_z, (x_source, z_source))
                likelihood_map[iz, ix] = self.likelihood(current_odor_concentration, expected, self.sigma_noise)

        self.prob_map *= likelihood_map

        map_sum = np.sum(self.prob_map)
        if map_sum > 1e-9:
            self.prob_map /= map_sum
        else:
            print("Warning: Probability map sum is close to zero after update.")
            self.prob_map = np.full((self.window_z, self.window_x), 1.0 / (self.window_z * self.window_x))


# ==========================
# Search Grid
# ==========================

@dataclass
class SearchGrid:
    """Search grid + reachable mask (ai2thor x/z convention, z = map y)."""

    x_points: np.ndarray
    z_points: np.ndarray
    bounds: tuple                      # (min_x, max_x, min_z, max_z) of the mask
    mask: np.ndarray                   # bool (H, W), True = free, row 0 at min_z
    reachable_positions: list = field(default_factory=list)  # [(x, z), ...]


def _grid_points(lo, hi, step):
    lo = np.floor(lo / step) * step
    hi = np.floor(hi / step) * step
    return np.round(np.arange(lo, hi + step / 2.0, step), 3)


def grid_from_bounds(x_min, x_max, z_min, z_max, step=0.25):
    """Search grid for user supplied map bounds (all cells reachable)."""
    x_points = _grid_points(x_min, x_max, step)
    z_points = _grid_points(z_min, z_max, step)
    mask = np.ones((len(z_points), len(x_points)), dtype=bool)
    reachable = [(float(x), float(z)) for z in z_points for x in x_points]
    return SearchGrid(x_points, z_points, (x_min, x_max, z_min, z_max), mask, reachable)


def grid_from_occupancy(occ, resolution, origin_x, origin_y, step=0.25,
                        margin=0.5, free_thresh=25):
    """Builds the search grid from a nav_msgs/OccupancyGrid.

    The map is cropped to the bounding box of known cells (+ margin) so large,
    mostly unknown SLAM canvases don't blow up the Bayesian grid.

    Parameters
    ----------
    occ : np.ndarray
        int8 occupancy data reshaped to (height, width); -1 unknown,
        0..100 occupancy probability. Row 0 is at origin_y.
    """
    occ = np.asarray(occ)
    known_r, known_c = np.nonzero(occ >= 0)
    if known_r.size == 0:
        raise ValueError("Occupancy grid has no known cells.")

    pad = int(math.ceil(margin / resolution))
    r0 = max(0, known_r.min() - pad)
    r1 = min(occ.shape[0], known_r.max() + 1 + pad)
    c0 = max(0, known_c.min() - pad)
    c1 = min(occ.shape[1], known_c.max() + 1 + pad)
    crop = occ[r0:r1, c0:c1]

    min_x = origin_x + c0 * resolution
    max_x = origin_x + c1 * resolution
    min_z = origin_y + r0 * resolution
    max_z = origin_y + r1 * resolution

    free = (crop >= 0) & (crop < free_thresh)

    x_points = _grid_points(min_x, max_x, step)
    z_points = _grid_points(min_z, max_z, step)

    reachable = []
    for z in z_points:
        r = int((z - min_z) / resolution)
        for x in x_points:
            c = int((x - min_x) / resolution)
            if 0 <= r < free.shape[0] and 0 <= c < free.shape[1] and free[r, c]:
                reachable.append((float(x), float(z)))

    return SearchGrid(x_points, z_points, (min_x, max_x, min_z, max_z), free, reachable)


# ==========================
# Maps + Plots
# ==========================

def compute_maps(prob_map, navKnowledge, x_points, z_points):
    """Returns (olfactory, visual, fused) maps on the search grid.

    olfactory = Bayesian posterior; visual = langSim heatmap; fused = goalSim
    heatmap -- the same three panels as fusion_controller.py.
    """
    if navKnowledge is not None and not navKnowledge.empty:
        vision_raw = generate_heatmap(navKnowledge, x_points, z_points, weight_key='langSim', sigma=1)
        goal_raw = generate_heatmap(navKnowledge, x_points, z_points, weight_key='goalSim', sigma=1)
    else:
        vision_raw = np.zeros_like(prob_map)
        goal_raw = np.zeros_like(prob_map)
    return prob_map.copy(), vision_raw, goal_raw


def _minmax(m):
    m = np.asarray(m, dtype=float)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)


def save_belief_maps(panels, x_points, z_points, out_fname):
    """Saves a row of belief maps, e.g. olfactory / visual / fused.

    Same rendering as the belief map block in fusion_controller.py (each map
    min-max normalised, 'hot' colormap, colorbar on the last panel).

    Parameters
    ----------
    panels : list[tuple[np.ndarray, str]]
        (map, title) pairs; one panel for olfactory-only runs, three for fusion.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    xmin, xmax = min(x_points), max(x_points)
    zmin, zmax = min(z_points), max(z_points)
    extent = [xmin, xmax, zmin, zmax]

    for i, (ax, (raw, title)) in enumerate(zip(axes, panels)):
        img = cv2.applyColorMap((np.clip(_minmax(raw), 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='lower', extent=extent, aspect='equal')
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        ax.set_aspect('equal', adjustable='box')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = cm.ScalarMappable(cmap='hot', norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        plt.colorbar(sm, cax=cax)
        if i < n - 1:
            cax.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_fname, dpi=150)
    plt.close(fig)


def plot_tb4_trajectory(trajectory_log_path, grid, save_path, source_xz=None,
                        estimate_xz=None, plume_params=None):
    """Top-down trajectory plot on the TB4 map.

    TB4 counterpart of sOSL_loggerFunctions.generate_trajectory_plot (which
    needs an ai2thor controller). Draws the plume field when the ground truth
    source is known, greys out non-free map cells, and marks start (lime),
    end (cyan), source (red star, 1 m success circle) and the final estimate.
    """
    try:
        df = pd.read_csv(trajectory_log_path)
    except Exception as e:
        print(f"Error reading trajectory log {trajectory_log_path}: {e}")
        return

    min_x, max_x, min_z, max_z = grid.bounds
    fig, ax = plt.subplots(figsize=(6, 6))

    if source_xz is not None:
        grid_steps = 60
        X = np.linspace(min_x, max_x, grid_steps)
        Z = np.linspace(min_z, max_z, grid_steps)
        field_map = np.array([[gaussian_plume(x, z, tuple(source_xz), **(plume_params or {}))
                               for x in X] for z in Z])
        ax.contourf(X, Z, field_map, levels=60, cmap='magma_r', zorder=1)

    grey = np.array([0.8, 0.8, 0.8, 1.0])
    ov = np.tile(grey, (grid.mask.shape[0], grid.mask.shape[1], 1))
    ov[..., 3] = (~grid.mask).astype(float)
    ax.imshow(ov, extent=[min_x, max_x, min_z, max_z], origin='lower', zorder=2, alpha=0.5, aspect='auto')

    if not df.empty and {'robot_x', 'robot_z'}.issubset(df.columns):
        ax.plot(df.robot_x, df.robot_z, color='purple', linewidth=1.5, zorder=4)
        ax.scatter(df.robot_x, df.robot_z, color='purple', s=30, zorder=5, edgecolors='black', linewidth=0.5)
        ax.scatter(df.robot_x.iloc[0], df.robot_z.iloc[0], color='lime', s=100, zorder=6, edgecolors='black')
        ax.scatter(df.robot_x.iloc[-1], df.robot_z.iloc[-1], color='cyan', s=100, zorder=6, edgecolors='black')

    if source_xz is not None:
        ax.scatter(source_xz[0], source_xz[1], color='red', s=200, marker='*', zorder=7, edgecolors='black')
        ax.add_patch(Circle(tuple(source_xz), radius=1.0, fill=False, edgecolor='red',
                            linewidth=1.5, linestyle='--', zorder=7))

    if estimate_xz is not None:
        ax.scatter(estimate_xz[0], estimate_xz[1], color='yellow', s=150, marker='X', zorder=8, edgecolors='black')

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('map x (m)')
    ax.set_ylabel('map y (m)')
    fig.tight_layout()
    try:
        fig.savefig(save_path, dpi=300)
        print(f"Successfully saved trajectory plot to {save_path}")
    except Exception as e:
        print(f"Error saving trajectory plot to {save_path}: {e}")
    plt.close(fig)
