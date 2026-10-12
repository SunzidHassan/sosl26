"""TurtleBot4 helpers: pose, image decoding, camera projection, YOLO detection,
search grid / boundary, and plots.

Nothing in this module imports rclpy, so it can be tested offline.

Coordinate convention
---------------------
The maps use the AI2-THOR (x, z) ground-plane naming:

    x  =  ROS map x
    z  =  ROS map y

So `robot_z`, `target_z` and `z_points` are the ROS map *y* coordinate.
Positions and top-down plots match ROS; only yaw angles differ:

    ROS yaw      : counter-clockwise from +x      (atan2(dy, dx))
    ai2thor yaw  : clockwise from +z (= map +y)   ai2thor_yaw_deg = (90 - ros_yaw_deg) mod 360

Use the ROS convention for anything sent to the robot; ros_yaw_to_ai2thor_deg()
is only for logs compared against the AI2-THOR runs.
"""

import math
import struct
from dataclasses import dataclass

import cv2
import matplotlib
matplotlib.use('Agg')   # no display needed: plots are only saved to files
import matplotlib.cm as cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from sOSL_olfactionFunctions import gaussian_plume  # noqa: E402

TITLE_FONTSIZE = 40


# ==========================
# Pose helpers
# ==========================

def yaw_from_quaternion(qx, qy, qz, qw):
    """Returns the yaw (rad) of a quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def ros_yaw_to_ai2thor_deg(yaw_rad):
    """ROS yaw (rad, CCW from +x) -> ai2thor yaw (deg, CW from +z = map +y)."""
    return (90.0 - math.degrees(yaw_rad)) % 360.0


# ==========================
# Image decoding
# ==========================

def decode_compressed_rgb(data):
    """Decodes a sensor_msgs/CompressedImage payload into a BGR image."""
    return cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)


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

    return img.astype(np.float32) / 1000.0   # 16UC1 (OAK-D default): millimeters


# ==========================
# Camera model
# ==========================

def intrinsics_from_fov(width, height, hfov_deg):
    """Pinhole intrinsics (fx, fy, cx, cy) from a horizontal field of view."""
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


def box_depth(x, y, w, h, depth_m, rgb_shape, percentile=50, shrink=0.5, min_depth=0.2, max_depth=5.0):
    """Object depth (m) from its bounding box (center x, y, size w, h in RGB pixels).

    The box is rescaled to the depth image, shrunk around its center (boxes
    contain background), and the `percentile` of its valid depth pixels is
    returned. Returns 0.0 when no valid depth exists.
    """
    rgb_h, rgb_w = rgb_shape[:2]
    frame_h, frame_w = depth_m.shape[:2]
    sx, sy = frame_w / rgb_w, frame_h / rgb_h
    cx, cy = x * sx, y * sy
    half_w, half_h = max(1.0, w * sx * shrink / 2.0), max(1.0, h * sy * shrink / 2.0)
    v0, v1 = max(0, int(round(cy - half_h))), min(frame_h, int(round(cy + half_h)))
    u0, u1 = max(0, int(round(cx - half_w))), min(frame_w, int(round(cx + half_w)))
    if v0 >= v1 or u0 >= u1:
        return 0.0
    d = depth_m[v0:v1, u0:u1]
    d = d[np.isfinite(d) & (d >= min_depth) & (d <= max_depth)]
    return round(float(np.percentile(d, percentile)), 2) if d.size else 0.0


def pixel_to_optical(u, v, d, K):
    """Back-projects pixel (u, v) at depth d into the optical frame (+X right, +Y down, +Z forward)."""
    fx, fy, cx, cy = K
    return np.array([(u - cx) * d / fx, (v - cy) * d / fy, d])


def transform_point(p, translation, quaternion_xyzw):
    """Applies a TF transform (translation, quaternion x,y,z,w) to a point."""
    return Rotation.from_quat(quaternion_xyzw).apply(p) + np.asarray(translation)


def optical_to_map_from_robot_pose(p_opt, robot_x, robot_y, robot_yaw, camera_height=0.25):
    """Fallback optical -> map transform when the camera TF is unavailable
    (level camera at the robot center, looking along the robot heading)."""
    forward, left, up = p_opt[2], -p_opt[0], camera_height - p_opt[1]
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    return np.array([robot_x + forward * c - left * s, robot_y + forward * s + left * c, up])


def camera_pose_in_map(optical_to_map):
    """Camera (x, y) and optical-axis yaw in the map frame from an optical -> map transform."""
    origin = np.asarray(optical_to_map(np.zeros(3)), dtype=float)
    ahead = np.asarray(optical_to_map(np.array([0.0, 0.0, 1.0])), dtype=float)   # 1 m along the optical axis
    return (origin[0], origin[1]), math.atan2(ahead[1] - origin[1], ahead[0] - origin[0])


# ==========================
# YOLO detection
# ==========================

def detect_objects(model, rgb_bgr, depth_m, K, optical_to_map, conf_thr=0.3, exclude_names=(),
                   depth_percentile=50, depth_range=(0.2, 5.0), logger=print):
    """Runs YOLO on one frame and projects every detection into the map frame.

    Parameters
    ----------
    optical_to_map : callable
        Maps an optical-frame point (np.ndarray of 3) to a map-frame point (x, y, z).

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        The annotated frame (BGR) and the detections: dicts with objectType,
        Conf, map_x, map_y, map_z, depth (m) and radius (m, half the metric
        bounding box width, used as the Dirichlet footprint radius).
    """
    results = model(rgb_bgr, verbose=False, conf=conf_thr)
    annotated = results[0].plot()
    detections = []

    for box in results[0].boxes:
        confidence = box.conf[0].item()
        name = model.names[int(box.cls[0].item())]
        if confidence <= conf_thr or name in exclude_names:
            continue

        # Box center + size (pixels) -> depth of the object -> 3D point in the map frame
        x, y, w, h = (round(v.item()) for v in box.xywh[0])
        d = box_depth(x, y, w, h, depth_m, rgb_bgr.shape, percentile=depth_percentile,
                      min_depth=depth_range[0], max_depth=depth_range[1])
        if d <= 0:
            continue   # no valid depth in the box (too close / too far / invalid)
        map_x, map_y, map_z = (float(v) for v in optical_to_map(pixel_to_optical(x, y, d, K)))
        radius = (w / 2.0) * d / K[0]   # half the box width in meters at that depth
        detections.append(dict(objectType=name, Conf=confidence, map_x=map_x, map_y=map_y,
                               map_z=map_z, depth=d, radius=radius))
        logger(f"Detected {name} ({confidence:.2f}) at depth {d:.2f} m, radius {radius:.2f} m "
               f"-> map ({map_x:.2f}, {map_y:.2f})")
        cv2.putText(annotated, f"{d:.2f}m ({map_x:.2f}, {map_y:.2f})",
                    (max(0, x - w // 2), min(annotated.shape[0] - 5, y + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return annotated, detections


# ==========================
# Search grid / boundary
# ==========================

@dataclass
class SearchGrid:
    """Grid of candidate source cells (z = map y); row 0 is the lowest z.

    `inside` (bool, (len(z_points), len(x_points))) marks the cells inside the
    search boundary; source probabilities are 0 everywhere else.
    """

    x_points: np.ndarray               # cell centres along map x
    z_points: np.ndarray               # cell centres along map y
    step: float                        # cell size (m)
    inside: np.ndarray

    def extent(self):
        """[x_min, x_max, z_min, z_max] of the cell edges, for imshow."""
        h = self.step / 2.0
        return [self.x_points[0] - h, self.x_points[-1] + h, self.z_points[0] - h, self.z_points[-1] + h]

    def nearest_cell(self, x, z):
        """(row, col) of the cell closest to map (x, z)."""
        return int(np.abs(self.z_points - z).argmin()), int(np.abs(self.x_points - x).argmin())

    def contains(self, x, z):
        """True if the cell nearest to map (x, z) is inside the boundary."""
        return bool(self.inside[self.nearest_cell(x, z)])

    def argmax_xz(self, prob_map):
        """Map (x, z) of the most likely cell of prob_map, and its (row, col)."""
        row, col = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        return np.array([self.x_points[col], self.z_points[row]]), (row, col)


def points_in_polygon(xs, zs, polygon, margin=0.0):
    """True where (xs, zs) is inside `polygon` [(x, y), ...] or within `margin` (m) of its edges."""
    pts = np.column_stack([np.ravel(xs), np.ravel(zs)])
    poly = np.asarray(polygon, dtype=float)
    inside = Path(poly).contains_points(pts)
    if margin > 0:
        # Distance of every point to every edge a -> b (projection clipped to the segment)
        a, b = poly, np.roll(poly, -1, axis=0)
        ab = b - a
        t = np.clip(((pts[:, None, :] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
        dist = np.linalg.norm(pts[:, None, :] - (a + t[..., None] * ab), axis=-1).min(axis=1)
        inside |= dist <= margin
    return inside.reshape(np.shape(xs))


def grid_from_polygon(polygon, step=0.25, margin=0.0):
    """Search grid over the bounding box of `polygon`; a cell is inside if its centre is."""
    poly = np.asarray(polygon, dtype=float)
    # Cell centres on multiples of `step`, from the box's lower edge up to its upper edge
    x_points = np.round(np.arange(np.floor(poly[:, 0].min() / step) * step, poly[:, 0].max() + step / 2, step), 3)
    z_points = np.round(np.arange(np.floor(poly[:, 1].min() / step) * step, poly[:, 1].max() + step / 2, step), 3)
    Xc, Zc = np.meshgrid(x_points, z_points)
    return SearchGrid(x_points, z_points, step, points_in_polygon(Xc, Zc, poly, margin))


# ==========================
# Plots
# ==========================

def _minmax(m):
    m = np.asarray(m, dtype=float)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)


def draw_boundary(ax, polygon):
    """Dashed outline of the search boundary."""
    ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor='deepskyblue',
                         linewidth=1.5, linestyle='--', zorder=3))


def save_belief_maps(panels, grid, out_fname, trajectory=None, boundary=None):
    """Saves a row of belief maps (each min-max normalised, 'hot'), e.g. olfactory / visual / fused.

    panels : list[(map, title)]
    trajectory : list[(x, z)] robot path, drawn from start (lime) to now (cyan)
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharex=True, sharey=True, squeeze=False)
    extent = grid.extent()

    for i, (ax, (raw, title)) in enumerate(zip(axes[0], panels)):
        img = cv2.applyColorMap((_minmax(raw) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin='lower', extent=extent, aspect='equal')
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        if boundary is not None:
            draw_boundary(ax, boundary)
        if trajectory:
            tx, tz = zip(*trajectory)
            ax.plot(tx, tz, '-', color='cyan', linewidth=1.5, zorder=4)
            ax.plot(tx[0], tz[0], 'o', ms=8, mfc='lime', mec='black', zorder=5)     # start
            ax.plot(tx[-1], tz[-1], 'o', ms=8, mfc='cyan', mec='black', zorder=5)   # current
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # A colorbar axis on every panel keeps them the same size; only the last one is shown.
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        plt.colorbar(cm.ScalarMappable(cmap='hot', norm=mcolors.Normalize(vmin=0.0, vmax=1.0)), cax=cax)
        cax.set_visible(i == n - 1)

    plt.tight_layout()
    fig.savefig(out_fname, dpi=150)
    plt.close(fig)


def plot_tb4_trajectory(trajectory_log_path, grid, save_path, source_xz=None, estimate_xz=None,
                        plume_params=None, boundary=None):
    """Top-down trajectory plot: plume field (if the source is known), cells outside the
    boundary greyed out, start (lime), end (cyan), source (red star, 1 m circle) and
    final estimate (yellow X)."""
    df = pd.read_csv(trajectory_log_path)
    min_x, max_x, min_z, max_z = grid.extent()
    fig, ax = plt.subplots(figsize=(6, 6))

    if source_xz is not None:
        X, Z = np.meshgrid(np.linspace(min_x, max_x, 60), np.linspace(min_z, max_z, 60))
        ax.contourf(X, Z, gaussian_plume(X, Z, tuple(source_xz), **(plume_params or {})),
                    levels=60, cmap='magma_r', zorder=1)

    outside = np.zeros(grid.inside.shape + (4,))   # RGBA: grey, opaque where outside
    outside[..., :3] = 0.8
    outside[..., 3] = ~grid.inside
    ax.imshow(outside, extent=grid.extent(), origin='lower', zorder=2, alpha=0.5, aspect='auto')
    if boundary is not None:
        draw_boundary(ax, boundary)

    if not df.empty:
        ax.plot(df.robot_x, df.robot_z, color='purple', linewidth=1.5, zorder=4)
        ax.scatter(df.robot_x, df.robot_z, color='purple', s=30, zorder=5, edgecolors='black', linewidth=0.5)
        ax.scatter(df.robot_x.iloc[0], df.robot_z.iloc[0], color='lime', s=100, zorder=6, edgecolors='black')
        ax.scatter(df.robot_x.iloc[-1], df.robot_z.iloc[-1], color='cyan', s=100, zorder=6, edgecolors='black')
    if source_xz is not None:
        ax.scatter(*source_xz, color='red', s=200, marker='*', zorder=7, edgecolors='black')
        ax.add_patch(Circle(tuple(source_xz), radius=1.0, fill=False, edgecolor='red',
                            linewidth=1.5, linestyle='--', zorder=7))
    if estimate_xz is not None:
        ax.scatter(*estimate_xz, color='yellow', s=150, marker='X', zorder=8, edgecolors='black')

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('map x (m)')
    ax.set_ylabel('map y (m)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
