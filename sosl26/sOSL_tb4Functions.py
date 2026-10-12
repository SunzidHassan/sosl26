"""ROS-free helpers for running the sOSL pipeline on a TurtleBot4.

The AI2-THOR code reads everything from an ai2thor `controller`. The functions
here do the same jobs from plain TB4 sensor data, so the olfaction code
(sOSL_olfactionFunctions.BayesianAgent, gaussian_plume) is reused as is.
Nothing here imports rclpy, so everything can be tested offline.

Sections
--------
  Pose / frame helpers   yaw conversions, Position strings
  Grid / map helpers     world <-> grid cell, map entropy
  Image decoding         CompressedImage / compressedDepth -> numpy arrays
  Camera model           intrinsics, box depth, pixel -> optical frame -> map frame
  Olfaction              TB4BayesianAgent: BayesianAgent with configurable plume parameters
  Search grid            grid cells + reachable mask from /map (or fixed bounds)
  Maps + plots           belief-map figure, trajectory plot

Coordinate convention
---------------------
The AI2-THOR pipeline works on the (x, z) ground plane with y pointing up and
stores object positions as the string "x, y, z". To reuse those functions
unchanged, ROS `map` coordinates are stored as:

    ai2thor x  <-  map x
    ai2thor y  <-  map z   (height)
    ai2thor z  <-  map y

So `robot_z`, `z_points` and the third value of a "Position" string are all
the ROS map *y* coordinate.

Handedness: AI2-THOR/Unity is left-handed (x right, y up, z forward), ROS
(REP 103) is right-handed (x forward, y left, z up). Swapping y and z as above
*is* the right-to-left-handed conversion, so positions map 1:1 and top-down
plots (x right, map y up, origin='lower') look the same. Only angles differ:

    ROS yaw      : counter-clockwise from +x      (atan2(dy, dx))
    ai2thor yaw  : clockwise from +z (= map +y)   (atan2(dx, dz), rotation.y)
    ai2thor_yaw_deg = (90 - ros_yaw_deg) mod 360

Use the ROS convention for anything sent to the robot (Nav2 goals, cmd_vel);
ros_yaw_to_ai2thor_deg() is only for logs compared with the AI2-THOR runs.
"""

import math
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

# Same title size as ControlAlgorithms/Fusion/fusion_controller.py
TITLE_FONTSIZE = 40


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
    """Formats a ROS map-frame point as an ai2thor style "x, y, z" string.

    Note the order: map y goes last (ai2thor z) and the height in the middle.
    """
    return f"{map_x:.2f}, {map_z:.2f}, {map_y:.2f}"


# ==========================
# Grid / Map Helpers (copied unchanged from the AI2-THOR sOSL_utils.py / sOSL_loggerFunctions.py)
# ==========================

def parse_position_string(pos_str):
    """Converts a position string "x, y, z" into a NumPy array.

    Parameters
    ----------
    pos_str : str
        A string containing three comma-separated float values (e.g., "1.0, 0.9, -2.5").

    Returns
    -------
    np.ndarray
        A NumPy array of shape (3,) containing the [x, y, z] coordinates as floats.
    """
    # Expecting a string like "x, y, z"
    return np.array([float(val.strip()) for val in pos_str.split(',')])


def world_to_grid(x, z, x_points, z_points):
    """Converts world coordinates (x, z) to the nearest grid cell indices.

    Finds the indices corresponding to the closest values in the `x_points`
    and `z_points` arrays.

    Parameters
    ----------
    x : float
        The world x-coordinate.
    z : float
        The world z-coordinate.
    x_points : np.ndarray
        A 1D array of the x-coordinates defining the grid columns.
    z_points : np.ndarray
        A 1D array of the z-coordinates defining the grid rows.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (2,) containing the [row_index, column_index].
    """
    col = (np.abs(x_points - x)).argmin()
    row = (np.abs(z_points - z)).argmin()
    return np.array([row, col])


def grid_to_world(pos, x_points, z_points):
    """Converts grid cell indices [row, col] back to world coordinates (x, z).

    Parameters
    ----------
    pos : np.ndarray or list-like
        The grid cell indices [row_index, column_index].
    x_points : np.ndarray
        A 1D array of the x-coordinates defining the grid columns.
    z_points : np.ndarray
        A 1D array of the z-coordinates defining the grid rows.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (2,) containing the world [x, z] coordinates
        corresponding to the center of the grid cell.
    """
    row, col = pos
    return np.array([x_points[col], z_points[row]])


def map_entropy(raw):
    """Calculates the Shannon entropy of a raw (unnormalized) 2D map.

    Flattens the map, normalizes it to sum to 1, removes zero probabilities,
    and then calculates the entropy: -sum(p * log2(p)).

    Parameters
    ----------
    raw : np.ndarray
        The 2D input map (e.g., a heatmap or probability map). Values should be non-negative.

    Returns
    -------
    float
        The calculated entropy in bits. Returns 0 if the map sum is zero.
    """
    p = raw.flatten()
    map_sum = p.sum()

    if map_sum < 1e-12: # Check if map sum is effectively zero
        return 0.0 # Entropy of an empty/zero distribution is 0

    p = p / map_sum     # normalize
    # p = scaler.fit_transform(p)  # Scale to [0, 1]
    p = p[p > 1e-12]    # drop zeros/near-zeros to avoid log2(0) issues

    if p.size == 0: # Check if only zeros were present
        return 0.0

    return -np.sum(p * np.log2(p))


# --- FUNCTION TO GENERATE TRAJECTORY PLOT ---


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
    # Skip the header by locating the PNG signature (more robust than a fixed offset)
    png_start = raw.find(b'\x89PNG')
    if png_start < 0:
        png_start = 12
    img = cv2.imdecode(np.frombuffer(raw[png_start:], np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if '32FC1' in fmt and png_start >= 12:
        # Float depth is stored as quantised inverse depth: depth = A / (value - B)
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

    TB4 version of the AI2-THOR boxDepth. The RGB bounding box is
    rescaled to the depth image resolution, shrunk around its center (real
    boxes contain a lot of background), and the requested percentile of the
    valid depth pixels is returned. Returns 0.0 when no valid depth exists.
    """
    if depth_m is None:
        return 0.0
    # The box comes from the RGB image; the depth image can have another resolution
    # (it must be aligned to the RGB camera, which the TB4 OAK-D driver does).
    rgb_h, rgb_w = rgb_shape[:2]
    frame_h, frame_w = depth_m.shape[:2]
    sx, sy = frame_w / rgb_w, frame_h / rgb_h

    # Keep only the central `shrink` fraction of the box, where the object is
    cx, cy = x * sx, y * sy
    half_w, half_h = max(1.0, w * sx * shrink / 2.0), max(1.0, h * sy * shrink / 2.0)
    vMin = max(0, int(round(cy - half_h)))
    vMax = min(frame_h, int(round(cy + half_h)))
    hMin = max(0, int(round(cx - half_w)))
    hMax = min(frame_w, int(round(cx + half_w)))
    if vMin >= vMax or hMin >= hMax:
        return 0.0

    # Ignore invalid pixels (0 / NaN) and readings outside the trusted range
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
    # Pinhole model: a pixel offset from the image centre, scaled by depth / focal length
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
    # Optical axes -> robot axes: forward = +Z, left = -X, up = -Y
    forward = p_opt[2] + camera_forward
    left = -p_opt[0]
    up = camera_height - p_opt[1]
    # Robot axes -> map: rotate by the robot yaw and add its position
    c, s = math.cos(robot_yaw), math.sin(robot_yaw)
    return np.array([robot_x + forward * c - left * s,
                     robot_y + forward * s + left * c,
                     up])


def camera_pose_in_map(optical_to_map):
    """Camera (x, y) and optical-axis yaw in the map frame from an optical->map transform.

    The camera sits at the optical origin and looks along optical +Z, so mapping
    those two points gives its position and heading.
    """
    origin = np.asarray(optical_to_map(np.zeros(3)), dtype=float)
    ahead = np.asarray(optical_to_map(np.array([0.0, 0.0, 1.0])), dtype=float)
    return (origin[0], origin[1]), math.atan2(ahead[1] - origin[1], ahead[0] - origin[0])


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
        """Same Bayes update as BayesianAgent.posterior, with self.plume_params.

        For every cell: "if the source were here, what would the robot read?"
        (plume model). Cells whose prediction is close to the actual reading
        (within ~sigma_noise) gain probability.
        """
        likelihood_map = np.zeros_like(self.prob_map)
        for iz, z_source in enumerate(self.z_points):
            for ix, x_source in enumerate(self.x_points):
                expected = self.expected_concentration(robot_x, robot_z, (x_source, z_source))
                likelihood_map[iz, ix] = self.likelihood(current_odor_concentration, expected, self.sigma_noise)

        # posterior ∝ prior * likelihood, renormalised to sum to 1
        self.prob_map *= likelihood_map

        map_sum = np.sum(self.prob_map)
        if map_sum > 1e-9:
            self.prob_map /= map_sum
        else:
            # Every cell was ruled out (readings inconsistent with the model): restart from uniform
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
    """Cell centres from lo to hi, snapped to multiples of `step` (as sOSL_main.py does)."""
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

    # Bounding box of the known cells, plus a margin
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

    # Free = known and occupancy probability below free_thresh (%)
    free = (crop >= 0) & (crop < free_thresh)

    x_points = _grid_points(min_x, max_x, step)
    z_points = _grid_points(min_z, max_z, step)

    # A grid cell is reachable if the map cell under its centre is free
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

def _minmax(m):
    """Rescales a map to [0, 1] for display (an all-constant map becomes all 0)."""
    m = np.asarray(m, dtype=float)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)


def save_belief_maps(panels, x_points, z_points, out_fname, trajectory=None):
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
        if trajectory is not None and len(trajectory) > 0:
            tx, tz = zip(*trajectory)
            ax.plot(tx, tz, '-', color='cyan', linewidth=1.5, zorder=4)
            ax.plot(tx[0], tz[0], 'o', ms=8, mfc='lime', mec='black', zorder=5)     # start
            ax.plot(tx[-1], tz[-1], 'o', ms=8, mfc='cyan', mec='black', zorder=5)   # current    

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

    TB4 counterpart of the AI2-THOR generate_trajectory_plot (which
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
