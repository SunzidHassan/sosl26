"""Navigation planning on a fixed set of robot poses (no ROS, testable offline).

The robot only stops at `reachable cells` (map x, y) and faces one of a few
`headings` (ROS yaw, rad, CCW from map +x). Each step:

  * next cell   : among the current cell and its adjacent reachable cells,
                  the one closest to the current source estimate,
  * heading     : chosen by the controller (fusion: the heading whose camera
                  view covers the most visual entropy; olfactory-only: the
                  heading facing the source estimate).

The first step is a scan: the robot drives to its initial position and turns
to every heading there.
"""

import math

import numpy as np


def yaw_from_wz(w, z):
    """Yaw (rad) of a quaternion rotating only about the vertical axis, given as (w, z)."""
    return math.atan2(2.0 * w * z, w * w - z * z)    # = 2 * atan2(z, w), normalised to (-pi, pi]


def wz_from_yaw(yaw):
    """Unit quaternion (w, z) for a rotation of `yaw` rad about the vertical axis."""
    return math.cos(yaw / 2.0), math.sin(yaw / 2.0)


def angle_diff(a, b):
    """Smallest signed difference a - b between two angles (rad), in [-pi, pi)."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def nearest_cell(cells, xy):
    """Index of the reachable cell closest to map point xy."""
    return int(np.argmin(np.hypot(cells[:, 0] - xy[0], cells[:, 1] - xy[1])))


def adjacent_cells(cells, idx, radius):
    """Indices of the reachable cells within `radius` (m) of cell idx, including idx itself.

    With 1 m spacing and radius 1.5 m this is the 8-neighbourhood (diagonals included).
    """
    d = np.hypot(cells[:, 0] - cells[idx, 0], cells[:, 1] - cells[idx, 1])
    return np.flatnonzero(d <= radius)


def next_cell(cells, idx, target_xy, radius):
    """Adjacent cell (or the current one) that brings the robot closest to target_xy.

    If no adjacent cell is closer than the current one, the robot stays and only turns.
    """
    candidates = adjacent_cells(cells, idx, radius)
    d = np.hypot(cells[candidates, 0] - target_xy[0], cells[candidates, 1] - target_xy[1])
    # Ties: prefer staying (current cell), then the order of the list
    best = candidates[np.flatnonzero(np.isclose(d, d.min()))]
    return int(idx) if idx in best else int(best[0])


def heading_towards(headings, from_xy, to_xy, default=None):
    """The heading (from `headings`) closest to the bearing from from_xy to to_xy.

    Returns `default` (or headings[0]) if the two points coincide.
    """
    dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
    if math.hypot(dx, dy) < 1e-6:
        return headings[0] if default is None else default
    bearing = math.atan2(dy, dx)
    return min(headings, key=lambda h: abs(angle_diff(h, bearing)))
