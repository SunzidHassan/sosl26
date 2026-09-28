"""Olfactory-only ('O') TurtleBot4 controller.

TB4 counterpart of ControlAlgorithms/OlfactoryOnly/olfactory_controller.py:
no camera, no YOLO. Each step only the Bayesian olfactory map is updated and
saved (maps_all_*.png with a single panel, maps_XXX.npz), and the source
estimate is the arg-max cell of that map.
"""

import numpy as np

from sOSL_tb4Functions import format_position
from sOSL_utils import grid_to_world
from sosl26.tb4_base_controller import TB4BaseController


class TB4OlfactoryController(TB4BaseController):

    uses_camera = False

    def __init__(self, cfg, save_dir):
        super().__init__(cfg, save_dir, node_name="sosl_tb4_olfactory_controller")

    def perceive(self, pose, step_count, srcProbGivenOlfactory):
        g = self.grid
        max_idx = np.unravel_index(np.argmax(srcProbGivenOlfactory), srcProbGivenOlfactory.shape)
        x, z = grid_to_world(max_idx, g.x_points, g.z_points)
        return dict(
            panels=[],
            arrays={},
            target_object='N/A',
            target_coordinate=format_position(x, z, 0.0),
            target_xz=np.array([x, z]),
            visual_entropy=np.nan,
            fused_entropy=np.nan,
        )
