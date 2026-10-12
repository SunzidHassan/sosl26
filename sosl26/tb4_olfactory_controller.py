"""Olfactory-only ('O') TurtleBot4 controller.

No camera, no YOLO. Each step only the Bayesian olfactory map is updated and
saved (maps_all_*.png with a single panel, maps_XXX.npz), and the source
estimate is the arg-max cell of that map.
"""

import numpy as np

from tb4_base_controller import TB4BaseController


class TB4OlfactoryController(TB4BaseController):

    uses_camera = False

    def __init__(self, cfg, save_dir):
        super().__init__(cfg, save_dir, node_name="sosl_tb4_olfactory_controller")

    def perceive(self, pose, step_count, srcProbGivenOlfactory):
        target_xz, _ = self.argmax_xz(srcProbGivenOlfactory)
        return dict(
            panels=[],
            arrays={},
            target_object='N/A',
            target_xz=target_xz,
            visual_entropy=np.nan,
            fused_entropy=np.nan,
        )
