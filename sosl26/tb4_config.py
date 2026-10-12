"""Experiment parameters for the TurtleBot4 sOSL runs.

Every setting lives in ExperimentConfig, with its default and what it does.
sOSL_tb4_main.main() creates one ExperimentConfig, overriding the values for
the current experiment; anything not set there keeps the default below.

Coordinates are in the ROS `map` frame (x, y). Internally the AI2-THOR
functions call map y "z" (see sOSL_tb4Functions.py).
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import pandas as pd


@dataclass
class ExperimentConfig:

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
    alg_choice: str = 'F'                # 'F' = fusion (olfaction + Dirichlet object map), 'O' = olfactory-only
    odor: str = "burnt"                  # used in the goal phrase and the save folder name
    goal_phrase: Optional[str] = None    # text compared with class names; None -> "Is emitting {odor} odor:"
    entropy_frac: float = 0.8            # behaviour switches to "goal_navigation" once the olfactory
                                         # entropy drops below entropy_frac * (entropy of a uniform map)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    sample_period: float = 5.0           # length of one step (s). Stop the robot: olfaction is averaged over
                                         # this window and camera frames are accumulated during it
    vision_period: float = 0.5           # 'F': run YOLO on a new camera frame at most every vision_period s
    step_threshold: int = 100            # the run ends after this many steps ...
    run_time_limit: Optional[float] = None  # ... or after this many seconds (None = no limit)

    # ------------------------------------------------------------------
    # Olfaction: Gaussian plume model used by the Bayesian source map
    # ------------------------------------------------------------------
    q_s: float = 2000.0                  # source emission rate
    D: float = 10.0                      # diffusion coefficient
    tau: float = 1000.0                  # odor lifetime
    U: float = 0.0                       # wind speed (0 = isotropic plume, as in the AI2-THOR runs)
    psi_deg: float = 0.0                 # wind direction (deg), only used if U > 0
    sigma_noise: float = 1.5             # sensor noise std in the likelihood, same units as the rescaled reading

    # Raw /olfaction readings are rescaled before the update, as in the offline mapping:
    #   rescaled = olfaction_rescale_to * (raw - olfaction_min_conc) / (olfaction_max_conc - olfaction_min_conc)
    # Not clipped (readings below min become negative). Leave both None to use raw readings.
    olfaction_min_conc: Optional[float] = None
    olfaction_max_conc: Optional[float] = None
    olfaction_rescale_to: float = 100.0

    use_simulated_olfaction: bool = False  # testing only: replace /olfaction by the plume model at source_position

    # ------------------------------------------------------------------
    # Map / search grid
    # ------------------------------------------------------------------
    grid_step: float = 0.25              # cell size (m) of the olfactory, visual and fused maps
    map_bounds: Optional[Tuple[float, float, float, float]] = None  # (x_min, x_max, y_min, y_max); None = from /map
    map_wait_timeout: float = 10.0       # seconds to wait for /map before falling back to ...
    fallback_map_size: float = 10.0      # ... a square of this size (m) around the start pose
    source_position: Optional[Tuple[float, float]] = None  # true source (x, y) if known; only used for evaluation

    # ------------------------------------------------------------------
    # Vision ('F' only)
    # ------------------------------------------------------------------
    yolo_conf: float = 0.3               # minimum YOLO confidence
    yolo_target_classes: Optional[Sequence[str]] = None  # only keep these classes (None = all model classes)
    yolo_exclude_classes: Sequence[str] = field(default_factory=list)  # always drop these classes
    depth_percentile: float = 50.0       # percentile of the depth pixels inside a box used as the object's depth
    depth_range: Tuple[float, float] = (0.2, 5.0)  # valid depth (m); also the max range of background evidence
    camera_hfov_deg: float = 69.0        # only used until /oakd/rgb/camera_info arrives
    camera_height: float = 0.25          # only used if the camera TF frame is unavailable
    save_vision_frames: bool = True      # save the annotated YOLO image of every processed frame

    # Dirichlet object map, see sOSL_dirichletFunctions.py
    dirichlet_object_evidence: float = 1.0      # added to an object's class, per frame, in its footprint cells
    dirichlet_background_evidence: float = 0.5  # added to Background, per frame, in empty cells camera -> object
    dirichlet_prior_strength: float = 1.0       # uniform prior: each class starts at prior_strength / K
    dirichlet_min_radius: Optional[float] = None  # footprint radius limits (m); None = half a grid cell
    dirichlet_max_radius: float = 1.0
    dirichlet_background_similarity: float = 0.0  # similarity of Background to the goal (empty space has no odor)

    # ------------------------------------------------------------------
    # ROS topics and frames
    # ------------------------------------------------------------------
    rgb_topic: str = "/oakd/rgb/image_raw/compressed"
    depth_topic: str = "/oakd/stereo/image_raw/compressedDepth"
    camera_info_topic: str = "/oakd/rgb/camera_info"
    olfaction_topic: str = "/olfaction"  # geometry_msgs/Vector3: x wind direction, y wind speed, z concentration
    map_topic: str = "/map"
    cmd_vel_topic: str = "/cmd_vel"      # reserved for navigate(); nothing is published yet
    map_frame: str = "map"
    base_frame: str = "base_footprint"
    camera_frame: str = ""               # "" = use the frame_id of the RGB images

    show_window: bool = True             # 'F': show the annotated YOLO image ('q' ends the run)

    def resolved_goal_phrase(self):
        return self.goal_phrase or f"Is emitting {self.odor} odor:"

    def validate(self):
        if self.alg_choice not in ('F', 'O'):
            raise ValueError(f"alg_choice must be 'F' or 'O', got '{self.alg_choice}'")
        if (self.olfaction_min_conc is None) != (self.olfaction_max_conc is None):
            raise ValueError("Set both olfaction_min_conc and olfaction_max_conc, or neither.")
        if self.olfaction_min_conc is not None and self.olfaction_max_conc == self.olfaction_min_conc:
            raise ValueError("olfaction_max_conc must differ from olfaction_min_conc.")


def mean_concentration(csv_path):
    """Mean `chemicalConc` of an olfaction_data.csv (rounded to 2 decimals, as in the offline notebook).

    Use it in main() to set olfaction_min_conc / olfaction_max_conc from a recording, e.g.
        olfaction_max_conc=mean_concentration('/path/to/sensorDump_high/olfaction_data.csv')
    """
    return round(float(pd.read_csv(csv_path)['chemicalConc'].mean()), 2)
