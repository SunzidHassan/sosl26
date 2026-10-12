"""Entry point for the TurtleBot4 (ROS 2 Jazzy) sOSL experiments.

    ros2 run sosl26 sosl_tb4          (or: python3 -m sosl26.sOSL_tb4_main)

Set the experiment in main() below; every option and its default is explained in
tb4_config.py. Drive the robot with keyboard teleop
(`ros2 run teleop_twist_keyboard teleop_twist_keyboard`) and stop it during each
step window. End the run with Ctrl+C or 'q' in the YOLO window. Results go to
    <sosl26>/save/save_{F|O}_{Odor}/{entropy_frac}/{run}/
"""

import os
import random
import re

import numpy as np
import rclpy

from sosl26.tb4_config import ExperimentConfig, mean_concentration  # noqa: F401 (mean_concentration: see below)
from sosl26.tb4_controller import TB4Controller, spin_with_ui

# Every class of the pretrained COCO model except microwave, toaster and refrigerator
COCO_EXCLUDE = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "oven", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

YOLO_MODEL_PATH = "yolo26m.pt"   # pretrained YOLO26m (ultralytics downloads it on first use)


def main(args=None):
    # ===================== Experiment =====================
    cfg = ExperimentConfig(
        alg_choice='O',                  # 'F' fusion (olfaction + Dirichlet object map) | 'O' olfactory-only
        odor="burnt",
        entropy_frac=0.8,

        # Timing
        sample_period=5.0,               # n: step window (s); olfaction is averaged over it
        vision_period=0.5,               # 'F': at most one YOLO frame every 0.5 s
        step_threshold=20,
        run_time_limit=None,             # seconds, None = until Ctrl+C / 'q'

        # Olfaction: plume model
        q_s=12.8,
        D=0.01,
        tau=1000.0,
        sigma_noise=60,                  # sensor noise std, in rescaled units

        # Olfaction: rescaling raw readings, min -> 0, max -> 100 (as in the offline a * raw + b).
        # Use a number, or the mean of a recording:
        #   olfaction_max_conc=mean_concentration('/path/to/sensorDump_high/olfaction_data.csv')
        olfaction_max_conc=500,
        olfaction_min_conc=350,

        # Map / evaluation (ROS map frame)
        map_bounds=None,                 # (x_min, x_max, y_min, y_max); None = use /map
        source_position=None,            # true source (x, y) if known

        # Vision
        yolo_conf=0.3,
        yolo_exclude_classes=COCO_EXCLUDE,
        dirichlet_object_evidence=1.0,      # per frame, per object footprint cell
        dirichlet_background_evidence=0.5,  # per frame, per empty cell between camera and object
        dirichlet_max_radius=5.0,           # footprint radius limit (m); radius = half the metric box width
    )
    # ======================================================

    save_dir, run_serial = make_run_dir(cfg)
    print(f"--- STARTING TB4 RUN {run_serial} --- Saving files to: {save_dir}")
    random.seed(run_serial)
    np.random.seed(run_serial)

    yolo_model = None
    if cfg.alg_choice == 'F':
        from ultralytics import YOLO      # only fusion runs need YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH)

    rclpy.init(args=args)
    spin_with_ui(TB4Controller(cfg, save_dir, yolo_model))
    print(f"--- COMPLETED TB4 RUN {run_serial} ---")


def make_run_dir(cfg):
    """Creates save/save_{alg}_{Odor}/{entropy_frac}/{n} with n = next free run number.

    `save` is in the sosl26 source folder when running from the source tree
    (colcon build --symlink-install), otherwise in the current directory.
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    save_root = os.path.join(pkg_root if os.path.isfile(os.path.join(pkg_root, "package.xml")) else os.getcwd(),
                             "save")

    odor_tag = re.sub(r"[^A-Za-z0-9]+", "", cfg.odor.title())        # "burnt" -> "Burnt"
    alg_folder = f"save_{cfg.alg_choice}_{odor_tag}"
    base = os.path.join(save_root, alg_folder, str(cfg.entropy_frac))
    os.makedirs(base, exist_ok=True)

    runs = [int(d) for d in os.listdir(base) if d.isdigit() and os.path.isdir(os.path.join(base, d))]
    run_serial = max(runs, default=0) + 1
    save_dir = os.path.join(base, str(run_serial))
    os.makedirs(save_dir)
    return save_dir, run_serial


if __name__ == "__main__":
    main()
