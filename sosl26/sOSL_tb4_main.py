"""Entry point for TurtleBot4 (ROS 2 Jazzy) sOSL experiments.

    ros2 run sosl26 sosl_tb4

Set the experiment parameters in main() below. Drive the robot with keyboard
teleop (e.g. `ros2 run teleop_twist_keyboard teleop_twist_keyboard`); the
controller only senses, fuses and logs. Stop with Ctrl+C or 'q' in the
YOLO window: all results are written to
    <sosl26>/save/save_{alg}_{odor}/{entropy_frac}/{run}/
"""

import os
import random
import re

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from ultralytics import YOLO

from sosl26.tb4_fusion_controller import ExperimentConfig, TB4FusionController


def default_save_root():
    """<sosl26 source dir>/save when running from the source tree (colcon
    --symlink-install), otherwise ./save in the current working directory."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if os.path.isfile(os.path.join(pkg_root, "package.xml")):
        return os.path.join(pkg_root, "save")
    return os.path.join(os.getcwd(), "save")


def next_run_dir(base_save_dir):
    """Creates and returns base_save_dir/<n>, with n = 1 + highest existing run number."""
    os.makedirs(base_save_dir, exist_ok=True)
    runs = [int(d) for d in os.listdir(base_save_dir)
            if d.isdigit() and os.path.isdir(os.path.join(base_save_dir, d))]
    run_serial = max(runs, default=0) + 1
    run_dir = os.path.join(base_save_dir, str(run_serial))
    os.makedirs(run_dir)
    return run_serial, run_dir


# ==========================
# MAIN FUNCTION
# ==========================

def main(args=None):
    # ---------------- Experiment parameters ----------------
    alg_choice = 'F'                 # 'F' fusion ('O' olfaction-only / 'V' vision-only goalSim)
    odor = "burnt"
    entropy_frac = 0.8
    sample_period = 2.0              # n: seconds between olfactory + visual readings
    step_threshold = 100
    run_time_limit = None            # seconds, None = until Ctrl+C / 'q'

    # Olfactory (Gaussian plume) parameters
    q_s = 2000.0
    D = 10.0
    tau = 1000.0
    sigma_noise = 1.5

    # Map / ground truth (ROS map frame)
    map_bounds = None                # (x_min, x_max, y_min, y_max); None = use /map
    source_position = None           # ground-truth source (x, y) if known, for evaluation only

    # Vision
    yolo_model_path = "yolo26m.pt"   # pretrained YOLO26m (downloaded by ultralytics if missing)
    yolo_conf = 0.3
    yolo_exclude_classes = ["person"]

    save_root = default_save_root()
    # --------------------------------------------------------

    cfg = ExperimentConfig(
        odor=odor,
        alg_choice=alg_choice,
        entropy_frac=entropy_frac,
        sample_period=sample_period,
        step_threshold=step_threshold,
        run_time_limit=run_time_limit,
        q_s=q_s,
        D=D,
        tau=tau,
        sigma_noise=sigma_noise,
        map_bounds=map_bounds,
        source_position=source_position,
        yolo_conf=yolo_conf,
        yolo_exclude_classes=yolo_exclude_classes,
    )

    odor_tag = re.sub(r"[^A-Za-z0-9]+", "", odor.title())
    base_save_dir = os.path.join(save_root, f"save_{alg_choice}_{odor_tag}", str(entropy_frac))
    run_serial, save_dir = next_run_dir(base_save_dir)
    print(f"--- STARTING TB4 RUN {run_serial} --- Saving files to: {save_dir}")

    random.seed(run_serial)
    np.random.seed(run_serial)

    yolo_model = YOLO(yolo_model_path)

    rclpy.init(args=args)
    node = TB4FusionController(cfg, yolo_model, save_dir)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finalize()
        node.destroy_node()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if rclpy.ok():
            rclpy.shutdown()
    print(f"--- COMPLETED TB4 RUN {run_serial} ---")


if __name__ == "__main__":
    main()
