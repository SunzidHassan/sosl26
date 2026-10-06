"""Entry point for TurtleBot4 (ROS 2 Jazzy) sOSL experiments.

    ros2 run sosl26 sosl_tb4

Set the experiment parameters in main() below. Drive the robot with keyboard
teleop (e.g. `ros2 run teleop_twist_keyboard teleop_twist_keyboard`); the
controller only senses, fuses and logs. Stop with Ctrl+C or 'q' in the
YOLO window: all results are written to
    <sosl26>/save/save_F_{Odor}_{vision_mode}/{entropy_frac}/{run}/   (fusion)
    <sosl26>/save/save_O_{Odor}/{entropy_frac}/{run}/                 (olfactory-only)
"""

import os
import random
import re
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from tb4_base_controller import ExperimentConfig
# from sosl26.tb4_base_controller import ExperimentConfig


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


def resolve_concentration(value):
    """A number, or the mean chemicalConc (rounded to 2 decimals) of an olfaction_data.csv."""
    if value is None or isinstance(value, (int, float)):
        return value
    import pandas as pd
    candidates = [value] if os.path.isabs(value) else [
        os.path.join(os.path.dirname(default_save_root()), value), os.path.join(os.getcwd(), value)]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        raise FileNotFoundError(f"Concentration CSV not found (tried {candidates}); set a number or fix the path.")
    mean = round(float(pd.read_csv(path)['chemicalConc'].mean()), 2)
    print(f"Average concentration of {path}: {mean}")
    return mean


# ==========================
# MAIN FUNCTION
# ==========================

def main(args=None):
    # ---------------- Experiment parameters ----------------
    alg_choice = 'F'                 # 'F' fusion (olfaction + vision) | 'O' olfactory-only
    vision_mode = 'dirichlet'        # 'F' only: 'navKnowledge' (AI2-THOR object list) | 'dirichlet' (object map)
    odor = "burnt"
    entropy_frac = 0.8
    sample_period = 5.0              # n: step window (s). Stop the robot; olfaction is averaged over it
    vision_period = 0.5              # 'F': process a new camera frame at most every vision_period s
    step_threshold = 20
    run_time_limit = None            # seconds, None = until Ctrl+C / 'q'

    # Olfactory (Gaussian plume) parameters
    q_s = 8000
    D = 10
    tau = 1000.0
    sigma_noise = 200                 # std of the sensor noise, in rescaled units

    # Rescaling of raw /olfaction readings: min -> 0, max -> 100 (same as the offline a * raw + b).
    # Each can be a number or the path of an olfaction_data.csv, whose mean chemicalConc is used.
    olfaction_max_conc = 500
    olfaction_min_conc = 300         # dummy min (measured: 'testData/2026-09-15_sensorDump_low/olfaction_data.csv')

    # Map / ground truth (ROS map frame)
    map_bounds = None                # (x_min, x_max, y_min, y_max); None = use /map
    source_position = None           # ground-truth source (x, y) if known, for evaluation only

    # Vision
    yolo_model_path = "models/YOLO/neth234YOLO26m100epoch.pt"
    yolo_conf = 0.3
    yolo_exclude_classes = []
    # yolo_exclude_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    # "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    # "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    # "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    # "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    # "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    # "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "oven",
    # "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # Dirichlet object map (vision_mode 'dirichlet')
    # Dirichlet object map (vision_mode 'dirichlet'): YOLO validation metrics (Five_Objects.yolo26, 88 val images)
    # Dirichlet object map: YOLO26m validation confusion matrix (normalized; rows predicted, columns true)
    dirichlet_conf_labels = ['Cardboard box', 'Garbage can', 'Microwave', 'Refrigerator', 'Toaster', 'background']
    dirichlet_conf_matrix = [
        [0.91, 0.00, 0.00, 0.00, 0.00, 0.12],   # pred Cardboard box
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.10],   # pred Garbage can
        [0.04, 0.00, 1.00, 0.05, 0.00, 0.41],   # pred Microwave
        [0.00, 0.00, 0.00, 0.85, 0.04, 0.14],   # pred Refrigerator
        [0.00, 0.00, 0.00, 0.00, 0.96, 0.23],   # pred Toaster
        [0.04, 0.00, 0.00, 0.10, 0.00, 0.00],   # pred background (missed)
    ]
    dirichlet_bg_fp_rate = 0.05
    dirichlet_conf_temper = 1.0
    dirichlet_bg_false_neg_rate = None   # None -> 1 - mean recall (~0.05); try 0.15-0.2 for real range / occlusion
    dirichlet_bg_dist_decay = 0.5
    dirichlet_max_radius = 3.0

    # Ground-truth objects (map frame x, y) and plot colors
    object_positions = {
        'Garbage can':   (-1.5, -0.25),
        'Toaster':       (-4.25, -0.4),
        'Microwave':     (-6.4, -0.5),
        'Refrigerator':  (-7.3, -3.1),
        'Cardboard box': (-4.3, -5.2),
    }
    object_colors = {
        'Garbage can': 'green', 'Toaster': 'red', 'Microwave': 'black',
        'Refrigerator': 'blue', 'Cardboard box': 'saddlebrown', 'Background': 'cyan',
    }

    save_root = default_save_root()
    # --------------------------------------------------------

    cfg = ExperimentConfig(
        odor=odor,
        alg_choice=alg_choice,
        vision_mode=vision_mode,
        entropy_frac=entropy_frac,
        sample_period=sample_period,
        vision_period=vision_period,
        step_threshold=step_threshold,
        run_time_limit=run_time_limit,
        q_s=q_s,
        D=D,
        tau=tau,
        sigma_noise=sigma_noise,
        olfaction_min_conc=resolve_concentration(olfaction_min_conc),
        olfaction_max_conc=resolve_concentration(olfaction_max_conc),
        map_bounds=map_bounds,
        source_position=source_position,
        yolo_conf=yolo_conf,
        yolo_exclude_classes=yolo_exclude_classes,
        dirichlet_conf_matrix=dirichlet_conf_matrix,
        dirichlet_conf_labels=dirichlet_conf_labels,
        dirichlet_bg_fp_rate=dirichlet_bg_fp_rate,
        dirichlet_conf_temper=dirichlet_conf_temper,
        dirichlet_bg_false_neg_rate=dirichlet_bg_false_neg_rate,
        dirichlet_bg_dist_decay=dirichlet_bg_dist_decay,
        dirichlet_max_radius=dirichlet_max_radius,
        object_positions=object_positions,
        object_colors=object_colors,
    )
    if alg_choice not in ('F', 'O'):
        raise ValueError(f"alg_choice must be 'F' or 'O', got '{alg_choice}'")

    odor_tag = re.sub(r"[^A-Za-z0-9]+", "", odor.title())
    alg_folder = f"save_{alg_choice}_{odor_tag}" + (f"_{vision_mode}" if alg_choice == 'F' else "")
    base_save_dir = os.path.join(save_root, alg_folder, str(entropy_frac))
    run_serial, save_dir = next_run_dir(base_save_dir)
    print(f"--- STARTING TB4 RUN {run_serial} --- Saving files to: {save_dir}")

    random.seed(run_serial)
    np.random.seed(run_serial)

    rclpy.init(args=args)
    if alg_choice == 'F':
        from ultralytics import YOLO
        from tb4_fusion_controller import TB4FusionController
        # from sosl26.tb4_fusion_controller import TB4FusionController
        node = TB4FusionController(cfg, YOLO(yolo_model_path), save_dir)
    else:
        from tb4_olfactory_controller import TB4OlfactoryController
        # from sosl26.tb4_olfactory_controller import TB4OlfactoryController
        node = TB4OlfactoryController(cfg, save_dir)
    # Sensor callbacks and processing (steps / YOLO) run in parallel threads;
    # the OpenCV window is driven from this (main) thread.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:   # Ctrl+C shuts the context down
            pass

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok() and not node.done and spin_thread.is_alive():
            node.ui_update()
            time.sleep(0.03)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown(timeout_sec=30.0)   # waits for a running step / frame to finish
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
