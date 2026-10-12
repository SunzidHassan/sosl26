"""Entry point for TurtleBot4 (ROS 2 Jazzy) sOSL experiments.

    ros2 run sosl26 sosl_tb4

Set the experiment parameters in main() below. Drive the robot with keyboard
teleop (e.g. `ros2 run teleop_twist_keyboard teleop_twist_keyboard`); the
controller only senses, fuses and logs. Stop with Ctrl+C or 'q' in the
YOLO window. Results are written to
    <sosl26>/save/save_F_{Odor}_dirichlet/{entropy_frac}/{run}/       (fusion)
    <sosl26>/save/save_O_{Odor}/{entropy_frac}/{run}/                 (olfactory-only)
"""

import os
import re
import threading
import time

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from tb4_base_controller import ExperimentConfig, TB4BaseController


def default_save_root():
    """<sosl26 source dir>/save when running from the source tree (colcon
    --symlink-install), otherwise ./save in the current working directory."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if os.path.isfile(os.path.join(pkg_root, "package.xml")):
        return os.path.join(pkg_root, "save")
    return os.path.join(os.getcwd(), "save")


def next_run_dir(base_save_dir):
    """Creates base_save_dir/<n>, n = 1 + highest existing run number. Returns (n, path)."""
    os.makedirs(base_save_dir, exist_ok=True)
    runs = [int(d) for d in os.listdir(base_save_dir)
            if d.isdigit() and os.path.isdir(os.path.join(base_save_dir, d))]
    run_serial = max(runs, default=0) + 1
    run_dir = os.path.join(base_save_dir, str(run_serial))
    os.makedirs(run_dir)
    return run_serial, run_dir


def resolve_concentration(value):
    """A number, or the path of an olfaction_data.csv whose mean chemicalConc (2 decimals) is used."""
    if value is None or isinstance(value, (int, float)):
        return value
    import pandas as pd
    # Relative paths are tried against the package source dir, then the current directory.
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
    # Every field and its default is documented in tb4_base_controller.ExperimentConfig.
    cfg = ExperimentConfig(
        alg_choice='F',                 # 'F' fusion (olfaction + Dirichlet vision) | 'O' olfactory-only
        odor="burnt",                   # goal phrase: "Is emitting burnt odor:"
        entropy_frac=0.8,               # search -> goal_navigation once H(olfactory map) < 0.8 * H_max
        sample_period=5.0,              # step window (s). Stop the robot; olfaction is averaged over it
        vision_period=0.5,              # 'F': process a new camera frame at most every 0.5 s
        step_threshold=20,              # number of steps in a run
        run_time_limit=None,            # seconds, None = until Ctrl+C / 'q'

        # Olfactory (Gaussian plume) model
        q_s=6.48,                       # emission rate
        D=0.004,                        # diffusion coefficient
        tau=1000.0,                     # decay time
        sigma_noise=100,                # std of the sensor noise, in rescaled units

        # Rescaling of raw /olfaction readings: min -> 0, max -> 100 (same as the offline a * raw + b).
        # Each can be a number or the path of an olfaction_data.csv, whose mean chemicalConc is used.
        olfaction_max_conc=resolve_concentration(610),
        # min: dummy value (measured: 'testData/2026-09-15_sensorDump_low/olfaction_data.csv')
        olfaction_min_conc=resolve_concentration(400),

        # Search area: corners (map x, y) in order around the boundary. The grid covers its
        # bounding box, and the source probability is 0 outside it.
        search_boundary=[
            (-7.5, -6.4),    # South East corner
            (-6.0, -1.0),    # Middle East
            (-3.0, -1.0),    # Center
            (0.0, -3.0),     # North East
            (0.0, 0.0),      # North West
            (-7.5, -0.7),    # South West
        ],
        boundary_margin=0.0,            # m; cells this close to the boundary also count as inside
        source_position=(-6.4, -0.5),   # ground-truth source (map x, y), for evaluation only (None = unknown)

        # Vision ('F')
        yolo_model_path="models/YOLO/neth234YOLO26m100epoch.pt",
        yolo_conf=0.3,                  # minimum YOLO box confidence
        yolo_exclude_classes=[],        # model classes to ignore

        # Dirichlet object map: YOLO26m validation confusion matrix (Five_Objects.yolo26, 88 val images;
        # normalized, rows predicted, columns true)
        dirichlet_conf_labels=['Cardboard box', 'Garbage can', 'Microwave', 'Refrigerator', 'Toaster', 'background'],
        dirichlet_conf_matrix=[
            [0.91, 0.00, 0.00, 0.00, 0.00, 0.12],   # pred Cardboard box
            [0.00, 1.00, 0.00, 0.00, 0.00, 0.10],   # pred Garbage can
            [0.04, 0.00, 1.00, 0.05, 0.00, 0.41],   # pred Microwave
            [0.00, 0.00, 0.00, 0.85, 0.04, 0.14],   # pred Refrigerator
            [0.00, 0.00, 0.00, 0.00, 0.96, 0.23],   # pred Toaster
            [0.04, 0.00, 0.00, 0.10, 0.00, 0.00],   # pred background (missed)
        ],
        dirichlet_bg_fp_rate=0.05,          # P(any false detection | background region)
        dirichlet_conf_temper=1.0,          # evidence weight = YOLO confidence * conf_temper
        dirichlet_bg_false_neg_rate=None,   # None -> 1 - mean recall (~0.05); try 0.15-0.2 for range / occlusion
        dirichlet_bg_dist_decay=0.5,        # Background evidence = (1 - eta) / (1 + 0.5 * distance)
        dirichlet_max_radius=3.0,           # largest detection footprint radius (m)

        # Ground-truth objects (map x, y) and plot colors
        object_positions={
            'Garbage can':   (-1.5, -0.25),
            'Toaster':       (-4.25, -0.4),
            'Microwave':     (-6.4, -0.5),
            'Refrigerator':  (-7.3, -3.1),
            'Cardboard box': (-4.3, -5.2),
        },
        object_colors={
            'Garbage can': 'green', 'Toaster': 'red', 'Microwave': 'black',
            'Refrigerator': 'blue', 'Cardboard box': 'saddlebrown', 'Background': 'cyan',
        },
    )
    save_root = default_save_root()
    # --------------------------------------------------------

    if cfg.alg_choice not in ('F', 'O'):
        raise ValueError(f"alg_choice must be 'F' or 'O', got '{cfg.alg_choice}'")

    # save/save_F_Burnt_dirichlet/0.8/<run>/ or save/save_O_Burnt/0.8/<run>/
    odor_tag = re.sub(r"[^A-Za-z0-9]+", "", cfg.odor.title())
    alg_folder = f"save_{cfg.alg_choice}_{odor_tag}" + ("_dirichlet" if cfg.alg_choice == 'F' else "")
    run_serial, save_dir = next_run_dir(os.path.join(save_root, alg_folder, str(cfg.entropy_frac)))
    print(f"--- STARTING TB4 RUN {run_serial} --- Saving files to: {save_dir}")

    rclpy.init(args=args)
    if cfg.alg_choice == 'F':
        from tb4_fusion_controller import TB4FusionController   # loads YOLO + the sentence model
        node = TB4FusionController(cfg, save_dir)
    else:
        node = TB4BaseController(cfg, save_dir)                   # the base controller is olfactory-only

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
        node.finalize()                       # end-of-run files, even after Ctrl+C
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
