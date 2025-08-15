# utils/logging_setup.py
import os
import warnings
import logging

def quiet_logs():
    # Silence TensorFlow / TFLite / XNNPACK chatter
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 3=error
    os.environ.setdefault("GLOG_minloglevel", "3")

    # absl noise (used by MediaPipe)
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create a logger for the application
    logger = logging.getLogger(__name__)

    # cv2, matplotlib, other libs to WARNING level
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)

    # NumPy warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

    # General warnings filter
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
