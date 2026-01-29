import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models" / "checkpoints"

MAX_FILE_SIZE = 200 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8912
MIN_IMAGE_DIMENSION = 16

DEFAULT_NST_PARAMS = {
    "max_size": 512,
    "num_steps": 300,
    "content_weight": 1.0,
    "style_weight": 1e5,
}

DEVICE = "cuda" if os.getenv("FORCE_CPU") != "1" else "cpu"
