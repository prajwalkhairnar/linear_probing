# config.py

import os
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(ROOT_DIR / ".env")

# Device setup for Apple Silicon
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
DTYPE = torch.float32  # float16 unstable on MPS, float32 is safe

MODEL_NAME = "google/gemma-2-2b-it"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths
DATA_DIR = "data/"
ACTIVATIONS_DIR = "activations/"
MODELS_DIR = "models/"
OUTPUTS_DIR = "outputs/"

# Dataset
N_CANDIDATES = 150
MAX_NEW_TOKENS = 200

# Probe
CV_FOLDS = 5

# Steering
STEERING_SCALES = [0.01, 0.05, 0.1]
N_STEERING_TEST = 20