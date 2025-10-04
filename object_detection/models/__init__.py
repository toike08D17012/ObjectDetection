from pathlib import Path

from ultralytics import YOLO
from ultralytics import settings as yolo_settings

# Specify save path regarging to YOLO model
MODELS_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = MODELS_ROOT
REPO_ROOT = SCRIPT_ROOT

WEIGHT_PATH = REPO_ROOT / "models"