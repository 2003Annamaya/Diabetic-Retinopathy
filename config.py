"""
Diabetic Retinopathy Detection - Configuration
================================================
Centralized configuration for hyperparameters, paths, and constants.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# ──────────────────────────────────────────────
# Image settings
# ──────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# ──────────────────────────────────────────────
# Class labels  (Diabetic Retinopathy severity)
# ──────────────────────────────────────────────
CLASS_NAMES = [
    "No_DR",          # 0 – No diabetic retinopathy
    "Mild",           # 1 – Mild non-proliferative DR
    "Moderate",       # 2 – Moderate non-proliferative DR
    "Severe",         # 3 – Severe non-proliferative DR
    "Proliferative",  # 4 – Proliferative DR
]
NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
MODEL_BACKBONE = "ResNet50"  # Options: ResNet50, EfficientNetB0, VGG16
DROPOUT_RATE = 0.4
FINE_TUNE_AT = 100  # unfreeze layers from this index onward

# ──────────────────────────────────────────────
# Saved model file
# ──────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_dr_model.pth")

# ──────────────────────────────────────────────
# Ensure directories exist
# ──────────────────────────────────────────────
for _dir in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, UPLOAD_DIR]:
    os.makedirs(_dir, exist_ok=True)
