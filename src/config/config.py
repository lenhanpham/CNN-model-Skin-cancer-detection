import os

# Directory configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Project root
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Data parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_NAME = "fanconic/skin-cancer-malignant-vs-benign"
VALIDATION_SPLIT = 0.2

# Data augmentation parameters
ROTATION_RANGE = 30
BRIGHTNESS_RANGE = [0.9, 1.1]
ZOOM_RANGE = 0.1
FILL_MODE = "constant"
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
RESCALE = 1.0 / 255

# Model parameters
MODEL_NAME = "fine_tuned_xception_model"
LEARNING_RATE = 2e-5
EPOCHS = 50
DENSE_UNITS = 16
PATIENCE = 10

# Training parameters
NUM_THREADS = 16
METRICS_TO_PLOT = ["accuracy", "loss", "auc", "recall"]