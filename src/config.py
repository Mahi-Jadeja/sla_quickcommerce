import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "quick_commerce.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METRICS_FILE_PATH = BASE_DIR / "metrics.json"

# Business Logic Configuration
SLA_THRESHOLD_MINUTES = 30.0  # > 30 mins means SLA Breached (1)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mlops-sla-mahipune01")
S3_MODEL_KEY = "models/best_model.pkl"

# Model Training Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
