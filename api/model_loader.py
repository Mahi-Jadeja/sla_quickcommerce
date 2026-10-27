import joblib
import boto3
from pathlib import Path
from botocore.exceptions import BotoCoreError, ClientError
from src.config import BEST_MODEL_PATH, S3_BUCKET_NAME, S3_MODEL_KEY, AWS_REGION
from src.logger import logger


_model_instance = None
_model_source = "None"


def download_model_from_s3(bucket_name: str, object_key: str, destination_path: Path):
    """Downloads model binary from AWS S3 at runtime."""
    logger.info(f"Attempting to download model binary from s3://{bucket_name}/{object_key}...")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    s3_client.download_file(bucket_name, object_key, str(destination_path))
    logger.info(f"Successfully downloaded model binary to {destination_path}")


def get_model():
    """
    Returns cached model instance. If not loaded:
    1. Tries to load from local BEST_MODEL_PATH
    2. If local model missing, downloads runtime binary from S3
    """
    global _model_instance, _model_source

    if _model_instance is not None:
        return _model_instance, _model_source

    if BEST_MODEL_PATH.exists():
        logger.info(f"Loading best model from local path: {BEST_MODEL_PATH}")
        _model_instance = joblib.load(BEST_MODEL_PATH)
        _model_source = "Local Disk"
        return _model_instance, _model_source

    # Fallback to S3 download at runtime
    try:
        download_model_from_s3(S3_BUCKET_NAME, S3_MODEL_KEY, BEST_MODEL_PATH)
        _model_instance = joblib.load(BEST_MODEL_PATH)
        _model_source = f"AWS S3 (s3://{S3_BUCKET_NAME}/{S3_MODEL_KEY})"
        return _model_instance, _model_source
    except (BotoCoreError, ClientError, Exception) as e:
        logger.error(f"Failed to fetch model binary from AWS S3: {e}")
        raise RuntimeError("Model binary unavailable locally and could not be fetched from S3.") from e
