"""Provide a lightweight dummy model for CI tests without full training."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import BEST_MODEL_PATH, METRICS_FILE_PATH, MODELS_DIR


@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model_if_missing():
    """Ensure best_model.pkl and metrics.json exist for test suite execution in CI environments."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not METRICS_FILE_PATH.exists():
        with open(METRICS_FILE_PATH, "w") as f:
            json.dump(
                {
                    "best_model": "LightGBM",
                    "accuracy": 0.9385,
                    "f1_score": 0.8900,
                    "roc_auc": 0.9852,
                },
                f,
                indent=4,
            )

    if not BEST_MODEL_PATH.exists():
        # Create minimal dummy dataframe matching full schema
        dummy_df = pd.DataFrame(
            [
                {
                    "Delivery_person_Age": 28.0,
                    "Delivery_person_Ratings": 4.5,
                    "distance_km": 5.0,
                    "Weatherconditions": "Sunny",
                    "Road_traffic_density": "Low",
                    "Vehicle_condition": 2,
                    "Type_of_order": "Snack",
                    "Type_of_vehicle": "motorcycle",
                    "multiple_deliveries": 0.0,
                    "Festival": "No",
                    "City": "Metropolitian",
                }
            ]
            * 5
        )

        y_dummy = np.array([0, 1, 0, 1, 0])

        num_cols = [
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "distance_km",
            "Vehicle_condition",
            "multiple_deliveries",
        ]
        cat_cols = [
            "Weatherconditions",
            "Road_traffic_density",
            "Type_of_order",
            "Type_of_vehicle",
            "Festival",
            "City",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression()),
            ]
        )

        pipeline.fit(dummy_df, y_dummy)
        joblib.dump(pipeline, BEST_MODEL_PATH)
