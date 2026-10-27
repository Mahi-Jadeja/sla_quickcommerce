import os
import joblib
import pandas as pd
import pytest
from src.config import BEST_MODEL_PATH, METRICS_FILE_PATH

# Smoke-check the saved model artifact without running a full training cycle.
def test_model_artifact_creation():
    """Verify that best_model.pkl exists and can produce predictions on sample data."""
    assert os.path.exists(BEST_MODEL_PATH), "best_model.pkl was not created"
    assert os.path.exists(METRICS_FILE_PATH), "metrics.json was not created"

    model = joblib.load(BEST_MODEL_PATH)
    
    sample_payload = pd.DataFrame([{
        "Delivery_person_Age": 28,
        "Delivery_person_Ratings": 4.7,
        "distance_km": 5.2,
        "Weatherconditions": "Sunny",
        "Road_traffic_density": "Low",
        "Vehicle_condition": 2,
        "Type_of_order": "Snack",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 0,
        "Festival": "No",
        "City": "Metropolitian"
    }])

    pred = model.predict(sample_payload)
    proba = model.predict_proba(sample_payload)

    assert pred[0] in [0, 1]
    assert 0.0 <= proba[0][1] <= 1.0
