import numpy as np
import pandas as pd
import pytest
from src.preprocessing import haversine_distance, clean_raw_data


def test_haversine_distance():
    dist = haversine_distance(19.0760, 72.8777, 18.5204, 73.8567)
    assert 110.0 < dist < 140.0


def test_clean_raw_data_sla_target():
    dummy_data = pd.DataFrame({
        "Delivery_person_Age": ["30 ", "25 "],
        "Delivery_person_Ratings": ["4.8 ", "4.2 "],
        "Restaurant_latitude": [19.07, 19.08],
        "Restaurant_longitude": [72.87, 72.88],
        "Delivery_location_latitude": [19.12, 19.15],
        "Delivery_location_longitude": [72.90, 72.92],
        "Weatherconditions": ["conditions Sunny", "conditions Rainy"],
        "Road_traffic_density": ["Low ", "Jam "],
        "Vehicle_condition": [2, 1],
        "Type_of_order": ["Snack ", "Meal "],
        "Type_of_vehicle": ["motorcycle ", "scooter "],
        "multiple_deliveries": ["0 ", "1 "],
        "Festival": ["No ", "No "],
        "City": ["Metropolitian ", "Urban "],
        "Time_taken(min)": ["(min) 20", "(min) 45"]
    })

    cleaned = clean_raw_data(dummy_data)
    assert "will_breach_sla" in cleaned.columns
    assert cleaned.loc[0, "will_breach_sla"] == 0  # 20 mins <= 30 mins
    assert cleaned.loc[1, "will_breach_sla"] == 1  # 45 mins > 30 mins
    assert cleaned.shape[0] == 2
