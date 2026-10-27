"""
Preprocessing pipeline for the quick-commerce SLA prediction project.

Cleans raw delivery data, engineers features such as delivery distance,
and prepares the target used to predict whether an order will breach
its promised delivery time.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    SLA_THRESHOLD_MINUTES,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.logger import logger


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    r = 6371.0  # Earth radius in kilometers
    
    lat1_rad = np.radians(np.abs(lat1))
    lon1_rad = np.radians(np.abs(lon1))
    lat2_rad = np.radians(np.abs(lat2))
    lon2_rad = np.radians(np.abs(lon2))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return r * c


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Quick-Commerce delivery data:
    - Strip whitespace from string columns
    - Clean target 'Time_taken(min)' -> numerical minutes
    - Extract binary target 'will_breach_sla' (1 if > 30.0 mins, else 0)
    - Clean numeric features: Age, Ratings, Multiple Deliveries
    - Engineer Haversine Distance between Restaurant and Delivery location
    """
    logger.info("Starting data cleaning and feature engineering...")
    df = df.copy()

    # Clean string column whitespace
    string_cols = df.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    # Clean Target Column: 'Time_taken(min)' format is usually '(min) 24'
    if "Time_taken(min)" in df.columns:
        df["delivery_time_min"] = (
            df["Time_taken(min)"]
            .str.replace(r"[^\d.]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    elif "delivery_time_min" not in df.columns:
        raise KeyError("Neither 'Time_taken(min)' nor 'delivery_time_min' found in dataset.")

    # Drop rows missing delivery time
    df = df.dropna(subset=["delivery_time_min"]).reset_index(drop=True)

    # Engineering SLA Risk Target (Binary Classification)
    df["will_breach_sla"] = (df["delivery_time_min"] > SLA_THRESHOLD_MINUTES).astype(int)

    # Clean Delivery Person Age & Ratings
    df["Delivery_person_Age"] = pd.to_numeric(
        df["Delivery_person_Age"].replace("NaN", np.nan), errors="coerce"
    )
    df["Delivery_person_Ratings"] = pd.to_numeric(
        df["Delivery_person_Ratings"].replace("NaN", np.nan), errors="coerce"
    )
    df["multiple_deliveries"] = pd.to_numeric(
        df["multiple_deliveries"].replace("NaN", np.nan), errors="coerce"
    )

    # Impute missing numerical features with median
    df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(df["Delivery_person_Age"].median())
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(df["Delivery_person_Ratings"].median())
    df["multiple_deliveries"] = df["multiple_deliveries"].fillna(0)

    # Haversine Distance Calculation
    df["distance_km"] = haversine_distance(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )

    # Remove invalid coordinate distances (e.g. 0 or extreme outliers)
    df["distance_km"] = df["distance_km"].clip(lower=0.1, upper=100.0)

    # Weather condition cleaning
    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    df["Weatherconditions"] = df["Weatherconditions"].replace("NaN", "Sunny")
    
    # Traffic density cleaning
    df["Road_traffic_density"] = df["Road_traffic_density"].replace("NaN", "Medium")

    # Select final modeling features
    feature_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "distance_km",
        "Weatherconditions",
        "Road_traffic_density",
        "Vehicle_condition",
        "Type_of_order",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival",
        "City",
        "will_breach_sla"
    ]

    clean_df = df[feature_cols].dropna().reset_index(drop=True)
    logger.info(f"Cleaned dataset shape: {clean_df.shape}. Target distribution:\n{clean_df['will_breach_sla'].value_counts(normalize=True)}")
    
    return clean_df


def run_preprocessing():
    """Reads raw data, cleans, splits into train/test, and saves to disk."""
    logger.info(f"Loading raw dataset from {RAW_DATA_PATH}")
    raw_df = pd.read_csv(RAW_DATA_PATH)

    processed_df = clean_raw_data(raw_df)

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        processed_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=processed_df["will_breach_sla"],
    )

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    logger.info(f"Successfully saved train dataset ({len(train_df)} rows) to {TRAIN_DATA_PATH}")
    logger.info(f"Successfully saved test dataset ({len(test_df)} rows) to {TEST_DATA_PATH}")


if __name__ == "__main__":
    run_preprocessing()
