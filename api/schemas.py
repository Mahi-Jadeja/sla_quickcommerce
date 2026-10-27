from pydantic import BaseModel, Field, ConfigDict
from typing import List


class DeliveryInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Delivery_person_Age": 28.0,
                "Delivery_person_Ratings": 4.7,
                "distance_km": 5.2,
                "Weatherconditions": "Sunny",
                "Road_traffic_density": "Medium",
                "Vehicle_condition": 2,
                "Type_of_order": "Snack",
                "Type_of_vehicle": "motorcycle",
                "multiple_deliveries": 0.0,
                "Festival": "No",
                "City": "Metropolitian",
            }
        }
    )

    Delivery_person_Age: float = Field(..., description="Age of delivery rider")
    Delivery_person_Ratings: float = Field(..., description="Rating of delivery rider (1.0 - 5.0)")
    distance_km: float = Field(..., description="Haversine distance in km from restaurant to customer")
    Weatherconditions: str = Field(..., description="Weather condition (Sunny, Stormy, Sandstorms, Windy, Fog, Cloudy)")
    Road_traffic_density: str = Field(..., description="Traffic level (Low, Medium, High, Jam)")
    Vehicle_condition: int = Field(..., description="Vehicle condition index (0, 1, 2)")
    Type_of_order: str = Field(..., description="Order category (Snack, Meal, Drinks, Buffet)")
    Type_of_vehicle: str = Field(..., description="Vehicle type (motorcycle, scooter, electric_scooter, bicycle)")
    multiple_deliveries: float = Field(..., description="Number of concurrent orders assigned")
    Festival: str = Field(..., description="Festival period (No, Yes)")
    City: str = Field(..., description="City type (Metropolitian, Urban, Semi-Urban)")


class PredictionOutput(BaseModel):
    will_breach_sla: int = Field(..., description="1 if SLA breach predicted (>30 mins), 0 otherwise")
    sla_breach_probability: float = Field(..., description="Probability of exceeding 30-minute delivery SLA")
    risk_level: str = Field(..., description="SLA Breach Risk Category (LOW, MEDIUM, HIGH)")
    recommendation: str = Field(..., description="Actionable recommendation for delivery dispatch")


class BatchDeliveryInput(BaseModel):
    orders: List[DeliveryInput]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_source: str
    version: str
