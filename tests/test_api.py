from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_success():
    payload = {
        "Delivery_person_Age": 29.0,
        "Delivery_person_Ratings": 4.8,
        "distance_km": 3.5,
        "Weatherconditions": "Sunny",
        "Road_traffic_density": "Low",
        "Vehicle_condition": 2,
        "Type_of_order": "Snack",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 0.0,
        "Festival": "No",
        "City": "Metropolitian",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "will_breach_sla" in data
    assert "sla_breach_probability" in data
    assert data["will_breach_sla"] in [0, 1]
    assert 0.0 <= data["sla_breach_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_endpoint_validation_error():
    # Missing required field 'distance_km'
    payload = {
        "Delivery_person_Age": 29.0,
        "Weatherconditions": "Sunny",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
