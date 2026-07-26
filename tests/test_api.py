from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input():
    # Minimal input example for a basic smoke test
    sample_input = {
        "features": {
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "Y",
            "AMT_INCOME_TOTAL": 157500.0,
            "AMT_CREDIT": 770292.0,
        }
    }

    response = client.post("/predict", json=sample_input)

    # Since the input is incomplete (not the full set of 132 features),
    # either a successful response or a 400 validation error is acceptable.
    # This test ensures the API handles incomplete input gracefully
    # instead of crashing.
    assert response.status_code in [200, 400]