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
    # Contoh input minimal untuk smoke test
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
    # Karena input tidak lengkap (bukan 132 fitur penuh), kita expect error 400,
    # bukan crash - ini tes bahwa API menangani input tidak lengkap dengan baik
    assert response.status_code in [200, 400]