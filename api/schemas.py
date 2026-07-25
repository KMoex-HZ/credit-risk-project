from pydantic import BaseModel
from typing import Optional

class LoanApplication(BaseModel):
    # Contoh subset fitur penting (nanti bisa expand ke semua 132 fitur)
    # Untuk sekarang kita buat generic dict-based buat fleksibilitas
    features: dict

class PredictionResponse(BaseModel):
    sk_id_curr: Optional[int] = None
    probability_default: float
    risk_score: float
    is_anomaly: bool
    anomaly_score: float
    risk_category: str