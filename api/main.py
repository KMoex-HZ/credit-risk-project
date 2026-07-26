from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.schemas import LoanApplication, PredictionResponse

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_final.pkl'))
iso_forest = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
anomaly_scaler = joblib.load(os.path.join(MODEL_DIR, 'anomaly_scaler.pkl'))
categories_dict = joblib.load(os.path.join(MODEL_DIR, 'categorical_categories.pkl'))

ISO_FEATURES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
    'AGE_YEARS', 'EMPLOYED_YEARS',
    'AMT_REQ_CREDIT_BUREAU_YEAR', 'TOTAL_DOCUMENTS_SUBMITTED'
]

def get_risk_category(prob):
    if prob < 0.05:
        return "Low Risk"
    elif prob < 0.15:
        return "Medium Risk"
    else:
        return "High Risk"

@app.get("/")
def root():
    return {"status": "ok", "message": "Credit Risk Scoring API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        df = pd.DataFrame([application.features])

        # --- Anomaly detection (requires AMT_GOODS_PRICE) ---
        df_anomaly = df.copy()
        if 'AMT_GOODS_PRICE' not in df_anomaly.columns:
            df_anomaly['AMT_GOODS_PRICE'] = df_anomaly['AMT_CREDIT']

        X_iso = df_anomaly[ISO_FEATURES].copy()
        X_iso_scaled = anomaly_scaler.transform(X_iso)
        anomaly_pred = iso_forest.predict(X_iso_scaled)[0]
        anomaly_score = iso_forest.decision_function(X_iso_scaled)[0]
        is_anomaly = bool(anomaly_pred == -1)

        # --- Probability of Default (PD) prediction using XGBoost ---
        df_xgb = df.copy()
        if 'AMT_GOODS_PRICE' in df_xgb.columns:
            df_xgb = df_xgb.drop(columns=['AMT_GOODS_PRICE'])

        # Force categorical columns to use the EXACT same categories as during training
        for col, cats in categories_dict.items():
            if col in df_xgb.columns:
                df_xgb[col] = pd.Categorical(df_xgb[col], categories=cats)

        # Convert boolean columns received as strings from JSON
        if 'DAYS_EMPLOYED_ANOMALY' in df_xgb.columns:
            df_xgb['DAYS_EMPLOYED_ANOMALY'] = df_xgb['DAYS_EMPLOYED_ANOMALY'].astype(str).map(
                {'True': True, 'False': False}
            ).astype(bool)

        # Explicitly enable categorical support because this setting
        # may not be preserved when the model is reloaded from a .pkl file
        from xgboost import DMatrix
        dmatrix = DMatrix(df_xgb, enable_categorical=True)
        prob_default = xgb_model.get_booster().predict(dmatrix)[0]

        return PredictionResponse(
            sk_id_curr=application.features.get('SK_ID_CURR'),
            probability_default=round(float(prob_default), 4),
            risk_score=round((1 - prob_default) * 850, 0),
            is_anomaly=is_anomaly,
            anomaly_score=round(float(anomaly_score), 4),
            risk_category=get_risk_category(prob_default)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))