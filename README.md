# Credit Risk Scoring - End-to-End ML Pipeline

Sistem credit risk scoring end-to-end menggunakan dataset Home Credit Default Risk,
mengadopsi prinsip-prinsip Basel IRB Approach (PD modeling, discriminatory power,
calibration, stability testing). Mencakup anomaly detection layer, perbandingan dua
model (Logistic Regression + WOE Scorecard vs XGBoost), MLOps lengkap, dan dashboard
Tableau interaktif.

## Arsitektur Pipeline

```
Data (6 tabel Home Credit)
    → Cleaning & Feature Engineering (149 fitur)
    → Anomaly Detection (Isolation Forest)
    → Model A: Logistic Regression + WOE Scorecard
    → Model B: XGBoost
    → MLflow Tracking → FastAPI → Docker → CI/CD
    → Tableau Dashboard
```

## Hasil Utama

| Model                        | Test AUC | Test KS | Fitur Aktif | Explainability     |
| ---------------------------- | -------- | ------- | ----------- | ------------------ |
| A: Logistic Regression + WOE | 0.7672   | 0.400   | 62          | Native (Scorecard) |
| B: XGBoost                   | 0.7742   | 0.412   | 132         | SHAP               |

**Top predictor (konsisten di kedua model):** EXT_SOURCE_1/2/3, INST_RATIO_LATE,
PREV_RATIO_REFUSED, BUREAU_DEBT_CREDIT_RATIO, ANOMALY_SCORE

## Insight Kunci

- **Anomaly detection** mengisolasi profil high-income (avg $312K) dengan default
  rate lebih rendah (4.44% vs 8.1%) — berfungsi sebagai data-quality gate,
  bukan risk predictor langsung
- **Risiko tertinggi** pada nasabah usia <25 tahun dan segmen income menengah-bawah
- **Kalibrasi model** monotonic sempurna: default rate aktual turun konsisten
  seiring naiknya score (27.9% → 1.2%)
- **Rasio > raw count**: pola konsisten di semua fitur turunan (bureau,
  installments, previous application) — rasio selalu lebih predictive

## Struktur Project

```
├── notebooks/          # EDA, feature engineering, modeling (7 notebooks)
├── src/                 # Reusable modules (cleaning, feature eng, anomaly, MLflow)
├── api/                  # FastAPI serving
├── models/            # Trained model artifacts
├── tests/                # Unit tests
├── tableau/            # Dashboard (.twbx)
├── Dockerfile
└── requirements.txt
```

## Cara Menjalankan

**API (Docker):**

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

Buka `http://localhost:8000/docs` untuk dokumentasi interaktif.

**MLflow tracking:**

```bash
python src/train_with_mlflow.py
mlflow ui
```

**Testing:**

```bash
pytest tests/test_api.py -v
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, optbinning, SHAP, MLflow, FastAPI, Docker,
GitHub Actions, Evidently AI, Tableau

## Catatan Penting

Project ini mengadopsi **prinsip-prinsip** Basel IRB Approach (PD modeling,
validasi diskriminasi, kalibrasi, stability testing) untuk tujuan pembelajaran
dan portofolio — **bukan** model IRB-compliant untuk keperluan regulasi resmi.
Tidak mencakup LGD/EAD, capital requirement calculation, atau validasi independen
oleh regulator.

Dataset: [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)
