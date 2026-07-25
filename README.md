# Credit Risk Scoring - End-to-End ML Pipeline

Project credit risk scoring end-to-end menggunakan dataset Home Credit Default Risk,
mencakup anomaly detection, dual-model comparison (Logistic Regression + WOE Scorecard
vs XGBoost), MLOps (MLflow, FastAPI, Docker), dan dashboard (in progress).

**Status: Work in Progress**

## Struktur Project

```
├── notebooks/          # Jupyter notebooks (EDA, feature engineering, modeling)
├── src/                 # Reusable Python modules
├── api/                  # FastAPI serving
├── models/            # Trained model artifacts
├── tests/                # Unit tests
├── Dockerfile
└── requirements.txt
```

## Progress

- [x] EDA & Data Cleaning
- [x] Anomaly Detection (Isolation Forest)
- [x] Feature Engineering (5 tabel sekunder Home Credit)
- [x] Model A: Logistic Regression + WOE Scorecard (Test AUC 0.767, KS 0.400)
- [x] Model B: XGBoost (Test AUC 0.774, KS 0.412)
- [x] MLflow experiment tracking
- [x] FastAPI model serving
- [x] Docker containerization
- [ ] CI/CD (GitHub Actions)
- [ ] Monitoring (Evidently AI)
- [ ] Dashboard (Tableau)

## Cara Menjalankan API

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

Buka `http://localhost:8000/docs` untuk dokumentasi interaktif.

Dokumentasi lengkap menyusul di akhir project.
