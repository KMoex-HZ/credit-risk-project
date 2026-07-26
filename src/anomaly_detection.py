import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ISO_FEATURES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
    'AGE_YEARS', 'EMPLOYED_YEARS',
    'AMT_REQ_CREDIT_BUREAU_YEAR', 'TOTAL_DOCUMENTS_SUBMITTED'
]

def create_anomaly_features(df):
    df = df.copy()

    df['FLAG_HIGH_CREDIT_INCOME_RATIO'] = (
        df['CREDIT_INCOME_RATIO'] > df['CREDIT_INCOME_RATIO'].quantile(0.99)
    ).astype(int)

    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365

    df['FLAG_EMPLOYED_LONGER_THAN_POSSIBLE'] = (
        df['EMPLOYED_YEARS'] > (df['AGE_YEARS'] - 14)
    ).astype(int)

    df['FLAG_INCOME_OUTLIER'] = (
        df['AMT_INCOME_TOTAL'] > df['AMT_INCOME_TOTAL'].quantile(0.999)
    ).astype(int)

    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
    df['TOTAL_DOCUMENTS_SUBMITTED'] = df[doc_cols].sum(axis=1)

    df['FLAG_HIGH_BUREAU_INQUIRY'] = (
        df['AMT_REQ_CREDIT_BUREAU_YEAR'] >
        df['AMT_REQ_CREDIT_BUREAU_YEAR'].quantile(0.99)
    ).astype(int)

    return df

def fit_anomaly_detector(df, contamination=0.02, random_state=42):
    """
    Fit the Isolation Forest model and return the trained model
    along with the fitted scaler for reuse on test or inference data.
    """
    X = df[ISO_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    iso_forest.fit(X_scaled)

    return iso_forest, scaler

def apply_anomaly_detector(df, iso_forest, scaler):
    """
    Apply the trained anomaly detector to either training or test data
    and return the dataframe with anomaly score and prediction columns.
    """
    df = df.copy()

    X = df[ISO_FEATURES].copy()
    X_scaled = scaler.transform(X)

    df['ANOMALY_SCORE'] = iso_forest.decision_function(X_scaled)
    df['IS_ANOMALY'] = (iso_forest.predict(X_scaled) == -1).astype(int)

    return df