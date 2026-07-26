import pandas as pd
import numpy as np

def clean_application_data(df):
    """
    Data cleaning pipeline for application_train.csv and application_test.csv.

    Handles:
    - DAYS_EMPLOYED anomalies
    - Missing values by feature group
    - Feature engineering flags for predictive modeling
    """
    df = df.copy()

    # 1. Handle DAYS_EMPLOYED anomaly
    # 365243 is a placeholder value used for retired applicants or special statuses.
    df['DAYS_EMPLOYED_ANOMALY'] = df['DAYS_EMPLOYED'] == 365243
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())

    # 2. OWN_CAR_AGE: missing values indicate the applicant does not own a car,
    # rather than representing unknown information.
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)

    # 3. EXT_SOURCE_1/2/3:
    # Impute missing values with the median since these are highly predictive features.
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df[col] = df[col].fillna(df[col].median())

    # 4. OCCUPATION_TYPE:
    # Treat missing values as a separate category.
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

    # 5. AMT_REQ_CREDIT_BUREAU_*:
    # Missing values indicate no recorded credit bureau inquiries.
    bureau_inquiry_cols = [
        c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU')
    ]
    df[bureau_inquiry_cols] = df[bureau_inquiry_cols].fillna(0)

    # 6. Property/building-related features:
    # Create a flag indicating whether property information exists,
    # then remove the detailed property attributes.
    property_cols = [
        c for c in df.columns if any(
            keyword in c for keyword in [
                'APARTMENTS', 'BASEMENTAREA', 'YEARS_BEGINEXPLUATATION',
                'YEARS_BUILD', 'COMMONAREA', 'ELEVATORS', 'ENTRANCES',
                'FLOORSMAX', 'FLOORSMIN', 'LANDAREA',
                'LIVINGAPARTMENTS', 'LIVINGAREA',
                'NONLIVINGAPARTMENTS', 'NONLIVINGAREA',
                'TOTALAREA_MODE', 'WALLSMATERIAL_MODE',
                'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE',
                'FONDKAPREMONT_MODE'
            ]
        )
    ]

    df['HAS_PROPERTY_INFO'] = (
        df[property_cols].notnull().any(axis=1).astype(int)
    )

    df = df.drop(columns=property_cols)

    # 7. Impute remaining numeric features with less than 1% missing values.
    # Numeric columns use the median, while categorical columns use the mode.
    remaining_missing = df.isnull().mean()

    small_missing_numeric = remaining_missing[
        (remaining_missing > 0) & (remaining_missing < 0.01)
    ].index

    for col in small_missing_numeric:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 8. Create baseline financial ratio features.
    # Although EDA showed limited predictive power,
    # they are retained as baseline engineered features.
    df['CREDIT_INCOME_RATIO'] = (
        df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    )

    df['ANNUITY_INCOME_RATIO'] = (
        df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    )

    return df