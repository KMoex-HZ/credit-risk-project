import pandas as pd
import numpy as np

def clean_application_data(df):
    """
    Cleaning pipeline untuk application_train/test.csv
    Menangani: anomaly DAYS_EMPLOYED, missing value per kelompok kolom,
    dan feature flag yang berguna untuk modeling.
    """
    df = df.copy()

    # 1. DAYS_EMPLOYED anomaly (365243 = placeholder utk pensiunan/status khusus)
    df['DAYS_EMPLOYED_ANOMALY'] = df['DAYS_EMPLOYED'] == 365243
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())  # <-- tambahan ini


    # 2. OWN_CAR_AGE: missing = tidak punya mobil, bukan "tidak diketahui"
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)

    # 3. EXT_SOURCE 1/2/3: imputasi median (fitur paling predictive, jangan drop)
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df[col] = df[col].fillna(df[col].median())

    # 4. OCCUPATION_TYPE: missing dianggap kategori sendiri
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

    # 5. AMT_REQ_CREDIT_BUREAU_*: missing = tidak ada record inquiry -> 0
    bureau_inquiry_cols = [c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU')]
    df[bureau_inquiry_cols] = df[bureau_inquiry_cols].fillna(0)

    # 6. Kolom properti/bangunan: bikin flag ada/tidak info, lalu drop detailnya
    property_cols = [c for c in df.columns if any(
        keyword in c for keyword in [
            'APARTMENTS', 'BASEMENTAREA', 'YEARS_BEGINEXPLUATATION', 'YEARS_BUILD',
            'COMMONAREA', 'ELEVATORS', 'ENTRANCES', 'FLOORSMAX', 'FLOORSMIN',
            'LANDAREA', 'LIVINGAPARTMENTS', 'LIVINGAREA', 'NONLIVINGAPARTMENTS',
            'NONLIVINGAREA', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE',
            'HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE', 'FONDKAPREMONT_MODE'
        ]
    )]
    df['HAS_PROPERTY_INFO'] = df[property_cols].notnull().any(axis=1).astype(int)
    df = df.drop(columns=property_cols)

    # 7. Sisa kolom numerik missing kecil (<1%) -> imputasi median
    remaining_missing = df.isnull().mean()
    small_missing_numeric = remaining_missing[
        (remaining_missing > 0) & (remaining_missing < 0.01)
    ].index
    for col in small_missing_numeric:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 8. Feature ratio dasar (walau predictive power-nya lemah dari EDA,
    #    tetap disimpan sebagai baseline feature)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    return df