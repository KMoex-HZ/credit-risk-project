import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load reference (train) dan current (test) data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

# Ambil subset kolom numerik penting saja (biar report tidak terlalu berat/panjang)
key_features = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AGE_YEARS', 'EMPLOYED_YEARS',
    'BUREAU_MEAN_DAYS_CREDIT', 'INST_RATIO_LATE', 'CC_MEAN_UTILIZATION',
    'ANOMALY_SCORE', 'PREV_RATIO_REFUSED'
]

reference_data = X_train[key_features]
current_data = X_test[key_features]

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference_data, current_data=current_data)

result.save_html('data/processed/drift_report.html')
print("Drift report saved!")