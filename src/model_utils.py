# src/model_utils.py
import pandas as pd
from scipy.stats import ks_2samp

def calculate_ks(y_true, y_pred):
    df_ks = pd.DataFrame({'target': y_true, 'pred': y_pred})
    good = df_ks[df_ks['target'] == 0]['pred']
    bad = df_ks[df_ks['target'] == 1]['pred']
    ks_stat, _ = ks_2samp(good, bad)
    return ks_stat