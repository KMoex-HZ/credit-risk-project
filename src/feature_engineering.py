import pandas as pd
import numpy as np

# ----------------------------------------------------
# 1. FITUR DARI TABEL BUREAU (BI CHECKING LUAR)
# ----------------------------------------------------
def engineer_bureau_features(bureau):
    bureau = bureau.copy()
    bureau['FLAG_OVERDUE'] = (bureau['CREDIT_DAY_OVERDUE'] > 0).astype(int)
    bureau['FLAG_BAD_DEBT'] = (bureau['CREDIT_ACTIVE'] == 'Bad debt').astype(int)
    
    agg = bureau.groupby('SK_ID_CURR').agg(
        BUREAU_COUNT_LOANS=('SK_ID_BUREAU', 'count'),
        BUREAU_COUNT_ACTIVE=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        BUREAU_COUNT_CLOSED=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
        BUREAU_COUNT_BAD_DEBT=('FLAG_BAD_DEBT', 'sum'),
        BUREAU_SUM_OVERDUE_COUNT=('FLAG_OVERDUE', 'sum'),
        BUREAU_MAX_DAYS_OVERDUE=('CREDIT_DAY_OVERDUE', 'max'),
        BUREAU_MEAN_DAYS_OVERDUE=('CREDIT_DAY_OVERDUE', 'mean'),
        BUREAU_TOTAL_CREDIT_SUM=('AMT_CREDIT_SUM', 'sum'),
        BUREAU_TOTAL_CREDIT_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_TOTAL_CREDIT_OVERDUE=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        BUREAU_MAX_CREDIT_OVERDUE=('AMT_CREDIT_MAX_OVERDUE', 'max'),
        BUREAU_MEAN_CNT_PROLONG=('CNT_CREDIT_PROLONG', 'mean'),
        BUREAU_MAX_CNT_PROLONG=('CNT_CREDIT_PROLONG', 'max'),
        BUREAU_MEAN_DAYS_CREDIT=('DAYS_CREDIT', 'mean'),
    ).reset_index()
    
    agg['BUREAU_RATIO_ACTIVE'] = agg['BUREAU_COUNT_ACTIVE'] / agg['BUREAU_COUNT_LOANS']
    agg['BUREAU_DEBT_CREDIT_RATIO'] = agg['BUREAU_TOTAL_CREDIT_DEBT'] / agg['BUREAU_TOTAL_CREDIT_SUM']
    agg['BUREAU_DEBT_CREDIT_RATIO'] = agg['BUREAU_DEBT_CREDIT_RATIO'].replace([np.inf, -np.inf], np.nan)
    agg['BUREAU_RATIO_OVERDUE'] = agg['BUREAU_SUM_OVERDUE_COUNT'] / agg['BUREAU_COUNT_LOANS']
    
    return agg

# ----------------------------------------------------
# 2. FITUR DARI TABEL PREVIOUS APPLICATION (RIWAYAT INTERNAL)
# ----------------------------------------------------
def engineer_previous_application_features(prev_app):
    prev_app = prev_app.copy()
    days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
                 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols:
        prev_app[col] = prev_app[col].replace(365243, np.nan)
    
    prev_app['FLAG_APPROVED'] = (prev_app['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev_app['FLAG_REFUSED'] = (prev_app['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    prev_app['APPLICATION_CREDIT_RATIO'] = prev_app['AMT_CREDIT'] / prev_app['AMT_APPLICATION']
    prev_app['APPLICATION_CREDIT_RATIO'] = prev_app['APPLICATION_CREDIT_RATIO'].replace([np.inf, -np.inf], np.nan)
    
    agg = prev_app.groupby('SK_ID_CURR').agg(
        PREV_COUNT_APPLICATIONS=('SK_ID_PREV', 'count'),
        PREV_COUNT_APPROVED=('FLAG_APPROVED', 'sum'),
        PREV_COUNT_REFUSED=('FLAG_REFUSED', 'sum'),
        PREV_MEAN_AMT_ANNUITY=('AMT_ANNUITY', 'mean'),
        PREV_MEAN_AMT_APPLICATION=('AMT_APPLICATION', 'mean'),
        PREV_MEAN_AMT_CREDIT=('AMT_CREDIT', 'mean'),
        PREV_MEAN_APPLICATION_CREDIT_RATIO=('APPLICATION_CREDIT_RATIO', 'mean'),
        PREV_MEAN_DAYS_FIRST_DUE=('DAYS_FIRST_DUE', 'mean'),
        PREV_MEAN_DAYS_LAST_DUE=('DAYS_LAST_DUE', 'mean'),
        PREV_MEAN_CNT_PAYMENT=('CNT_PAYMENT', 'mean'),
    ).reset_index()
    
    agg['PREV_RATIO_REFUSED'] = agg['PREV_COUNT_REFUSED'] / agg['PREV_COUNT_APPLICATIONS']
    agg['PREV_RATIO_APPROVED'] = agg['PREV_COUNT_APPROVED'] / agg['PREV_COUNT_APPLICATIONS']
    
    return agg

# ----------------------------------------------------
# 3. FITUR DARI TABEL INSTALLMENTS PAYMENTS (RIWAYAT CICILAN BULANAN)
# ----------------------------------------------------
def engineer_installments_features(installments):
    installments = installments.copy()
    installments['FLAG_NOT_PAID'] = installments['AMT_PAYMENT'].isnull().astype(int)
    installments['DAYS_LATE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['AMT_SHORTFALL'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    installments['FLAG_LATE'] = (installments['DAYS_LATE'] > 0).astype(int)
    installments['FLAG_SHORTFALL'] = (installments['AMT_SHORTFALL'] > 0).astype(int)
    
    agg = installments.groupby('SK_ID_CURR').agg(
        INST_COUNT=('SK_ID_PREV', 'count'),
        INST_COUNT_NOT_PAID=('FLAG_NOT_PAID', 'sum'),
        INST_MEAN_DAYS_LATE=('DAYS_LATE', 'mean'),
        INST_MAX_DAYS_LATE=('DAYS_LATE', 'max'),
        INST_SUM_FLAG_LATE=('FLAG_LATE', 'sum'),
        INST_MEAN_AMT_SHORTFALL=('AMT_SHORTFALL', 'mean'),
        INST_SUM_AMT_SHORTFALL=('AMT_SHORTFALL', 'sum'),
        INST_SUM_FLAG_SHORTFALL=('FLAG_SHORTFALL', 'sum'),
        INST_MEAN_AMT_INSTALMENT=('AMT_INSTALMENT', 'mean'),
        INST_MEAN_AMT_PAYMENT=('AMT_PAYMENT', 'mean'),
    ).reset_index()
    
    agg['INST_RATIO_LATE'] = agg['INST_SUM_FLAG_LATE'] / agg['INST_COUNT']
    agg['INST_RATIO_SHORTFALL'] = agg['INST_SUM_FLAG_SHORTFALL'] / agg['INST_COUNT']
    agg['INST_RATIO_NOT_PAID'] = agg['INST_COUNT_NOT_PAID'] / agg['INST_COUNT']
    
    return agg

# ----------------------------------------------------
# 4. FITUR DARI TABEL POS CASH BALANCE
# ----------------------------------------------------
def engineer_pos_cash_features(pos_cash):
    pos_cash = pos_cash.copy()
    pos_cash['FLAG_DPD'] = (pos_cash['SK_DPD'] > 0).astype(int)
    
    agg = pos_cash.groupby('SK_ID_CURR').agg(
        POS_COUNT=('SK_ID_PREV', 'count'),
        POS_MEAN_CNT_INSTALMENT_FUTURE=('CNT_INSTALMENT_FUTURE', 'mean'),
        POS_MEAN_SK_DPD=('SK_DPD', 'mean'),
        POS_MAX_SK_DPD=('SK_DPD', 'max'),
        POS_SUM_FLAG_DPD=('FLAG_DPD', 'sum'),
        POS_MEAN_SK_DPD_DEF=('SK_DPD_DEF', 'mean'),
        POS_MAX_SK_DPD_DEF=('SK_DPD_DEF', 'max'),
    ).reset_index()
    
    agg['POS_RATIO_DPD'] = agg['POS_SUM_FLAG_DPD'] / agg['POS_COUNT']
    
    return agg

# ----------------------------------------------------
# 5. FITUR DARI TABEL CREDIT CARD BALANCE (JUARA KITA!)
# ----------------------------------------------------
def engineer_credit_card_features(credit_card):
    credit_card = credit_card.copy()
    credit_card['FLAG_DPD'] = (credit_card['SK_DPD'] > 0).astype(int)
    
    credit_card['UTILIZATION_RATIO'] = credit_card['AMT_BALANCE'] / credit_card['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card['UTILIZATION_RATIO'] = credit_card['UTILIZATION_RATIO'].replace([np.inf, -np.inf], np.nan)
    
    agg = credit_card.groupby('SK_ID_CURR').agg(
        CC_COUNT=('SK_ID_PREV', 'count'),
        CC_MEAN_AMT_BALANCE=('AMT_BALANCE', 'mean'),
        CC_MEAN_CREDIT_LIMIT=('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
        CC_MEAN_UTILIZATION=('UTILIZATION_RATIO', 'mean'),
        CC_MAX_UTILIZATION=('UTILIZATION_RATIO', 'max'),
        CC_MEAN_DRAWINGS_ATM=('AMT_DRAWINGS_ATM_CURRENT', 'mean'),
        CC_MEAN_DRAWINGS_CURRENT=('AMT_DRAWINGS_CURRENT', 'mean'),
        CC_MEAN_SK_DPD=('SK_DPD', 'mean'),
        CC_MAX_SK_DPD=('SK_DPD', 'max'),
        CC_SUM_FLAG_DPD=('FLAG_DPD', 'sum'),
    ).reset_index()
    
    agg['CC_RATIO_DPD'] = agg['CC_SUM_FLAG_DPD'] / agg['CC_COUNT']
    
    return agg
    