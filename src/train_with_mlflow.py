import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import ks_2samp

def calculate_ks(y_true, y_pred):
    df_ks = pd.DataFrame({'target': y_true, 'pred': y_pred})
    good = df_ks[df_ks['target'] == 0]['pred']
    bad = df_ks[df_ks['target'] == 1]['pred']
    ks_stat, _ = ks_2samp(good, bad)
    return ks_stat

# Load data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

cat_cols = X_train.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Set experiment name
mlflow.set_experiment("credit-risk-scoring")

# ============ Model B: XGBoost ============
with mlflow.start_run(run_name="xgboost_v2_regularized"):
    params = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 1.5,
        'min_child_weight': 5,
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    }
    
    mlflow.log_params(params)
    
    model = XGBClassifier(
        **params,
        enable_categorical=True,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=30
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'train_auc': roc_auc_score(y_train, train_pred),
        'test_auc': roc_auc_score(y_test, test_pred),
        'train_ks': calculate_ks(y_train, train_pred),
        'test_ks': calculate_ks(y_test, test_pred),
    }
    
    mlflow.log_metrics(metrics)
    mlflow.xgboost.log_model(model, "model")
    
    print("XGBoost logged:", metrics)

print("\nDone! Run 'mlflow ui' to view results.")

# ============ Model A: Logistic Regression + WOE ============
print("\n>>> Memulai proses Model A...")

X_train_woe_final = pd.read_csv('data/processed/X_train_woe_final.csv')
X_test_woe_final = pd.read_csv('data/processed/X_test_woe_final.csv')

with mlflow.start_run(run_name="logistic_regression_woe_final"):
    params = {'penalty': 'l1', 'solver': 'liblinear', 'C': 0.1}
    mlflow.log_params(params)
    mlflow.log_param('n_features', X_train_woe_final.shape[1])
    
    model_a = LogisticRegression(**params, random_state=42, max_iter=1000)
    model_a.fit(X_train_woe_final, y_train)
    
    train_pred_a = model_a.predict_proba(X_train_woe_final)[:, 1]
    test_pred_a = model_a.predict_proba(X_test_woe_final)[:, 1]
    
    metrics_a = {
        'train_auc': roc_auc_score(y_train, train_pred_a),
        'test_auc': roc_auc_score(y_test, test_pred_a),
        'train_ks': calculate_ks(y_train, train_pred_a),
        'test_ks': calculate_ks(y_test, test_pred_a),
    }
    
    mlflow.log_metrics(metrics_a)
    mlflow.sklearn.log_model(model_a, "model")
    
    print("Logistic Regression logged:", metrics_a)