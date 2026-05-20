import pandas as pd
import numpy as np
import os
import re
import json
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator          # sklearn >= 1.6
except ImportError:
    FrozenEstimator = None                               # sklearn < 1.6
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, roc_curve,
)
from xgboost import XGBClassifier


def train():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir    = os.path.join(project_dir, 'data')
    models_dir  = os.path.join(project_dir, 'models')
    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    data_path = os.path.join(data_dir, 'clean_data.csv')

    print("Loading clean data...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['target'])
    y = df['target']

    # XGBoost cannot handle bool dtype or special chars in column names
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    X.columns = [re.sub(r'[\[\]<]', '_', col) for col in X.columns]

    # --- Split: 70 / 15 / 15 stratified ---
    print("Splitting data (70/15/15)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=(0.15 / 0.85), random_state=42, stratify=y_train_full
    )
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # --- Train Random Forest ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # --- Train XGBoost ---
    print("Training XGBoost...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        random_state=42, scale_pos_weight=scale_pos_weight,
        n_jobs=-1, eval_metric='auc',
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # --- Evaluation helper ---
    def evaluate(model, name, X_eval, y_eval) -> dict:
        probs = model.predict_proba(X_eval)[:, 1]
        # Use threshold that maximises F1 on this eval set
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1, best_thresh = 0.0, 0.5
        for thr in thresholds:
            preds_t = (probs >= thr).astype(int)
            f = f1_score(y_eval, preds_t, zero_division=0)
            if f > best_f1:
                best_f1, best_thresh = f, thr
        preds = (probs >= best_thresh).astype(int)
        metrics = {
            'roc_auc':   round(float(roc_auc_score(y_eval, probs)),               4),
            'f1':        round(float(f1_score(y_eval, preds, zero_division=0)),    4),
            'precision': round(float(precision_score(y_eval, preds, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_eval, preds, zero_division=0)),    4),
            'threshold': round(float(best_thresh),                                 3),
        }
        print(f"\n  [{name}]")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")
        return metrics

    print("\n=== Validation Metrics ===")
    rf_val_metrics  = evaluate(rf,  "Random Forest (val)", X_val, y_val)
    xgb_val_metrics = evaluate(xgb, "XGBoost       (val)", X_val, y_val)

    print("\n=== Test Metrics ===")
    rf_test_metrics  = evaluate(rf,  "Random Forest (test)", X_test, y_test)
    xgb_test_metrics = evaluate(xgb, "XGBoost       (test)", X_test, y_test)

    # --- Pick best model by val ROC-AUC ---
    rf_auc  = rf_val_metrics['roc_auc']
    xgb_auc = xgb_val_metrics['roc_auc']
    best_raw_model = xgb if xgb_auc > rf_auc else rf
    best_name      = 'xgboost' if xgb_auc > rf_auc else 'random_forest'
    print(f"\nBest model: {best_name} (val AUC = {max(rf_auc, xgb_auc):.4f})")

    # --- Calibrate the best model (isotonic regression on val set) ---
    print("Calibrating model probabilities (isotonic)...")
    if FrozenEstimator is not None:                      # sklearn >= 1.6
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_raw_model), method='isotonic'
        )
    else:                                                 # sklearn < 1.6
        calibrated_model = CalibratedClassifierCV(
            estimator=best_raw_model, method='isotonic', cv='prefit'
        )
    calibrated_model.fit(X_val, y_val)

    print("\n=== Calibrated Model — Test Metrics ===")
    cal_test_metrics = evaluate(calibrated_model, "Calibrated (test)", X_test, y_test)

    # --- Save models ---
    feature_list = list(X.columns)
    joblib.dump(feature_list,        os.path.join(models_dir, 'features.joblib'))
    joblib.dump(calibrated_model,    os.path.join(models_dir, 'risk_estimator.joblib'))
    joblib.dump(best_raw_model,      os.path.join(models_dir, 'base_estimator.joblib'))
    print(f"\nSaved calibrated model  → models/risk_estimator.joblib")
    print(f"Saved raw model         → models/base_estimator.joblib  (used for SHAP)")

    # --- Save metrics.json ---
    metrics_payload = {
        'best_model': best_name,
        'random_forest': {'val': rf_val_metrics,  'test': rf_test_metrics},
        'xgboost':       {'val': xgb_val_metrics, 'test': xgb_test_metrics},
        'calibrated':    {'test': cal_test_metrics},
    }
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics           → results/metrics.json")

    # --- Save ROC curve ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, label, color in [
        (rf,               f"Random Forest (AUC={rf_test_metrics['roc_auc']:.3f})",  '#636efa'),
        (xgb,              f"XGBoost       (AUC={xgb_test_metrics['roc_auc']:.3f})", '#ef553b'),
        (calibrated_model, f"Calibrated    (AUC={cal_test_metrics['roc_auc']:.3f})", '#00cc96'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=label, color=color, lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Test Set')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve         → results/roc_curve.png")
    print("\nTraining complete!")


if __name__ == '__main__':
    train()
