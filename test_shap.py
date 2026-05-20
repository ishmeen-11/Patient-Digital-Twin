"""
test_shap.py — Quick smoke test for SHAP integration.

Run from the project root:
    python test_shap.py

Verifies that SHAP values can be computed for both the raw tree model
(base_estimator.joblib) and prints the top 10 feature importances.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap

# Make sure the project root is on the path
project_dir = os.path.dirname(__file__)
sys.path.insert(0, project_dir)

from simulation.engine import DigitalTwinEngine

# Load engine
engine = DigitalTwinEngine(
    os.path.join(project_dir, 'models', 'risk_estimator.joblib'),
    os.path.join(project_dir, 'models', 'features.joblib'),
)

# Build a realistic test patient state
state = {f: 0 for f in engine.feature_names}
state['num_medications']   = 15
state['number_inpatient']  = 1
state['number_emergency']  = 1
state['HbA1c_result']      = 1
state['insulin']           = 0
state['diabetesMed']       = 1

model_input = pd.DataFrame([state], columns=engine.feature_names).fillna(0)

# Use base (uncalibrated) model for SHAP — TreeExplainer requires a tree model
base_path = os.path.join(project_dir, 'models', 'base_estimator.joblib')
base_model = joblib.load(base_path)
print(f"Loaded base model: {type(base_model).__name__}")

explainer = shap.TreeExplainer(base_model)
shap_vals = explainer.shap_values(model_input)

print(f"Raw SHAP output type  : {type(shap_vals)}")
shap_arr = np.array(shap_vals)
print(f"Raw SHAP array shape  : {shap_arr.shape}")

# Extract positive-class vector (handles RF 3D and XGB 2D)
if isinstance(shap_vals, list):
    sv = np.array(shap_vals[1]).squeeze()
else:
    if shap_arr.ndim == 3:
        sv = shap_arr[0, :, 1]
    elif shap_arr.ndim == 2:
        sv = shap_arr[0] if shap_arr.shape[0] == 1 else shap_arr[0]
    else:
        sv = shap_arr.flatten()

print(f"Extracted SHAP shape  : {sv.shape}  (should be ({len(engine.feature_names)},))")

shap_series = pd.Series(sv, index=engine.feature_names)
print("\nTop 10 SHAP features by absolute importance:")
print(shap_series.abs().nlargest(10).round(5).to_string())
print("\n✅ SHAP smoke test passed.")
