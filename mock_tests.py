"""
Mock test cases for the Patient Digital Twin.
Run from the project root:
    ~/miniconda3/envs/myenv/bin/python3 mock_tests.py
"""

import sys
sys.path.insert(0, '.')

from simulation.engine import DigitalTwinEngine

engine = DigitalTwinEngine('models/risk_estimator.joblib', 'models/features.joblib')


def run_test(name: str, overrides: dict, steps: int = 5):
    state = {f: 0 for f in engine.feature_names}
    state.update(overrides)

    mc = engine.simulate_all_mc(state, steps=steps, n_runs=50, seed=42)

    print(f"\n{'='*65}")
    print(f"  TEST: {name}")
    print(f"{'='*65}")

    t0_risk = mc[mc['time_step'] == 't0']['mean_risk'].mean()
    print(f"  Baseline risk at t0: {t0_risk*100:.1f}%\n")

    for iv in engine.INTERVENTIONS:
        grp  = mc[mc['intervention'] == iv]
        t_end = grp[grp['time_step'] == f't{steps}'].iloc[0]
        delta = t_end['mean_risk'] - t0_risk
        ci    = (t_end['upper'] - t_end['lower']) / 2
        flag  = "✅ improves" if delta < -0.005 else ("⚠️ worsens " if delta > 0.01 else "➡️  neutral ")
        print(
            f"  {flag}  [{iv:22s}]  "
            f"t{steps} = {t_end['mean_risk']*100:5.1f}%  "
            f"Δ = {delta*100:+5.1f}%  "
            f"CI ±{ci*100:.1f}%"
        )


# ── CASE 1: Young healthy patient ─────────────────────────────────────────────
run_test(
    "Young Healthy Patient (age 30-40, no comorbidities)",
    {
        'num_medications':  3,
        'number_inpatient': 0,
        'number_emergency': 0,
        'HbA1c_result':     0,
        'insulin':          0,
        'diabetesMed':      0,
    }
)

# ── CASE 2: Typical diabetic outpatient ───────────────────────────────────────
run_test(
    "Typical Diabetic (age 60-70, moderate meds, high HbA1c)",
    {
        'num_medications':  15,
        'number_inpatient':  1,
        'number_emergency':  0,
        'HbA1c_result':      1,
        'insulin':           0,
        'diabetesMed':       1,
    }
)

# ── CASE 3: High-risk frequent flyer ─────────────────────────────────────────
run_test(
    "High-Risk Patient (many prior visits, poor glycaemic control)",
    {
        'num_medications':  25,
        'number_inpatient':  5,
        'number_emergency':  3,
        'HbA1c_result':      1,
        'insulin':           1,
        'diabetesMed':       1,
    }
)

# ── CASE 4: Over-medicated (diminishing returns) ──────────────────────────────
run_test(
    "Over-medicated Patient (40 meds — diminishing returns expected)",
    {
        'num_medications':  40,
        'number_inpatient':  8,
        'number_emergency':  5,
        'HbA1c_result':      1,
        'insulin':           1,
        'diabetesMed':       1,
    }
)

print("\n" + "="*65)
print("  All mock tests complete.")
print("="*65)
