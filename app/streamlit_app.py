"""
Patient Digital Twin — Streamlit Dashboard
==========================================
Improvements over v1:
  • All 4 interventions shown simultaneously on the risk trajectory chart
  • Monte Carlo simulation (N configurable runs) with ±1-std confidence bands
  • SHAP feature-importance waterfall panel for the initial patient state
  • KPI scorecards compare chosen intervention vs baseline (No treatment)
  • Clinical narrative auto-generated
  • Noise is fully reproducible (seeded RNG)
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── project root on sys.path so simulation package is importable ──────────────
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_dir)

from simulation.engine import DigitalTwinEngine  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Digital Twin",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      [data-testid="stMetricDeltaIcon-Up"]   { color: #ef553b !important; }
      [data-testid="stMetricDeltaIcon-Down"] { color: #00cc96 !important; }
      .block-container { padding-top: 1.5rem; }
      h1 { font-size: 1.8rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏥 Patient Digital Twin & Simulation")
st.caption(
    "Interact with the predictive digital twin to simulate patient "
    "readmission risk under various clinical interventions. "
    "Trajectories are derived from a calibrated XGBoost/Random Forest "
    "risk estimator trained on the UCI Diabetes 130-US dataset."
)

st.warning(
    "**⚠️ Medical Disclaimer:** This portal is designed solely for understanding Digital Twins as a concept "
    "and for experimental purposes. It does not replace professional medical advice, diagnosis, or treatment. "
    "The predictions and simulations provided are not guaranteed to be factual or represent true clinical outcomes. "
    "Always consult a qualified healthcare professional or doctor for medical decisions."
)

# ─────────────────────────────────────────────────────────────────────────────
# Load engine (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.experimental_singleton
def load_engine() -> DigitalTwinEngine:
    return DigitalTwinEngine(
        os.path.join(project_dir, 'models', 'risk_estimator.joblib'),
        os.path.join(project_dir, 'models', 'features.joblib'),
    )


engine = load_engine()

# ─────────────────────────────────────────────────────────────────────────────
# SHAP explainer (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.experimental_singleton
def load_shap_explainer():
    import shap
    import joblib

    # Prefer raw (uncalibrated) tree model for SHAP; fall back to main model
    base_path = os.path.join(project_dir, 'models', 'base_estimator.joblib')
    if os.path.exists(base_path):
        base_model = joblib.load(base_path)
    else:
        base_model = engine.model  # may be calibrated; TreeExplainer still works

    return shap.TreeExplainer(base_model)


@st.experimental_memo
def compute_shap_values(state_key: tuple) -> pd.Series:
    """Compute SHAP values for the initial patient state.

    state_key is a sorted tuple of (feature, value) pairs so Streamlit
    can hash it for caching.
    """
    explainer = load_shap_explainer()
    state_dict = dict(state_key)
    model_input = pd.DataFrame(
        [state_dict], columns=engine.feature_names
    ).fillna(0)

    shap_vals = explainer.shap_values(model_input)

    # Normalize across all SHAP / model version shape conventions:
    #  - RF (new SHAP):  ndarray of shape (n_samples, n_features, n_classes)
    #  - RF (old SHAP):  list of [neg_class, pos_class], each (n_samples, n_features)
    #  - XGB binary:     ndarray of shape (n_samples, n_features)
    import numpy as _np
    if isinstance(shap_vals, list):
        # Old-style list — take positive class, squeeze sample dim
        sv = _np.array(shap_vals[1]).squeeze()        # (n_features,)
    else:
        sv = _np.array(shap_vals)
        if sv.ndim == 3:
            # (n_samples, n_features, n_classes) → positive class, first sample
            sv = sv[0, :, 1]
        elif sv.ndim == 2:
            # Could be (n_samples, n_features) or (n_features, n_classes)
            if sv.shape[0] == 1:
                sv = sv[0]                            # (n_features,)
            elif sv.shape[1] == len(engine.feature_names):
                sv = sv[0]                            # (n_features,)
            else:
                # (n_features, n_classes) → positive class
                sv = sv[:, 1]
        else:
            sv = sv.flatten()                         # fallback

    return pd.Series(sv, index=engine.feature_names)



# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation (cached by input fingerprint)
# ─────────────────────────────────────────────────────────────────────────────
@st.experimental_memo
def run_all_mc(state_key: tuple, steps: int, n_runs: int, seed: int) -> pd.DataFrame:
    """Cache MC simulations keyed by patient state + simulation params."""
    return engine.simulate_all_mc(
        dict(state_key), steps=steps, n_runs=n_runs, seed=seed
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Patient State Inputs
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🧑‍⚕️ Demographics")

age_options = [
    '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)',
]
selected_age    = st.sidebar.selectbox("Age Group", age_options, index=6)
selected_gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])

st.sidebar.markdown("---")
st.sidebar.header("🏥 Hospital Encounter")
number_diagnoses  = st.sidebar.slider("Number of Diagnoses",        1, 16,  5)
time_in_hospital  = st.sidebar.slider("Days in Hospital",           1, 14,  4)
num_lab_procedures= st.sidebar.slider("Lab Procedures",             1, 132, 44)
num_procedures    = st.sidebar.slider("Surgical Procedures",        0, 6,   1)
num_medications   = st.sidebar.slider("Active Medications",         1, 80,  15)

discharge_risk    = st.sidebar.selectbox(
    "Discharge Destination",
    options=[0, 1, 2, 3],
    format_func=lambda x: [
        "0 — Home (lowest risk)",
        "1 — Home with care",
        "2 — Facility / Transfer (high risk)",
        "3 — Left AMA (highest risk)"
    ][x],
    index=0,
)
admission_type = st.sidebar.selectbox(
    "Admission Type",
    options=[0, 1, 2],
    format_func=lambda x: ['Elective', 'Urgent', 'Emergency'][x],
    index=2,
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Visit History")
number_inpatient   = st.sidebar.slider("Prior Inpatient Visits",  0, 20, 1)
number_outpatient  = st.sidebar.slider("Prior Outpatient Visits", 0, 20, 0)
number_emergency   = st.sidebar.slider("Prior Emergency Visits",  0, 20, 0)

st.sidebar.markdown("---")
st.sidebar.header("🧪 Lab Results")
hba1c_level = st.sidebar.selectbox(
    "HbA1c Result",
    options=[0, 1, 2, 3],
    format_func=lambda x: [
        "0 — Not tested",
        "1 — Normal",
        "2 — >7 (elevated)",
        "3 — >8 (severely elevated)"
    ][x],
    index=2,
)
glu_level = st.sidebar.selectbox(
    "Max Glucose Serum",
    options=[0, 1, 2, 3],
    format_func=lambda x: [
        "0 — Not tested",
        "1 — Normal",
        "2 — >200 mg/dL",
        "3 — >300 mg/dL"
    ][x],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.header("💊 Medications")
on_insulin    = st.sidebar.selectbox("Insulin",        [0,1,2], format_func=lambda x: ['Not prescribed','Steady','Dose changed'][x], index=0)
on_metformin  = st.sidebar.selectbox("Metformin",      [0,1,2], format_func=lambda x: ['Not prescribed','Steady','Dose changed'][x], index=1)
on_glipizide  = st.sidebar.selectbox("Glipizide",      [0,1,2], format_func=lambda x: ['Not prescribed','Steady','Dose changed'][x], index=0)
diabetes_med  = st.sidebar.checkbox("On Diabetes Medication",  value=True)
meds_changed  = st.sidebar.checkbox("Medication Regimen Changed This Visit", value=False)
drug_changes  = st.sidebar.slider("# Drugs with Dose Change", 0, 10, 0)
drugs_active  = st.sidebar.slider("# Active Drug Classes",    0, 22, 5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Simulation Settings")
steps  = st.sidebar.slider("Time Horizon (steps)", 1, 10, 5)
n_runs = st.sidebar.slider("Monte Carlo Runs",    10, 200, 50, step=10)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Focus Intervention")
INTERVENTIONS = DigitalTwinEngine.INTERVENTIONS
chosen_intervention = st.sidebar.selectbox(
    "Intervention to compare vs. Baseline",
    [iv for iv in INTERVENTIONS if iv != 'No treatment'],
)

# ─────────────────────────────────────────────────────────────────────────────
# Build initial patient state vector
# ─────────────────────────────────────────────────────────────────────────────
state: dict = {f: 0 for f in engine.feature_names}

# Demographics (OHE)
age_col    = f'age_{selected_age}'
gender_col = f'gender_{selected_gender}'
if age_col    in state: state[age_col]    = 1
if gender_col in state: state[gender_col] = 1

# Hospital encounter
state['time_in_hospital']   = time_in_hospital
state['num_lab_procedures'] = num_lab_procedures
state['num_procedures']     = num_procedures
state['num_medications']    = num_medications
state['number_diagnoses']   = number_diagnoses
state['discharge_risk']     = discharge_risk
state['admission_type']     = admission_type

# Visit history
state['number_inpatient']   = number_inpatient
state['number_outpatient']  = number_outpatient
state['number_emergency']   = number_emergency

# Lab results (now ordinal 0-3)
state['HbA1c_result']       = hba1c_level
state['max_glu_serum']      = glu_level

# Medications
state['insulin']            = on_insulin
state['metformin']          = on_metformin
state['glipizide']          = on_glipizide
state['diabetesMed']        = 1 if diabetes_med else 0
state['change']             = 1 if meds_changed else 0
state['drug_changes_count'] = drug_changes
state['drugs_active_count'] = drugs_active

# Hashable key for Streamlit caching
state_key = tuple(sorted(state.items()))

# ─────────────────────────────────────────────────────────────────────────────
# Run simulations
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Running {n_runs} Monte Carlo simulations × 4 interventions…"):
    mc_all = run_all_mc(state_key, steps=steps, n_runs=n_runs, seed=42)

# Extract final-step summaries for KPI cards
def final_summary(intervention: str) -> dict:
    """Return the mean end-of-sim risk and a single seeded trajectory's final state."""
    mc_row = mc_all[
        (mc_all['intervention'] == intervention) &
        (mc_all['time_step'] == f't{steps}')
    ].iloc[0]

    # Get a deterministic final state for non-risk metrics
    traj = engine.simulate_trajectory(state, intervention, steps=steps, seed=42)
    return {
        'mean_risk':  float(mc_row['mean_risk']),
        'upper':      float(mc_row['upper']),
        'lower':      float(mc_row['lower']),
        'final_state': traj.iloc[-1]['state'],
    }


baseline_sum = final_summary('No treatment')
target_sum   = final_summary(chosen_intervention)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Monte Carlo Risk Trajectory (all 4 arms)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📈 Monte Carlo Risk Trajectories — All Interventions")
st.caption(
    f"Each arm shows the mean risk ± 1 std across **{n_runs} independent runs**. "
    "Shaded bands represent the natural variability from stochastic dynamics."
)

COLORS = DigitalTwinEngine.INTERVENTION_COLORS
fig_mc = go.Figure()

# Convert raw [0,1] probabilities → percentage [0,100] BEFORE plotting.
# This avoids the Plotly tickformat='.0%' double-multiplication bug where
# the axis labels and polygon y-values end up in different units.
def to_pct(series):
    return (series * 100).clip(0, 100)

y_max = float(to_pct(mc_all['upper']).max())

for intervention in INTERVENTIONS:
    grp = mc_all[mc_all['intervention'] == intervention].reset_index(drop=True)
    solid, fill = COLORS[intervention]

    upper_pct = to_pct(grp['upper']).tolist()
    lower_pct = to_pct(grp['lower']).tolist()
    mean_pct  = to_pct(grp['mean_risk']).tolist()
    time_steps = grp['time_step'].tolist()

    # Confidence-band fill (upper → lower, closed polygon)
    fig_mc.add_trace(go.Scatter(
        x=time_steps + time_steps[::-1],
        y=upper_pct  + lower_pct[::-1],
        fill='toself',
        fillcolor=fill,
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip',
        name=intervention,
    ))

    # Mean line
    fig_mc.add_trace(go.Scatter(
        x=time_steps,
        y=mean_pct,
        mode='lines+markers',
        name=intervention,
        line=dict(color=solid, width=3),
        marker=dict(size=8, color=solid),
        hovertemplate=(
            f'<b>{intervention}</b><br>'
            'Step: %{x}<br>'
            'Risk: %{y:.1f}%'
            '<extra></extra>'
        ),
    ))

fig_mc.update_layout(
    xaxis_title="Time Step",
    yaxis_title="30-Day Readmission Risk (%)",
    yaxis=dict(
        ticksuffix="%",
        range=[0, min(100, y_max * 1.15)],   # % scale, hard-capped at 100%
    ),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    height=420,
    margin=dict(t=20, b=40),
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)
fig_mc.update_xaxes(showgrid=False)
fig_mc.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

st.plotly_chart(fig_mc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — KPI Scorecards (chosen intervention vs baseline)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header(f"🩺 Twin Vital Signs — *{chosen_intervention}* vs Baseline")
st.caption("Values at the end of the simulation horizon (deterministic, seed=42).")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

risk_delta = target_sum['mean_risk'] - baseline_sum['mean_risk']
col_m1.metric(
    "Readmission Risk",
    f"{target_sum['mean_risk']*100:.1f}%",
    delta=f"{risk_delta*100:+.1f}%",
    delta_color="inverse",
    help="Mean MC risk at final time step (30-day readmission probability)",
)

b_state = baseline_sum['final_state']
t_state = target_sum['final_state']

meds_delta = t_state['num_medications'] - b_state['num_medications']
col_m2.metric(
    "Active Medications",
    f"{t_state['num_medications']:.1f}",
    delta=f"{meds_delta:+.1f}",
    delta_color="inverse",
)

inp_delta = t_state['number_inpatient'] - b_state['number_inpatient']
col_m3.metric(
    "Inpatient Visits",
    f"{t_state['number_inpatient']:.1f}",
    delta=f"{inp_delta:+.1f}",
    delta_color="inverse",
)

hba1c_b = b_state.get('_hba1c_real', 7.5 if b_state.get('HbA1c_result', 0) == 1 else 6.0)
hba1c_t = t_state.get('_hba1c_real', 7.5 if t_state.get('HbA1c_result', 0) == 1 else 6.0)
hba1c_delta = hba1c_t - hba1c_b
col_m4.metric(
    "HbA1c Level",
    f"{hba1c_t:.2f}%",
    delta=f"{hba1c_delta:+.2f}%",
    delta_color="inverse",
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — SHAP Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🧠 SHAP Feature Importance — Initial Patient State")
st.caption(
    "SHAP (SHapley Additive exPlanations) values show how each feature "
    "**pushes the risk up (🔴) or down (🟢)** relative to the model's average prediction."
)

try:
    with st.spinner("Computing SHAP values…"):
        shap_series = compute_shap_values(state_key)

    top_n = 15
    top_shap = shap_series.abs().nlargest(top_n).index
    plot_shap = shap_series[top_shap].sort_values()

    colors_shap = ['#ef553b' if v > 0 else '#00cc96' for v in plot_shap.values]

    fig_shap = go.Figure(go.Bar(
        x=plot_shap.values,
        y=plot_shap.index,
        orientation='h',
        marker_color=colors_shap,
        hovertemplate='<b>%{y}</b><br>SHAP value: %{x:.4f}<extra></extra>',
    ))
    fig_shap.add_vline(x=0, line_color='rgba(128,128,128,0.6)', line_width=1)
    fig_shap.update_layout(
        xaxis_title="SHAP Value (impact on readmission probability)",
        yaxis_title="",
        height=max(350, top_n * 28),
        margin=dict(t=10, b=40, l=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig_shap.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig_shap.update_yaxes(showgrid=False)
    st.plotly_chart(fig_shap, use_container_width=True)

    with st.expander("📋 Full SHAP table"):
        full_shap = (
            shap_series
            .rename("SHAP Value")
            .to_frame()
            .assign(Direction=lambda df: df['SHAP Value'].apply(
                lambda v: "↑ Increases risk" if v > 0 else "↓ Reduces risk"
            ))
            .sort_values('SHAP Value', key=abs, ascending=False)
        )
        st.dataframe(full_shap.style.format({'SHAP Value': '{:.5f}'}), height=300)

except Exception as e:
    st.warning(
        f"⚠️ SHAP computation unavailable: {e}\n\n"
        "Retrain the model (`python models/train_model.py`) and restart the app."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Clinical Narrative
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📝 Clinical Summary & Takeaways")

action_word = "improved" if risk_delta < 0 else "worsened"
color = "#00cc96" if risk_delta < 0 else "#ef553b"
ci_half = (target_sum['upper'] - target_sum['lower']) / 2

st.markdown(
    f"""
    <div style="background-color: rgba(128,128,128,0.1); border-left: 5px solid {color}; padding: 1.5rem; border-radius: 5px; margin-bottom: 2rem;">
        <h4 style="margin-top: 0;">Simulation Results: {chosen_intervention}</h4>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            Implementing <strong>{chosen_intervention}</strong> {action_word} the projected 30-day readmission risk by 
            <strong style="color: {color};">{risk_delta*100:+.1f}%</strong> relative to baseline.
        </p>
        <ul style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 0;">
            <li><strong>End-of-simulation risk:</strong> {target_sum['mean_risk']*100:.1f}% <em>(±{ci_half*100:.1f}% across {n_runs} Monte Carlo runs)</em></li>
            <li><strong>Active medications:</strong> {t_state['num_medications']:.1f}</li>
            <li><strong>Projected HbA1c:</strong> {hba1c_t:.2f}%</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
