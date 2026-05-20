import pandas as pd
import numpy as np
import joblib
import os


class DigitalTwinEngine:
    """Patient Digital Twin simulation engine.

    Wraps a trained readmission-risk model and steps patient clinical
    state forward under different intervention scenarios using rule-based
    transition dynamics with Gaussian noise. All randomness is seeded for
    full reproducibility.
    """

    INTERVENTIONS = [
        'No treatment',
        'Medication added',
        'Lifestyle improvement',
        'Poor adherence',
    ]

    INTERVENTION_COLORS = {
        'No treatment':        ('#ef553b', 'rgba(239,85,59,0.15)'),
        'Medication added':    ('#00cc96', 'rgba(0,204,150,0.15)'),
        'Lifestyle improvement': ('#636efa', 'rgba(99,110,250,0.15)'),
        'Poor adherence':      ('#ffa15a', 'rgba(255,161,90,0.15)'),
    }

    def __init__(self, model_path, features_path):
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model_input(self, state: dict) -> pd.DataFrame:
        """Create a model-compatible DataFrame row from a state dict.
        Extra internal keys (e.g. '_hba1c_real') are silently dropped."""
        return pd.DataFrame([state], columns=self.feature_names).fillna(0)

    # ------------------------------------------------------------------
    # Transition dynamics
    # ------------------------------------------------------------------

    def transition(self, state: dict, intervention: str, rng=None) -> dict:
        """Advance patient state by one time step under a given intervention.

        Args:
            state:        dict of current patient feature values.
                          May contain '_hba1c_real' (hidden continuous HbA1c).
            intervention: one of INTERVENTIONS
            rng:          np.random.Generator for reproducible noise.
                          If None, creates a new unseeded generator.
        Returns:
            new_state dict
        """
        if rng is None:
            rng = np.random.default_rng()

        new_state = state.copy()
        noise = rng.normal(0, 0.2)

        # General medication count drift (small random walk)
        new_state['num_medications'] = max(
            0, new_state['num_medications'] + rng.normal(0, 0.5)
        )

        # Hidden continuous HbA1c drives the binary HbA1c_result feature
        current_hba1c = new_state.get(
            '_hba1c_real',
            7.5 if state['HbA1c_result'] == 1 else 6.0
        )

        if intervention == 'No treatment':
            current_hba1c += 0.3 + noise
            new_state['number_inpatient'] = max(0, new_state['number_inpatient'] + 0.1)

        elif intervention == 'Medication added':
            new_state['insulin'] = 1
            current_hba1c -= max(0.2, 0.7 - noise)   # always at least some benefit
            new_state['num_medications'] = min(80, new_state['num_medications'] + 1)

        elif intervention == 'Lifestyle improvement':
            current_hba1c -= max(0.1, 0.5 - noise)
            new_state['number_outpatient'] = max(0, new_state['number_outpatient'] - 0.5)
            new_state['number_inpatient']  = max(0, new_state['number_inpatient']  - 0.5)

        elif intervention == 'Poor adherence':
            current_hba1c += 0.5 + noise
            new_state['number_emergency']  = max(0, new_state['number_emergency']  + 0.5)
            new_state['number_inpatient']  = max(0, new_state['number_inpatient']  + 0.5)

        new_state['_hba1c_real'] = current_hba1c
        new_state['HbA1c_result'] = 1 if current_hba1c > 7.0 else 0
        return new_state

    # ------------------------------------------------------------------
    # Single-run trajectory
    # ------------------------------------------------------------------

    def simulate_trajectory(
        self,
        initial_state: dict,
        intervention: str,
        steps: int = 5,
        seed=None,
    ) -> pd.DataFrame:
        """Single seeded trajectory.

        Returns:
            DataFrame with columns [time_step, risk, state]
        """
        rng = np.random.default_rng(seed)
        trajectory = []
        state = initial_state.copy()

        for t in range(steps + 1):
            model_input = self._build_model_input(state)
            risk = float(self.model.predict_proba(model_input)[0, 1])
            trajectory.append({
                'time_step': f't{t}',
                'risk': risk,
                'state': state.copy(),
            })
            if t < steps:
                state = self.transition(state, intervention, rng=rng)

        return pd.DataFrame(trajectory)

    # ------------------------------------------------------------------
    # Monte Carlo trajectory
    # ------------------------------------------------------------------

    def simulate_trajectory_mc(
        self,
        initial_state: dict,
        intervention: str,
        steps: int = 5,
        n_runs: int = 50,
        seed=None,
    ) -> pd.DataFrame:
        """Monte Carlo simulation: mean ± 1-std risk band at each time step.

        Args:
            n_runs: number of independent MC replications
            seed:   master seed for full reproducibility
        Returns:
            DataFrame with columns [time_step, mean_risk, upper, lower]
        """
        master_rng = np.random.default_rng(seed)
        all_risks = []

        for _ in range(n_runs):
            run_seed = int(master_rng.integers(0, 1_000_000))
            traj = self.simulate_trajectory(
                initial_state, intervention, steps=steps, seed=run_seed
            )
            all_risks.append(traj['risk'].values)

        all_risks = np.array(all_risks)   # shape: (n_runs, steps+1)
        mean_risk = all_risks.mean(axis=0)
        std_risk  = all_risks.std(axis=0)

        return pd.DataFrame({
            'time_step': [f't{t}' for t in range(steps + 1)],
            'mean_risk': mean_risk,
            'upper': np.clip(mean_risk + std_risk, 0, 1),
            'lower': np.clip(mean_risk - std_risk, 0, 1),
        })

    def simulate_all_mc(
        self,
        initial_state: dict,
        steps: int = 5,
        n_runs: int = 50,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run Monte Carlo simulation for all 4 interventions.

        Returns:
            Combined DataFrame with an additional 'intervention' column.
        """
        master_rng = np.random.default_rng(seed)
        results = []
        for intervention in self.INTERVENTIONS:
            iv_seed = int(master_rng.integers(0, 1_000_000))
            mc_df = self.simulate_trajectory_mc(
                initial_state, intervention,
                steps=steps, n_runs=n_runs, seed=iv_seed,
            )
            mc_df['intervention'] = intervention
            results.append(mc_df)
        return pd.concat(results, ignore_index=True)


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing digital twin simulation engine...")
    project_dir = os.path.dirname(os.path.dirname(__file__))
    engine = DigitalTwinEngine(
        os.path.join(project_dir, 'models', 'risk_estimator.joblib'),
        os.path.join(project_dir, 'models', 'features.joblib'),
    )

    dummy_state = {f: 0 for f in engine.feature_names}
    dummy_state['num_medications']  = 15
    dummy_state['number_inpatient'] = 1
    dummy_state['number_emergency'] = 1
    dummy_state['HbA1c_result']     = 1
    dummy_state['insulin']          = 0

    print("\n--- Monte Carlo: No Treatment (50 runs) ---")
    mc = engine.simulate_trajectory_mc(dummy_state, 'No treatment', seed=42)
    print(mc[['time_step', 'mean_risk', 'lower', 'upper']])

    print("\n--- All 4 Interventions (pivot view) ---")
    all_mc = engine.simulate_all_mc(dummy_state, seed=42)
    pivot = all_mc.pivot(index='time_step', columns='intervention', values='mean_risk')
    print(pivot.round(3))

    print("\nEngine test successful!")
