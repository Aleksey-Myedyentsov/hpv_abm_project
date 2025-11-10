# simulate.py
from hpv_abm.model import HPVNetworkModel
import pandas as pd
import numpy as np

# Scenarios (tau_f, tau_m)
SCENARIOS = {
    "S0_no_intervention": dict(coverage_f=0.0, coverage_m=0.0),
    "S1_girls_only_80":   dict(coverage_f=0.8, coverage_m=0.0),
    "S2_both_60":         dict(coverage_f=0.6, coverage_m=0.6),
    "S3_both_80":         dict(coverage_f=0.8, coverage_m=0.8),
}

def run_scenario(name: str, params: dict, N=10000, years_total=50, burn_in=12, runs=30, seed=42):
    """
    - Burn-in (no vaccination) for 'burn_in' years
    - Then enable vaccination according to params for the remaining years
    - Repeat 'runs' times, average metrics per (t)
    """
    all_frames = []
    for r in range(runs):
        # 1) Burn-in with no vaccination
        model = HPVNetworkModel(
            N=N,
            coverage_f=0.0, coverage_m=0.0,
            burn_in_years=burn_in,
            seed=seed + r
        )
        burn_df = model.run(years=burn_in)

        # 2) Apply scenario coverages and continue
        model.coverage_f = params["coverage_f"]
        model.coverage_m = params["coverage_m"]

        # Remaining years
        remain = max(0, years_total - burn_in)
        inter_df = model.run(years=remain)

        df = pd.concat([burn_df, inter_df.iloc[1:, :]], ignore_index=True)  # avoid duplicate t=burn_in
        df["run"] = r
        df["scenario"] = name
        all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)
    # Aggregate over runs: mean (and std if needed)
    group_cols = ["scenario", "t"]
    agg = full.groupby(group_cols).agg({
        "N": "mean",
        "I_HR": "mean",
        "I_LR": "mean",
        "R": "mean",
        "V": "mean",
        "Prev": "mean",
        "CancerCum": "mean"
    }).reset_index()

    # Save raw and aggregated
    full.to_csv(f"out_{name}_raw.csv", index=False)
    agg.to_csv(f"out_{name}_mean.csv", index=False)
    return agg

def run_all():
    results = []
    for name, params in SCENARIOS.items():
        print(f"[RUN] {name} ...")
        agg = run_scenario(name, params)
        results.append(agg)
    all_agg = pd.concat(results, ignore_index=True)
    all_agg.to_csv("out_all_scenarios_mean.csv", index=False)
    print("Done. Saved out_all_scenarios_mean.csv and per-scenario CSVs.")

if __name__ == "__main__":
    run_all()