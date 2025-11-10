from hpv_abm.model import HPVNetworkModel
import pandas as pd
import random

# Scenarios: coverage for females/males (shares 0..1)
SCENARIOS = {
    "S0_no_intervention": dict(cov_f=0.0, cov_m=0.0),
    "S1_girls_only_80":   dict(cov_f=0.8, cov_m=0.0),
    "S2_both_60":         dict(cov_f=0.6, cov_m=0.6),
    "S3_both_80":         dict(cov_f=0.8, cov_m=0.8),
}

YEARS_TOTAL = 50
BURN_IN = 12          # years with no vaccination
RUNS = 30
N = 10_000
SEED0 = 42

# One-time catch-up window right after burn-in
CATCHUP_ENABLE = True
VACCINE_AGE = 12
CATCHUP_MAX_AGE = 15  # vaccinate [12..15] once when intervention starts


def _run_years(model: HPVNetworkModel, years: int) -> pd.DataFrame:
    """Run model for N years and return ONLY the rows added during this call."""
    if years <= 0:
        df0 = model.datacollector.get_model_vars_dataframe().copy()
        return df0.iloc[0:0, :]
    if hasattr(model, "run") and callable(getattr(model, "run")):
        before = len(model.datacollector.get_model_vars_dataframe())
        model.run(years=years)
        after_df = model.datacollector.get_model_vars_dataframe()
        return after_df.iloc[before:, :].reset_index(drop=True)
    # fallback
    for _ in range(years):
        model.step()
    df = model.datacollector.get_model_vars_dataframe().copy()
    return df.iloc[-years:, :].reset_index(drop=True)


def _apply_catchup(model: HPVNetworkModel):
    """One-time catch-up right after burn-in."""
    if not CATCHUP_ENABLE:
        return
    cov_f = getattr(model, "cov_f", 0.0)
    cov_m = getattr(model, "cov_m", 0.0)
    if cov_f <= 0.0 and cov_m <= 0.0:
        return

    changed = 0
    for a in model.agents:
        if a.vaccinated:
            continue
        if VACCINE_AGE <= a.age <= CATCHUP_MAX_AGE:
            p = cov_f if a.sex == "F" else cov_m
            if random.random() < p:
                a.vaccinated = True
                changed += 1
    # можно залогировать при отладке:
    # print(f"[catchup] vaccinated {changed}")


def run_scenario(name: str, params: dict,
                 N: int = N, years_total: int = YEARS_TOTAL,
                 burn_in: int = BURN_IN, runs: int = RUNS, seed: int = SEED0) -> pd.DataFrame:
    all_frames = []

    for r in range(runs):
        # 1) burn-in без вакцинации
        model = HPVNetworkModel(
            N=N,
            vacc_age=VACCINE_AGE,
            cov_f=0.0,
            cov_m=0.0,
            seed=seed + r,
        )
        burn_df = _run_years(model, burn_in)

        # 2) включаем сценарные coverage
        model.cov_f = params["cov_f"]
        model.cov_m = params["cov_m"]

        # 3) единоразовый catch-up (12..15)
        _apply_catchup(model)

        # 4) остаток лет
        remain = max(0, years_total - burn_in)
        inter_df = _run_years(model, remain)

        df = pd.concat([burn_df, inter_df], ignore_index=True)
        df["run"] = r
        df["scenario"] = name
        all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)

    # mean по ранам
    group_cols = ["scenario", "t"]
    keep_cols = ["N", "I_HR", "I_LR", "R", "V", "Prev", "CancerCum"]
    existing = [c for c in keep_cols if c in full.columns]
    agg = full.groupby(group_cols, as_index=False)[existing].mean()

    full.to_csv(f"out_{name}_raw.csv", index=False)
    agg.to_csv(f"out_{name}_mean.csv", index=False)
    return agg


def run_all():
    results = []
    for name, params in SCENARIOS.items():
        print(f"[RUN] {name} ...")
        results.append(run_scenario(name, params))
    all_agg = pd.concat(results, ignore_index=True)
    all_agg.to_csv("out_all_scenarios_mean.csv", index=False)
    print("Done. Saved out_all_scenarios_mean.csv and per-scenario CSVs.")


if __name__ == "__main__":
    run_all()