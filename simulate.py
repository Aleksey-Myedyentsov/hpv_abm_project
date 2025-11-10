from hpv_abm.model import HPVNetworkModel
import csv
from statistics import mean
import random

YEARS_WARMUP = 10
YEARS_MAIN = 100
RUNS_PER_SCENARIO = 10

SCENARIOS = {
    "S0_no_intervention": dict(coverage_f=0.0,  coverage_m=0.0),
    "S1_girls_only_80":   dict(coverage_f=0.80, coverage_m=0.0),
    "S2_both_60":         dict(coverage_f=0.60, coverage_m=0.60),
    "S3_both_80":         dict(coverage_f=0.80, coverage_m=0.80),
}

def run_one(scen_name: str, coverage_f: float, coverage_m: float, seed: int):
    # Model receive vaccine_age/coverage_f/coverage_m
    m = HPVNetworkModel(
        seed=seed,
        vaccine_age=12,
        coverage_f=coverage_f,
        coverage_m=coverage_m,
        contacts_per_year=20,
        p_transmission_hr=0.35,
        p_transmission_lr=0.25,
        p_cancer=0.005,
        cancer_risk_mult_if_vaccinated=0.2,
        catchup_years=12,
        catchup_age_min=12,
        catchup_age_max=26,
    )

    # warming up without vaccination
    for _ in range(YEARS_WARMUP):
        m.step(vacc_enabled=False)

    # we turn on the intervention
    rows = []
    met0 = m.metrics(); met0["t"] = 0; rows.append(met0)

    m.step(vacc_enabled=(coverage_f > 0 or coverage_m > 0))
    met1 = m.metrics(); met1["t"] = 1; rows.append(met1)
    print(f"[DEBUG] {scen_name}: year1 vaccinated={int(met1['V'])} ({met1['V']/met1['N']*100:.2f}%), "
          f"prev={met1['Prev']*100:.2f}%")

    for t in range(2, YEARS_MAIN + 1):
        m.step(vacc_enabled=(coverage_f > 0 or coverage_m > 0))
        met = m.metrics(); met["t"] = t; rows.append(met)

    return rows

def aggregate_mean(runs):
    keys = ["N","I_HR","I_LR","R","V","Prev","CancerCum"]
    T = len(runs[0])
    avg = []
    for i in range(T):
        avg.append({"t": i, **{k: mean(run[i][k] for run in runs) for k in keys}})
    return avg

def main():
    all_rows = []
    for scen, cov in SCENARIOS.items():
        print(f"[RUN] {scen} ...")
        runs = []
        for r in range(RUNS_PER_SCENARIO):
            seed = random.randint(1, 10**9)
            runs.append(run_one(scen, cov["coverage_f"], cov["coverage_m"], seed))
        mean_rows = aggregate_mean(runs)
        for row in mean_rows:
            row["scenario"] = scen
            all_rows.append(row)
        with open(f"out_{scen}_mean.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=mean_rows[0].keys())
            w.writeheader(); w.writerows(mean_rows)

    with open("out_all_scenarios_mean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario","t","N","I_HR","I_LR","R","V","Prev","CancerCum"])
        w.writeheader(); w.writerows(all_rows)
    print("Done. Saved out_all_scenarios_mean.csv and per-scenario CSVs.")

if __name__ == "__main__":
    main()