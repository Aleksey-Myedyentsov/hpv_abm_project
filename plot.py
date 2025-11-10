import csv
from collections import defaultdict
import matplotlib.pyplot as plt

INFILE = "out_all_scenarios_mean.csv"

def load_data(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["t"] = int(row["t"])
            for k in ["N","I_HR","I_LR","R","V","Prev","CancerCum"]:
                row[k] = float(row[k])
            rows.append(row)
    return rows

def group_by_scen(rows):
    by = defaultdict(list)
    for r in rows: by[r["scenario"]].append(r)
    for k in by: by[k].sort(key=lambda x: x["t"])
    return by

def plot_prevalence(by):
    plt.figure()
    for scen, ser in by.items():
        t = [r["t"] for r in ser]
        y = [100.0 * r["Prev"] for r in ser]
        plt.plot(t, y, label=scen)
    plt.xlabel("Years since intervention start"); plt.ylabel("HPV prevalence, %")
    plt.title("HPV prevalence over time by scenario")
    plt.legend(); plt.tight_layout(); plt.savefig("plot_prev_all.png"); plt.close()

def plot_vacc_cov(by):
    plt.figure()
    for scen, ser in by.items():
        t = [r["t"] for r in ser]
        y = [100.0 * (r["V"]/r["N"] if r["N"] else 0.0) for r in ser]
        plt.plot(t, y, label=scen)
    plt.xlabel("Years since intervention start"); plt.ylabel("Vaccination share, %")
    plt.title("Vaccination coverage over time")
    plt.legend(); plt.tight_layout(); plt.savefig("plot_vacc_coverage.png"); plt.close()

def plot_cancer_reduction(by):
    base = by.get("S0_no_intervention")
    if not base: return
    bm = {r["t"]: r["CancerCum"] for r in base}
    plt.figure()
    for scen, ser in by.items():
        if scen == "S0_no_intervention": continue
        t = [r["t"] for r in ser]
        red = []
        for r in ser:
            b = bm.get(r["t"], 0.0)
            red.append(0.0 if b <= 0 else 100.0 * (b - r["CancerCum"]) / b)
        plt.plot(t, red, label=scen)
    plt.xlabel("Years since intervention start"); plt.ylabel("Cancer reduction vs baseline, %")
    plt.title("Cervical cancer reduction vs S0 baseline")
    plt.legend(); plt.tight_layout(); plt.savefig("plot_cancer_reduction.png"); plt.close()

def main():
    rows = load_data(INFILE)
    by = group_by_scen(rows)
    plot_prevalence(by); plot_vacc_cov(by); plot_cancer_reduction(by)
    print("[plot] Done.")

if __name__ == "__main__":
    main()