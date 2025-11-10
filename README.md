# HPV ABM (Power-law network, assortative mixing, vaccination scenarios)

## Quickstart

```bash
git clone <your_repo_url>
cd hpv_abm_project-2

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
export PYTHONPATH=$(pwd)

python simulate.py
python plot.py

Model parameters
HPVModel(
    N=10_000,                   # population size
    seed=42,                    # random seed
    gamma=2.5,                  # power-law degree exponent
    assortativity=0.15,         # degree assortativity

    # Transmission
    p_transmission_hr=0.35,
    p_transmission_lr=0.25,
    hr_share=0.6,
    mean_duration=2.0,
    p_waning=0.05,

    # Cancer
    p_cancer=0.005,
    cancer_risk_mult_if_vaccinated=0.2,

    # Vaccination
    vaccine_eff=0.9,
    vacc_age=12,
    cov_f=0.0,
    cov_m=0.0,
    catchup_years=12,
    catchup_age_min=12,
    catchup_age_max=26,

    # Demography & contacts
    contacts_per_year=20,
    max_age=80,
    annual_birth_fraction=0.04
)

Typical usage
from hpv_abm import HPVModel

m = HPVModel(seed=1)
for _ in range(5):
    m.step(vacc_enabled=False)

m.params["cov_f"] = 0.8
m.params["cov_m"] = 0.0

m.step(vacc_enabled=True)
print(m.metrics())

