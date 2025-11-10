import random
from dataclasses import dataclass
from typing import List, Dict
import networkx as nx

STATE_S = "S"
STATE_I_HR = "I_HR"
STATE_I_LR = "I_LR"
STATE_R = "R"


# --- minimal data collector for simulate.py compatibility ---
class _MiniDataCollector:
    def __init__(self, model):
        self.model = model
        self._rows = []
    def collect(self):
        self._rows.append(self.model.metrics())
    def get_model_vars_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self._rows)
@dataclass
class Agent:
    idx: int
    sex: str
    age: int
    state: str
    vaccinated: bool = False
    inf_timer: int = 0

class HPVModel:
    def __init__(
        self,
        N: int = 10_000,
        seed: int = 42,
        gamma: float = 2.5,
        assortativity: float = 0.15,
        p_transmission_hr: float = 0.35,
        p_transmission_lr: float = 0.25,
        hr_share: float = 0.6,
        mean_duration: float = 2.0,
        p_waning: float = 0.05,
        p_cancer: float = 0.005,
        cancer_risk_mult_if_vaccinated: float = 0.2,
        vaccine_eff: float = 0.9,
        vacc_age: int = 12,
        cov_f: float = 0.0,
        cov_m: float = 0.0,
        contacts_per_year: int = 20,
        max_age: int = 80,
        annual_birth_fraction: float = 0.04,
        catchup_years: int = 12,
        catchup_age_min: int = 12,
        catchup_age_max: int = 26,
        **kwargs,
    ):
        # ---- compatibility aliases (older simulate.py) ----
        if "vaccine_age" in kwargs: vacc_age = kwargs.pop("vaccine_age")
        if "coverage_f" in kwargs: cov_f = kwargs.pop("coverage_f")
        if "coverage_m" in kwargs: cov_m = kwargs.pop("coverage_m")
        if "coverageFemale" in kwargs: cov_f = kwargs.pop("coverageFemale")
        if "coverageMale" in kwargs: cov_m = kwargs.pop("coverageMale")
        # silently ignore any other kwargs

        self.N = N
        self.rng = random.Random(seed)
        self.params = dict(
            gamma=gamma,
            assortativity=assortativity,
            p_transmission_hr=p_transmission_hr,
            p_transmission_lr=p_transmission_lr,
            hr_share=hr_share,
            mean_duration=mean_duration,
            p_waning=p_waning,
            p_cancer=p_cancer,
            cancer_risk_mult_if_vaccinated=cancer_risk_mult_if_vaccinated,
            vaccine_eff=vaccine_eff,
            vacc_age=vacc_age,
            cov_f=cov_f, cov_m=cov_m,
            contacts_per_year=contacts_per_year,
            max_age=max_age,
            annual_birth_fraction=annual_birth_fraction,
            catchup_years=catchup_years,
            catchup_age_min=catchup_age_min,
            catchup_age_max=catchup_age_max,
        )
        self.agents: List[Agent] = []
        self.time = 0
        self.cancer_cum = 0
        self.intervention_years = 0
        self._init_population()
        self._init_network()


        # init datacollector and capture baseline (t=0)
        self.datacollector = _MiniDataCollector(self)
        self.datacollector.collect()
    def _init_population(self) -> None:
        for i in range(self.N):
            sex = "F" if self.rng.random() < 0.5 else "M"
            age = self.rng.randint(0, 50)
            r = self.rng.random()
            if r < 0.06:
                state = STATE_I_HR
            elif r < 0.10:
                state = STATE_I_LR
            else:
                state = STATE_S
            self.agents.append(Agent(i, sex, age, state))

    def _sample_degree(self, gamma: float) -> int:
        k_max = 50
        u = self.rng.random()
        val = 1.0 / (1.0 - u) ** (1.0 / (gamma - 1.0))
        return int(min(max(round(val) - 1, 0), k_max))

    def _init_network(self) -> None:
        gamma = self.params["gamma"]
        deg_seq = [self._sample_degree(gamma) for _ in range(len(self.agents))]
        if sum(deg_seq) % 2 == 1:
            deg_seq[0] += 1
        Gm = nx.configuration_model(deg_seq, seed=self.rng.randint(0, 10**9))
        G = nx.Graph(Gm); G.remove_edges_from(nx.selfloop_edges(G))
        self._apply_assortativity(G)
        self.G = G

    def _apply_assortativity(self, G: nx.Graph) -> None:
        p = self.params["assortativity"]
        if p <= 0.0 or G.number_of_edges() == 0: return
        degs = sorted(G.degree, key=lambda x: x[1])
        low = [n for n, _ in degs[: len(degs)//2]]
        high = [n for n, _ in degs[len(degs)//2:]]
        edges = list(G.edges()); self.rng.shuffle(edges)
        for (u, v) in edges[:int(p * len(edges))]:
            if (u in low and v in high) or (u in high and v in low):
                group = low if u in low else high
                cand = self.rng.choice(group)
                if cand != u and not G.has_edge(u, cand):
                    G.remove_edge(u, v); G.add_edge(u, cand)

    def step(self, vacc_enabled: bool = False) -> None:
        p = self.params
        self.time += 1
        if vacc_enabled:
            self.intervention_years += 1

        # aging
        for a in self.agents: a.age += 1

        # vaccination (routine + catch-up)
        if vacc_enabled:
            for a in self.agents:
                if a.age == p["vacc_age"] and not a.vaccinated:
                    cov = p["cov_f"] if a.sex == "F" else p["cov_m"]
                    if self.rng.random() < cov: a.vaccinated = True
            if self.intervention_years <= p["catchup_years"]:
                for a in self.agents:
                    if (not a.vaccinated) and (p["catchup_age_min"] <= a.age <= p["catchup_age_max"]):
                        cov = p["cov_f"] if a.sex == "F" else p["cov_m"]
                        if self.rng.random() < cov: a.vaccinated = True

        # safety: ensure some vaccinated in year 1 if coverage>0
        if vacc_enabled and self.intervention_years == 1 and not any(a.vaccinated for a in self.agents):
            for a in self.agents:
                if 12 <= a.age <= 26 and not a.vaccinated:
                    cov = p["cov_f"] if a.sex == "F" else p["cov_m"]
                    if self.rng.random() < cov: a.vaccinated = True

        # transmission (by positions)
        contacts = max(1, int(p["contacts_per_year"]))
        newly = []
        for u, v in self.G.edges():
            au, av = self.agents[u], self.agents[v]
            def try_inf(s_pos, inf_ag):
                s = self.agents[s_pos]
                if s.state != STATE_S or inf_ag.state not in (STATE_I_HR, STATE_I_LR): return
                eff = p["vaccine_eff"] if s.vaccinated else 0.0
                base = p["p_transmission_hr"] if inf_ag.state == STATE_I_HR else p["p_transmission_lr"]
                prob = 1.0 - (1.0 - base * (1.0 - eff)) ** contacts
                if self.rng.random() < prob:
                    newly.append((s_pos, STATE_I_HR if self.rng.random() < p["hr_share"] else STATE_I_LR))
            try_inf(u, av); try_inf(v, au)
        for pos, st in newly:
            if self.agents[pos].state == STATE_S:
                self.agents[pos].state = st
                self.agents[pos].inf_timer = 0

        # cancer BEFORE recovery (with protection for vaccinated)
        for a in self.agents:
            if (a.sex == "F") and (a.age >= 25) and (a.state == STATE_I_HR):
                risk = p["p_cancer"]
                if a.vaccinated:
                    risk *= p["cancer_risk_mult_if_vaccinated"]
                if self.rng.random() < risk:
                    self.cancer_cum += 1

        # natural history
        for a in self.agents:
            if a.state in (STATE_I_HR, STATE_I_LR):
                a.inf_timer += 1
                if self.rng.random() < (1.0 / p["mean_duration"]):
                    a.state = STATE_R; a.inf_timer = 0
            elif a.state == STATE_R and self.rng.random() < p["p_waning"]:
                a.state = STATE_S

        # demography
        max_age = p["max_age"]
        births_target = int(self.N * p["annual_birth_fraction"])
        to_remove = [a for a in self.agents if a.age > max_age]
        if len(to_remove) < births_target:
            rest = sorted([a for a in self.agents if a not in to_remove], key=lambda x: x.age, reverse=True)
            to_remove += rest[: births_target - len(to_remove)]
        remove_ids = {a.idx for a in to_remove}
        survivors = [a for a in self.agents if a.idx not in remove_ids]
        next_idx = (max(a.idx for a in survivors) + 1) if survivors else 0
        newborns = [Agent(next_idx+i, "F" if self.rng.random()<0.5 else "M", 0, STATE_S)
                    for i in range(len(to_remove))]
        self.agents = survivors + newborns
        self._init_network()

        # collect after each step
        self.datacollector.collect()


    def metrics(self) -> Dict[str, float]:
        I_HR = sum(1 for a in self.agents if a.state == STATE_I_HR)
        I_LR = sum(1 for a in self.agents if a.state == STATE_I_LR)
        R = sum(1 for a in self.agents if a.state == STATE_R)
        V = sum(1 for a in self.agents if a.vaccinated)
        N_now = float(len(self.agents)) if self.agents else 1.0
        return dict(t=int(self.time), 
            N=N_now, I_HR=float(I_HR), I_LR=float(I_LR), R=float(R), V=float(V),
            Prev=(I_HR + I_LR) / N_now, CancerCum=float(self.cancer_cum),
        )

# Backward-compat alias for older code paths
HPVNetworkModel = HPVModel
