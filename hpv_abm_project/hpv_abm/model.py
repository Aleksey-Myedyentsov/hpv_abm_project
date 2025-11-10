# hpv_abm/model.py
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import networkx as nx

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


# -----------------------------
# Agent
# -----------------------------
class HPVAgent(Agent):
    """
    Agent with:
      - sex: 'M'/'F'
      - age: int (years)
      - state: 'S', 'I_LR', 'I_HR', 'R'
      - vaccinated: bool
      - infection_years: years elapsed in current infection (for display; recovery is stochastic)
      - cancer_flag: whether developed cancer (cumulative)
      - activity: 'low'/'mid'/'high' (derived from degree quantiles)
    """
    def __init__(self, unique_id: int, model: "HPVNetworkModel", sex: str, age: int):
        super().__init__(unique_id, model)
        self.sex = sex
        self.age = age
        self.state = 'S'
        self.vaccinated = False
        self.infection_years = 0
        self.cancer_flag = False
        self.activity = 'low'

    # --- helpers ---
    def infectious(self) -> bool:
        return self.state in ('I_LR', 'I_HR')

    def susceptibility_multiplier(self) -> float:
        # vaccine efficacy reduces susceptibility (but does not alter infectiousness here)
        if self.vaccinated:
            return max(0.0, 1.0 - self.model.vaccine_efficacy)
        return 1.0

    # --- per-step logic ---
    def step(self):
        # Ageing (annual step)
        self.age += 1

        # Vaccination at 12
        if self.age == 12 and not self.vaccinated:
            cov = self.model.coverage_f if self.sex == 'F' else self.model.coverage_m
            if self.random.random() < cov:
                self.vaccinated = True

        # Infection course: recover with prob gamma; R (immune) wanes after R_duration_years
        if self.infectious():
            self.infection_years += 1
            if self.random.random() < self.model.gamma_recovery:
                self.state = 'R'
                self.infection_years = 0
        elif self.state == 'R':
            # time in R is tracked implicitly by "age since R"? we approximate by fixed duration
            # Use agent-level memory via infection_years as "years since entering R"
            self.infection_years += 1
            if self.infection_years >= self.model.R_duration_years:
                self.state = 'S'
                self.infection_years = 0

        # Cancer progression: F, age >= 25, in I_HR
        if self.sex == 'F' and self.age >= 25 and self.state == 'I_HR' and not self.cancer_flag:
            if self.random.random() < self.model.p_cancer_per_year_HR:
                self.cancer_flag = True


# -----------------------------
# Model
# -----------------------------
class HPVNetworkModel(Model):
    """
    Agent-based HPV model on an assortative power-law-like contact network.
    Key params:
      N: population size
      T: total years to simulate
      vaccine_efficacy (epsilon)
      coverage_f, coverage_m (tau_f, tau_m)
      beta_f, beta_m: per-contact transmission probabilities (recipient sex specific)
      hr_fraction: prob(new infection is HR type)
      gamma_recovery: annual recovery prob
      R_duration_years: immunity waning
      p_cancer_per_year_HR: annual cancer prob for F>=25 in I_HR
      burn_in_years: years with no vaccination (Scenario 0 warm-up)
    """
    def __init__(
        self,
        N: int = 10000,
        vaccine_efficacy: float = 0.90,
        coverage_f: float = 0.0,
        coverage_m: float = 0.0,
        beta_f: float = 0.08,     # prob for female recipient
        beta_m: float = 0.08,     # prob for male recipient
        hr_fraction: float = 0.6, # share of HR among new infections
        gamma_recovery: float = 0.25,
        R_duration_years: int = 2,
        p_cancer_per_year_HR: float = 0.001,
        initial_prev: float = 0.01,   # 1% infected at init (HR)
        powerlaw_gamma: float = 2.5,
        mean_degree: float = 3.0,
        assort_within_activity: float = 0.85,   # rewiring preference
        burn_in_years: int = 12,  # run w/o vaccination to reach a baseline
        seed: Optional[int] = 42
    ):
        super().__init__(seed=seed)
        self.N = N
        self.vaccine_efficacy = vaccine_efficacy
        self.coverage_f = coverage_f
        self.coverage_m = coverage_m
        self.beta_f = beta_f
        self.beta_m = beta_m
        self.hr_fraction = hr_fraction
        self.gamma_recovery = gamma_recovery
        self.R_duration_years = R_duration_years
        self.p_cancer_per_year_HR = p_cancer_per_year_HR
        self.initial_prev = initial_prev
        self.powerlaw_gamma = powerlaw_gamma
        self.mean_degree = mean_degree
        self.assort_within_activity = assort_within_activity
        self.burn_in_years = burn_in_years

        self.schedule = RandomActivation(self)
        self.G = nx.Graph()

        self._init_population()
        self._build_powerlaw_network()
        self._assign_activity_by_degree()
        self._rewire_for_assortativity()

        self.t = 0
        self.datacollector = DataCollector(
            model_reporters={
                "t": lambda m: m.t,
                "N": lambda m: m.N,
                "I_HR": lambda m: sum(1 for a in m.schedule.agents if a.state == 'I_HR'),
                "I_LR": lambda m: sum(1 for a in m.schedule.agents if a.state == 'I_LR'),
                "R": lambda m: sum(1 for a in m.schedule.agents if a.state == 'R'),
                "V": lambda m: sum(1 for a in m.schedule.agents if a.vaccinated),
                "Prev": lambda m: (sum(1 for a in m.schedule.agents if a.state in ('I_HR','I_LR')) / m.N),
                "CancerCum": lambda m: sum(1 for a in m.schedule.agents if a.cancer_flag),
            }
        )

    # ---------- Population ----------
    def _init_population(self):
        for uid in range(self.N):
            sex = 'F' if self.random.random() < 0.5 else 'M'
            age = self.random.randrange(0, 51)  # 0..50
            agent = HPVAgent(uid, self, sex, age)

            # initial infection: 1% I_HR
            if self.random.random() < self.initial_prev:
                agent.state = 'I_HR'
            self.schedule.add(agent)
            self.G.add_node(uid)

    # ---------- Network ----------
    def _build_powerlaw_network(self):
        """
        Build a simple undirected network with a power-law-like degree sequence
        tuned to approximate 'mean_degree'. We use a truncated power-law sampler.
        """
        rng = np.random.default_rng(self.random.randint(0, 10**9))
        # sample degrees from truncated power law (k>=1)
        # P(k) ~ k^-gamma
        def sample_degree():
            # inverse transform sampling over [1, k_max]
            k_min, k_max = 1, max(5, int(2 * self.mean_degree) * 10)
            u = rng.uniform()
            if self.powerlaw_gamma == 1:
                k = int(np.floor(np.exp(u * np.log(k_max / k_min)) * k_min))
            else:
                g = self.powerlaw_gamma
                c = (k_max**(1-g) - k_min**(1-g))
                x = (u * c + k_min**(1-g)) ** (1/(1-g))
                k = int(np.clip(np.floor(x), k_min, k_max))
            return k

        deg_seq = [sample_degree() for _ in range(self.N)]
        # adjust sum to be even
        if sum(deg_seq) % 2 == 1:
            deg_seq[0] += 1

        # Configuration model, then project to simple graph
        Gcm = nx.configuration_model(deg_seq, seed=self.random.randint(0, 10**9))
        Gcm = nx.Graph(Gcm)   # remove parallel edges
        Gcm.remove_edges_from(nx.selfloop_edges(Gcm))
        # keep only nodes 0..N-1
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.N))
        self.G.add_edges_from(Gcm.edges())

    def _assign_activity_by_degree(self):
        # Quantile split by degree: low (0-70%), mid (70-95%), high (95-100%)
        deg = dict(self.G.degree())
        degrees = np.array([deg[i] for i in range(self.N)])
        q70, q95 = np.quantile(degrees, [0.7, 0.95])
        for i, agent in enumerate(self.schedule.agents):
            d = deg[i]
            if d <= q70:
                agent.activity = 'low'
            elif d <= q95:
                agent.activity = 'mid'
            else:
                agent.activity = 'high'

    def _rewire_for_assortativity(self):
        """
        Increase within-activity edges by simple stub rewiring:
        with probability 'assort_within_activity' keep edges within the same activity.
        """
        edges = list(self.G.edges())
        self.G.remove_edges_from(edges)
        nodes_by_act = {
            'low': [a.unique_id for a in self.schedule.agents if a.activity == 'low'],
            'mid': [a.unique_id for a in self.schedule.agents if a.activity == 'mid'],
            'high': [a.unique_id for a in self.schedule.agents if a.activity == 'high'],
        }
        # add edges back with assortative preference
        rng = self.random
        for u, v in edges:
            au = self.schedule.agents[u].activity
            av = self.schedule.agents[v].activity
            if rng.random() < self.assort_within_activity:
                # same-activity attempt
                pool = nodes_by_act[au]
                if len(pool) > 1:
                    v2 = rng.choice(pool)
                    if u != v2 and not self.G.has_edge(u, v2):
                        self.G.add_edge(u, v2)
                        continue
            # fallback: random original
            if not self.G.has_edge(u, v):
                self.G.add_edge(u, v)

    # ---------- Transmission ----------
    def _transmission_phase(self):
        """
        For each edge, if one infectious and the other susceptible, infect with prob:
          beta_f if recipient is F, beta_m if recipient is M,
        scaled by recipient's susceptibility_multiplier (vaccination).
        New infections go to HR with prob hr_fraction, else LR.
        """
        new_infections: List[int] = []
        for u, v in self.G.edges():
            au: HPVAgent = self.schedule._agents[u]
            av: HPVAgent = self.schedule._agents[v]

            # u -> v
            if au.infectious() and av.state == 'S':
                beta = self.beta_f if av.sex == 'F' else self.beta_m
                p = beta * av.susceptibility_multiplier()
                if self.random.random() < p:
                    new_infections.append(v)

            # v -> u
            if av.infectious() and au.state == 'S':
                beta = self.beta_f if au.sex == 'F' else self.beta_m
                p = beta * au.susceptibility_multiplier()
                if self.random.random() < p:
                    new_infections.append(u)

        for idx in new_infections:
            ag: HPVAgent = self.schedule._agents[idx]
            ag.state = 'I_HR' if (self.random.random() < self.hr_fraction) else 'I_LR'
            ag.infection_years = 0

    # ---------- Step / Run ----------
    def step(self):
        self._transmission_phase()
        self.schedule.step()
        self.datacollector.collect(self)
        self.t += 1

    def run(self, years: int):
        # collect t=0
        self.datacollector.collect(self)
        for _ in range(years):
            self.step()
        return self.datacollector.get_model_vars_dataframe()