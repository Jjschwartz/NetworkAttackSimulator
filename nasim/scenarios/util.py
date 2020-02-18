import os.path as osp

import nasim.scenarios.benchmark as bm
from nasim.envs.environment import NASim
import nasim.scenarios.benchmark.generated as bmgen


SCENARIO_DIR = osp.dirname(osp.abspath(__file__))

# scenario property keys
SUBNETS = "subnets"
TOPOLOGY = "topology"
SENSITIVE_HOSTS = "sensitive_hosts"
SERVICES = "services"
EXPLOITS = "exploits"
SCAN_COST = "scan_cost"
HOST_CONFIGS = "host_configurations"
FIREWALL = "firewall"
HOSTS = "host"

# scenario exploit keys
EXPLOIT_TARGET = "target"
EXPLOIT_PROB = "prob"
EXPLOIT_COST = "cost"


def make_benchmark_env(scenario_name, seed=1):
    if scenario_name in bm.AVAIL_GEN_BENCHMARKS:
        scenario = bm.AVAIL_GEN_BENCHMARKS[scenario_name]
        M = scenario["hosts"]
        S = scenario["services"]
        rve = scenario["restrictiveness"]
        env = NASim.from_params(M, S,
                                r_sensitive=bmgen.R_SENS,  r_user=bmgen.R_USR,
                                exploit_cost=bmgen.COST_EXP, scan_cost=bmgen.COST_SCAN,
                                restrictiveness=rve, exploit_probs=bmgen.EXPLOIT_PROB, seed=seed)
    elif scenario_name in bm.AVAIL_STATIC_BENCHMARKS:
        scenario_file = bm.AVAIL_STATIC_BENCHMARKS[scenario_name]["file"]
        env = NASim.from_file(scenario_file, seed=seed)
    else:
        raise NotImplementedError(f"Benchmark scenario '{scenario_name}' not available."
                                  f"Available scenarios are: {bm.AVAIL_BENCHMARKS}")
    return env


def get_scenario_max(scenario_name):
    if scenario_name in bm.AVAIL_GEN_BENCHMARKS:
        return bm.AVAIL_GEN_BENCHMARKS[scenario_name]["max_score"]
    elif scenario_name in bm.AVAIL_STATIC_BENCHMARKS:
        return bm.AVAIL_STATIC_BENCHMARKS[scenario_name]["max_score"]
    return None
