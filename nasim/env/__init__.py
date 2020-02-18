from .environment import NASimEnv
import nasim.scenarios.benchmark as bm
import nasim.scenarios.benchmark.generated as bmgen


def make_benchmark_env(scenario_name, seed=None):
    if scenario_name in bm.AVAIL_GEN_BENCHMARKS:
        scenario = bm.AVAIL_GEN_BENCHMARKS[scenario_name]
        M = scenario["hosts"]
        S = scenario["services"]
        rve = scenario["restrictiveness"]
        params = dict(r_sensitive=bmgen.R_SENS,
                      r_user=bmgen.R_USR,
                      exploit_cost=bmgen.COST_EXP,
                      scan_cost=bmgen.COST_SCAN,
                      restrictiveness=rve,
                      exploit_probs=bmgen.EXPLOIT_PROB,
                      seed=seed)
        env = NASimEnv.from_params(M, S, **params)
    elif scenario_name in bm.AVAIL_STATIC_BENCHMARKS:
        scenario_file = bm.AVAIL_STATIC_BENCHMARKS[scenario_name]["file"]
        env = NASimEnv.from_file(scenario_file)
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
