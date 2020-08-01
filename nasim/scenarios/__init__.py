from .utils import INTERNET
from .scenario import Scenario
from .loader import ScenarioLoader
from .generator import ScenarioGenerator
import nasim.scenarios.benchmark as benchmark


def make_benchmark_scenario(scenario_name, seed=None):
    """Generate or Load a benchmark Scenario.

    Parameters
    ----------
    scenario_name : str
        the name of the benchmark environment
    seed : int, optional
        random seed to use to generate environment (default=None)

    Returns
    -------
    Scenario
        a new scenario instance

    Raises
    ------
    NotImplementederror
        if scenario_name does no match any implemented benchmark scenarios.
    """
    if scenario_name in benchmark.AVAIL_GEN_BENCHMARKS:
        params = benchmark.AVAIL_GEN_BENCHMARKS[scenario_name]
        params['seed'] = seed
        return generate_scenario(**params)
    elif scenario_name in benchmark.AVAIL_STATIC_BENCHMARKS:
        scenario_def = benchmark.AVAIL_STATIC_BENCHMARKS[scenario_name]
        return load_scenario(scenario_def["file"], name=scenario_name)
    else:
        raise NotImplementedError(
            f"Benchmark scenario '{scenario_name}' not available."
            f"Available scenarios are: {benchmark.AVAIL_BENCHMARKS}"
        )


def generate_scenario(num_hosts, num_services, **params):
    """Generate Scenario from network parameters.

    Parameters
    ----------
    num_hosts : int
        number of hosts to include in network (minimum is 3)
    num_services : int
        number of services to use in environment (minimum is 1)
    params : dict, optional
        generator params (see :class:`ScenarioGenertor` for full list)

    Returns
    -------
    Scenario
        a new scenario object
    """
    generator = ScenarioGenerator()
    return generator.generate(num_hosts, num_services, **params)


def load_scenario(path, name=None):
    """Load NASim Environment from a .yaml scenario file.

    Parameters
    ----------
    path : str
        path to the .yaml scenario file
    name : str, optional
        the scenarios name, if None name will be generated from path
        (default=None)

    Returns
    -------
    Scenario
        a new scenario object
    """
    loader = ScenarioLoader()
    return loader.load(path, name=name)


def get_scenario_max(scenario_name):
    if scenario_name in benchmark.AVAIL_GEN_BENCHMARKS:
        return benchmark.AVAIL_GEN_BENCHMARKS[scenario_name]["max_score"]
    elif scenario_name in benchmark.AVAIL_STATIC_BENCHMARKS:
        return benchmark.AVAIL_STATIC_BENCHMARKS[scenario_name]["max_score"]
    return None
