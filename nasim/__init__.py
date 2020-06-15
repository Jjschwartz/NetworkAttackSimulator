from nasim.env import NASimEnv
import nasim.scenarios.benchmark as bm
from nasim.scenarios import ScenarioLoader, ScenarioGenerator


def make_benchmark(scenario_name,
                   seed=None,
                   fully_obs=True,
                   flat_actions=True,
                   flat_obs=True):
    """Make a new benchmark NASim environment.

    Parameters
    ----------
    scenario_name : str
        the name of the benchmark environment
    seed : int, optional
        random seed to use to generate environment (default=None)
    fully_obs : bool, optional
        the observability mode of environment, if True then uses fully
        observable mode, otherwise partially observable (default=True)
    flat_actions : bool, optional
        if true then uses a flat action space, otherwise will use
        parameterised action space (default=True).
    flat_obs : bool, optional
        if true then uses a 1D observation space. If False
        will use a 2D observation space (default=True)

    Returns
    -------
    NASimEnv
        a new environment instance

    Raises
    ------
    NotImplementederror
        if scenario_name does no match any implemented benchmark scenarios.
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs}
    if scenario_name in bm.AVAIL_GEN_BENCHMARKS:
        scenario = bm.AVAIL_GEN_BENCHMARKS[scenario_name]
        scenario['seed'] = seed
        env = generate(**env_kwargs, **scenario)
    elif scenario_name in bm.AVAIL_STATIC_BENCHMARKS:
        scenario_file = bm.AVAIL_STATIC_BENCHMARKS[scenario_name]["file"]
        env = load(scenario_file, **env_kwargs)
    else:
        raise NotImplementedError(
            f"Benchmark scenario '{scenario_name}' not available."
            f"Available scenarios are: {bm.AVAIL_BENCHMARKS}"
        )
    return env


def get_scenario_max(scenario_name):
    if scenario_name in bm.AVAIL_GEN_BENCHMARKS:
        return bm.AVAIL_GEN_BENCHMARKS[scenario_name]["max_score"]
    elif scenario_name in bm.AVAIL_STATIC_BENCHMARKS:
        return bm.AVAIL_STATIC_BENCHMARKS[scenario_name]["max_score"]
    return None


def load(path,
         fully_obs=True,
         flat_actions=True,
         flat_obs=True):
    """Load NASim Environment from a .yaml scenario file.

    Parameters
    ----------
    path : str
        path to the .yaml scenario file
    fully_obs : bool, optional
        The observability mode of environment, if True then uses fully
        observable mode, otherwise partially observable (default=True)
    flat_actions : bool, optional
        if true then uses a flat action space, otherwise will use
        parameterised action space (default=True).
    flat_obs : bool, optional
        if true then uses a 1D observation space. If False
        will use a 2D observation space (default=True)

    Returns
    -------
    NASimEnv
        a new environment object
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs}
    loader = ScenarioLoader()
    scenario = loader.load(path)
    return NASimEnv(scenario, **env_kwargs)


def generate(num_hosts,
             num_services,
             fully_obs=True,
             flat_actions=True,
             flat_obs=True,
             **params):
    """Construct Environment from an auto generated network.

    Parameters
    ----------
    num_hosts : int
        number of hosts to include in network (minimum is 3)
    num_services : int
        number of services to use in environment (minimum is 1)
    fully_obs : bool, optional
        The observability mode of environment, if True then uses fully
        observable mode, otherwise partially observable (default=True)
    flat_actions : bool, optional
        if true then uses a flat action space, otherwise will use
        parameterised action space (default=True).
    flat_obs : bool, optional
        if true then uses a 1D observation space. If False
        will use a 2D observation space (default=True)
    params : dict, optional
        generator params (see scenarios.generator.ScenarioGenertor
        for full list)

    Returns
    -------
    NASimEnv
        a new environment object
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs}
    generator = ScenarioGenerator()
    scenario = generator.generate(num_hosts, num_services, **params)
    return NASimEnv(scenario, **env_kwargs)
