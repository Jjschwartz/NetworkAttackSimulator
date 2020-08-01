from nasim.env import NASimEnv
from nasim.scenarios import \
    make_benchmark_scenario, load_scenario, generate_scenario


def make_benchmark(scenario_name,
                   seed=None,
                   fully_obs=False,
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
        observable mode, otherwise partially observable (default=False)
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
    scenario = make_benchmark_scenario(scenario_name, seed)
    return NASimEnv(scenario, **env_kwargs)


def load(path,
         fully_obs=False,
         flat_actions=True,
         flat_obs=True,
         name=None):
    """Load NASim Environment from a .yaml scenario file.

    Parameters
    ----------
    path : str
        path to the .yaml scenario file
    fully_obs : bool, optional
        The observability mode of environment, if True then uses fully
        observable mode, otherwise partially observable (default=False)
    flat_actions : bool, optional
        if true then uses a flat action space, otherwise will use
        parameterised action space (default=True).
    flat_obs : bool, optional
        if true then uses a 1D observation space. If False
        will use a 2D observation space (default=True)
    name : str, optional
        the scenarios name, if None name will be generated from path
        (default=None)

    Returns
    -------
    NASimEnv
        a new environment object
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs}
    scenario = load_scenario(path, name=name)
    return NASimEnv(scenario, **env_kwargs)


def generate(num_hosts,
             num_services,
             fully_obs=False,
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
        observable mode, otherwise partially observable (default=False)
    flat_actions : bool, optional
        if true then uses a flat action space, otherwise will use
        parameterised action space (default=True).
    flat_obs : bool, optional
        if true then uses a 1D observation space. If False
        will use a 2D observation space (default=True)
    params : dict, optional
        generator params (see :class:`ScenarioGenertor` for full list)

    Returns
    -------
    NASimEnv
        a new environment object
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs}
    scenario = generate_scenario(num_hosts, num_services, **params)
    return NASimEnv(scenario, **env_kwargs)
