import gymnasium as gym
from gymnasium.envs.registration import register

from nasim.envs import NASimEnv
from nasim.scenarios.benchmark import AVAIL_BENCHMARKS
from nasim.scenarios import \
    make_benchmark_scenario, load_scenario, generate_scenario


__all__ = ['make_benchmark', 'load', 'generate']


def make_benchmark(scenario_name,
                   seed=None,
                   fully_obs=False,
                   flat_actions=True,
                   flat_obs=True,
                   render_mode=None):
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
    render_mode : str, optional
            The render mode to use for the environment.

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
                  "flat_obs": flat_obs,
                  "render_mode": render_mode}
    scenario = make_benchmark_scenario(scenario_name, seed)
    return NASimEnv(scenario, **env_kwargs)


def load(path,
         fully_obs=False,
         flat_actions=True,
         flat_obs=True,
         name=None,
         render_mode=None):
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
    render_mode : str, optional
            The render mode to use for the environment.

    Returns
    -------
    NASimEnv
        a new environment object
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs,
                  "render_mode": render_mode}
    scenario = load_scenario(path, name=name)
    return NASimEnv(scenario, **env_kwargs)


def generate(num_hosts,
             num_services,
             fully_obs=False,
             flat_actions=True,
             flat_obs=True,
             render_mode=None,
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
    render_mode : str, optional
            The render mode to use for the environment.
    params : dict, optional
        generator params (see :class:`ScenarioGenertor` for full list)

    Returns
    -------
    NASimEnv
        a new environment object
    """
    env_kwargs = {"fully_obs": fully_obs,
                  "flat_actions": flat_actions,
                  "flat_obs": flat_obs,
                  "render_mode": render_mode}
    scenario = generate_scenario(num_hosts, num_services, **params)
    return NASimEnv(scenario, **env_kwargs)


def _register(id, entry_point, kwargs, nondeterministic, force=True):
    """Registers NASim as a Gymnasium Environment.

    Handles issues with re-registering gym environments.
    """
    if id in gym.envs.registry:
        if not force:
            return
        del gym.envs.registry[id]
    register(
        id=id,
        entry_point=entry_point,
        kwargs=kwargs,
        nondeterministic=nondeterministic
    )


for benchmark in AVAIL_BENCHMARKS:
    # PO - partially observable
    # 2D - use 2D Obs
    # VA - use param actions
    # tiny should yield Tiny and tiny-small should yield TinySmall
    for fully_obs in [True, False]:
        name = ''.join([g.capitalize() for g in benchmark.split("-")])
        if not fully_obs:
            name = f"{name}PO"

        _register(
            id=f"{name}-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": True,
                "flat_obs": True
            },
            nondeterministic=True
        )

        _register(
            id=f"{name}2D-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": True,
                "flat_obs": False
            },
            nondeterministic=True
        )

        _register(
            id=f"{name}VA-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": False,
                "flat_obs": True
            },
            nondeterministic=True
        )

        _register(
            id=f"{name}2DVA-v0",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": False,
                "flat_obs": False
            },
            nondeterministic=True
        )

__version__ = "0.12.0"
