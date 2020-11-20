import gym
from gym.envs.registration import register

from nasim.envs import NASimEnv
from nasim.scenarios.benchmark import AVAIL_BENCHMARKS
from nasim.scenarios import \
    make_benchmark_scenario, load_scenario, generate_scenario


__all__ = ['make_benchmark', 'load', 'generate']


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


# Register NASimEnv with OpenAI gym
def _register(id, entry_point, kwargs, nondeterministic, force=True):
    """Registers NASim Open AI Gym Environment.

    Handles issues with re-registering gym environments.
    """
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    register(
        id=id,
        entry_point=entry_point,
        kwargs=kwargs,
        nondeterministic=nondeterministic
    )


for benchmark in AVAIL_BENCHMARKS:
    # v0 - flat_actions, flat_obs
    # v1 - flat_actions, 2D_obs
    # v2 - param_actions, flat obs
    # v3 - param_actions, 2D obs
    # tiny should yield Tiny and tiny-small should yield TinySmall
    for fully_obs in [True, False]:
        name = ''.join([g.capitalize() for g in benchmark.split("-")])
        if not fully_obs:
            name = f"{name}-PO"

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
            id=f"{name}-v1",
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
            id=f"{name}-v2",
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
            id=f"{name}-v3",
            entry_point='nasim.envs:NASimGymEnv',
            kwargs={
                "scenario": benchmark,
                "fully_obs": fully_obs,
                "flat_actions": False,
                "flat_obs": False
            },
            nondeterministic=True
        )
