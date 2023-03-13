from nasim.envs.environment import NASimEnv
from nasim.scenarios import Scenario, make_benchmark_scenario


class NASimGymEnv(NASimEnv):
    """A wrapper around the NASimEnv compatible with gymnasium.make()

    See nasim.NASimEnv for details.
    """

    def __init__(self,
                 scenario,
                 fully_obs=False,
                 flat_actions=True,
                 flat_obs=True,
                 render_mode=None):
        """
        Parameters
        ----------
        scenario : str or or nasim.scenarios.Scenario
            either the name of benchmark environment (str) or a nasim Scenario
            instance
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
        """
        if not isinstance(scenario, Scenario):
            scenario = make_benchmark_scenario(scenario)
        super().__init__(scenario,
                         fully_obs=fully_obs,
                         flat_actions=flat_actions,
                         flat_obs=flat_obs,
                         render_mode=render_mode)
