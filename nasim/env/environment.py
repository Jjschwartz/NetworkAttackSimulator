import numpy as np

from nasim.env.state import State
from nasim.env.action import Action
from nasim.env.render import Viewer
from nasim.env.network import Network
from nasim.scenarios import ScenarioLoader, ScenarioGenerator


class NASimEnv:
    """A simple simulated computer network with subnetworks and hosts with
    different vulnerabilities.

    Properties
    ----------
    - current_state : the current knowledge the agent has observed
    - action_space : the set of all actions allowed for environment
    - mode : the observability mode of the environment.

    The mode can be either:
    1. MDP - Here the state is fully observable, so after each step the actual
             next state is returned
    2. POMDP - The state is partially observable, so after each step only what
               is observed of the next state is returned.

    For both modes the dimensions are the same for the returned
    state/observation. The only difference is for the POMDP mode, for parts of
    the state that were not observed the value returned will be a non-obs
    value (i.e. 0 in most cases).
    """
    rendering_modes = ["readable", "ASCI"]
    env_modes = ['MDP', 'POMDP']

    action_space = None
    current_state = None

    def __init__(self, scenario, partially_obs=False):
        """
        Arguments
        ---------
        scenario : Scenario
            Scenario object, defining the properties of the environment
        partially_obs : bool
            The observability mode of environment, if True then uses partially
            observable mode, otherwise is Fully observable (default=False)
        """
        self.scenario = scenario
        self.fully_obs = not partially_obs

        self.network = Network(scenario)
        self.address_space = scenario.address_space
        self.action_space = Action.load_action_space(self.scenario)

        self.current_state = State.generate_initial_state(self.network)
        self.last_obs = None
        self.renderer = None
        self.reset()

    @classmethod
    def from_file(cls, path, partially_obs):
        """Construct Environment from a scenario file.

        Arguments
        ---------
        path : str
            path to the scenario file
        partially_obs : bool
            The observability mode of environment, if True then uses partially
            observable mode, otherwise is Fully observable

        Returns
        -------
        NASimEnv
            a new environment object
        """
        loader = ScenarioLoader()
        scenario = loader.load(path)
        return cls(scenario, partially_obs)

    @classmethod
    def from_params(cls, num_hosts, num_services, partially_obs, **params):
        """Construct Environment from an auto generated network.

        Arguments
        ---------
        num_hosts : int
            number of hosts to include in network (minimum is 3)
        num_services : int
            number of services to use in environment (minimum is 1)
        partially_obs : bool
            The observability mode of environment, if True then uses partially
            observable mode, otherwise is Fully observable
        params : dict
            generator params (see scenarios.generator for full list)

        Returns
        -------
        NASimEnv
            a new environment object
        """
        generator = ScenarioGenerator()
        scenario = generator.generate(num_hosts, num_services, **params)
        return cls(scenario, partially_obs)

    def reset(self):
        """Reset the state of the environment and returns the initial state.

        Returns
        -------
        Obs
            the initial observation of the environment
        """
        self.current_state = self.network.reset(self.current_state)
        self.last_obs = self.current_state.get_initial_observation(
            self.fully_obs
        )
        return self.last_obs

    def step(self, action):
        """Run one step of the environment using action.

        N.B. Does not return a copy of the observation, and observation
        is changed by simulator. So if you need to store the observation
        you will need to copy it.

        Arguments
        ---------
        action : Action or int
            Action object from action space or index of action in action space

        Returns
        -------
        obs : Observation
            current observation of environment
        reward : float
            reward from performing action
        done : bool
            whether the episode has ended or not
        info : dict
            other information regarding step (see ActionObservation.info())
        """
        next_state, obs, reward, done, info = self.generative_step(
            self.current_state,
            action
        )
        self.current_state = next_state
        self.last_obs = obs
        return obs, reward, done, info

    def generative_step(self, state, action):
        """Run one step of the environment using action in given state.

        Arguments
        ---------
        state : State
            The state to perform the action in
        action : Action or int
            Action object from action space or index of action in action space

        Returns
        -------
        next_state : State
            the state after action was performed
        obs : Observation
            current observation of environment
        reward : float
            reward from performing action
        done : bool
            whether the episode has ended or not
        info : dict
            other information regarding step (see ActionObservation.info())
        """
        assert isinstance(action, (Action, int)), \
            "Step action must be an integer or an Action object"
        if isinstance(action, int):
            action = self.action_space[action]

        next_state, action_obs = self.network.perform_action(
            state,
            action
        )
        obs = next_state.get_observation(action,
                                         action_obs,
                                         self.fully_obs)
        done = self.is_goal(next_state)
        reward = action_obs.value - action.cost
        return next_state, obs, reward, done, action_obs.info()

    def render(self, mode="ASCI"):
        """Render last observation.

        See render module for more details on modes and symbols.

        If mode = ASCI:
            Machines displayed in rows, with one row for each subnet and
            hosts displayed in order of id within subnet

        Arguments
        ---------
        mode : str
            rendering mode
        """
        if self.renderer is None:
            self.renderer = Viewer(self.network)
        if mode == "ASCI":
            self.renderer.render_asci(self.last_obs)
        elif mode == "readable":
            self.renderer.render_readable(self.last_obs)
        else:
            print("Please choose correct render mode from :"
                  f"{self.rendering_modes}")

    def render_action(self, action):
        if isinstance(action, int):
            action = self.action_space[action]
        print(action)

    def render_episode(self, episode, width=7, height=7):
        """Render an episode as sequence of network graphs, where an episode
        is a sequence of (state, action, reward, done) tuples generated from
        interactions with environment.

        Arguments
        ---------
        episode : list
            list of (State, Action, reward, done) tuples
        width : int
            width of GUI window
        height : int
            height of GUI window
        """
        if self.renderer is None:
            self.renderer = Viewer(self.network)
        self.renderer.render_episode(episode)

    def render_network_graph(self, ax=None, show=False):
        """Render a plot of network as a graph with hosts as nodes arranged
        into subnets and showing connections between subnets. Renders current
        state of network.

        Arguments
        ---------
        ax : Axes
            matplotlib axis to plot graph on, or None to plot on new axis
        show : bool
            whether to display plot, or simply setup plot and showing plot
            can be handled elsewhere by user
        """
        if self.renderer is None:
            self.renderer = Viewer(self.network)
        state = self.current_state
        self.renderer.render_graph(state, ax, show)

    def get_state_shape(self, flat=True):
        """Get the shape of an environment state representation

        Arguments
        ---------
        flat : bool, optional
            whether to get shape of flattened state (True) or not (False)
            (default=True)

        Returns
        -------
        (int, int)
            shape of state representation
        """
        if flat:
            return self.current_state.flat_shape()
        return self.current_state.shape()

    def get_obs_shape(self, flat=True):
        """Get the shape of an environment observation representation

        Arguments
        ---------
        flat : bool, optional
            whether to get shape of flattened observation (True) or not (False)
            (default=True)

        Returns
        -------
        (int, int)
            shape of observation representation
        """
        # observation has same shape as state
        return self.get_state_shape(flat)

    def get_num_actions(self):
        """Get the size of the action space for environment

        Returns
        -------
        num_actions : int
            action space size
        """
        return len(self.action_space)

    def get_minimum_actions(self):
        """Get the minimum possible actions required to exploit all sensitive
        hosts from the initial state

        Returns
        -------
        minimum_actions : int
            minumum possible actions
        """
        return self.network.get_minimal_steps()

    def get_action_mask(self):
        """Get a vector mask for valid actions.

        Returns
        -------
        ndarray
            numpy vector of 1's and 0's, one for each action. Where an
            index will be 1 if action is valid given current state, or
            0 if action is invalid.
        """
        mask = np.zeros(len(self.action_space), dtype=np.float)
        for i, action in enumerate(self.action_space):
            if self.network.host_discovered(action.target):
                mask[i] = 1
        return mask

    def get_best_possible_score(self):
        """Get the best score possible for this environment, assuming action
        cost of 1 and each sensitive host is exploitable from any other
        connected subnet.

        The theoretical best score is where the agent only exploits a single
        host in each subnet that is required to reach sensitive hosts along
        the shortest bath in network graph, and exploits the two sensitive
        hosts (i.e. the minial steps)

        Returns
        -------
        max_score : float
            theoretical max score
        """
        max_reward = self.network.get_total_sensitive_host_value()
        max_reward -= self.network.get_minimal_steps()
        return max_reward

    def is_goal(self, state):
        """Check if the current state is the goal state.
        The goal state is  when all sensitive hosts have been compromised
        """
        return self.network.all_sensitive_hosts_compromised(state)

    def __str__(self):
        output = "Environment: "
        output += "Subnets = {}, ".format(self.network.subnets)
        output += "Services = {}, ".format(self.scenario.num_services)
        return output

    def outfile_name(self):
        """Generate name for environment for use when writing to a file.

        Output format:
            <list of size of each subnet>_<number of services>
        """
        output = "{}_".format(self.network.subnets)
        output += "{}_".format(self.scenario.num_services)
        return output
