import numpy as np

from nasim.envs.state import State
from nasim.envs.action import Action
from nasim.envs.render import Viewer
from nasim.envs.network import Network
from nasim.scenarios import Scenario, INTERNET


class NASimEnv:
    """A simple simulated computer network with subnetworks and hosts with
    different vulnerabilities.

    Properties:
    - current_state : the current knowledge the agent has observed
    - action_space : the set of all actions allowed for environment
    """

    rendering_modes = ["readable", "ASCI"]

    action_space = None
    current_state = None

    def __init__(self, scenario):
        """
        Arguments
        ---------
        scenario : Scenario
            Scenario object, defining the properties of the environment
        """
        self.scenario = scenario

        self.network = Network(scenario)
        self.address_space = self.network.get_address_space()

        self.service_map = {}
        for i, service in enumerate(self.scenario.exploits.keys()):
            self.service_map[service] = i

        self.action_space = Action.load_action_space(self.network, self.scenario)

        self.init_state = self._generate_initial_state()
        self.compromised_subnets = None
        self.renderer = None
        self.reset()

    @classmethod
    def from_file(cls, path):
        """Construct Environment from a scenario file.

        Arguments
        ---------
        path : str
            path to the scenario file

        Returns
        -------
        NASimEnv
            a new environment object
        """
        scenario = Scenario.load_from_file(path)
        return cls(scenario)

    @classmethod
    def from_params(cls, num_hosts, num_services, **params):
        """Construct Environment from an auto generated network.

        Arguments
        ---------
        num_hosts : int
            number of hosts to include in network (minimum is 3)
        num_services : int
            number of services to use in environment (minimum is 1)
        params : dict
            generator params (see scenarios.generator for full list)

        Returns
        -------
        NASimEnv
            a new environment object
        """
        scenario = Scenario.generate(num_hosts, num_services, **params)
        return cls(scenario)

    def reset(self):
        """Reset the state of the environment and returns the initial state.

        Returns
        -------
        initial_state : State
            the initial state of the environment
        """
        self.current_state = self.init_state.copy()
        self.compromised_subnets = set([INTERNET])
        return self.current_state

    def step(self, action):
        """Run one step of the environment using action.

        N.B. Does not return a copy of the state, and state is changed by simulator. So if you
        need to store the state you will need to copy it (see State.copy method)

        Arguments
        ---------
        action : Action
            Action object from action_space

        Returns
        -------
        obs : State
            current state of environment known by agent
        reward : float
            reward from performing action
        done : bool
            whether the episode has ended or not
        """
        if not self.current_state.reachable(action.target):
            return self.current_state, 0 - action.cost, False
        if not self._action_traffic_permitted(action):
            return self.current_state, 0 - action.cost, False

        # non-deterministic actions
        if np.random.rand() > action.prob:
            return self.current_state, 0 - action.cost, False

        success, value, services = self.network.perform_action(action)
        value = 0 if self.current_state.compromised(action.target) else value
        self._update_state(action, success, services)
        done = self._is_goal()
        reward = value - action.cost
        obs = self.current_state
        return obs, reward, done

    def render(self, mode="ASCI"):
        """Render current state.

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
            self.renderer.render_asci(self.current_state)
        elif mode == "readable":
            self.renderer.render_readable(self.current_state)
        else:
            print("Please choose correct render mode: {0}".format(self.rendering_modes))

    def render_episode(self, episode, width=7, height=7):
        """Render an episode as sequence of network graphs, where an episode is a sequence of
        (state, action, reward, done) tuples generated from interactions with environment.

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

    def render_network_graph(self, initial_state=True, ax=None, show=False):
        """Render a plot of network as a graph with hosts as nodes arranged into subnets and
        showing connections between subnets

        Arguments
        ---------
        initial_state : bool
            whether to render current or initial state of network
        ax : Axes
            matplotlib axis to plot graph on, or None to plot on new axis
        show : bool
            whether to display plot, or simply setup plot and showing plot
            can be handled elsewhere by user
        """
        if self.renderer is None:
            self.renderer = Viewer(self.network)
        state = self.init_state if initial_state else self.current_state
        self.renderer.render_graph(state, ax, show)

    def get_state_size(self):
        """Get the size of an environment state representation in terms of the number of features,
        where a feature is a value for an individual host (i.e. compromised, reachable,
        service1, ...).

        Returns
        -------
        state_size : int
            size of state representation
        """
        return self.init_state.get_state_size()

    def get_num_actions(self):
        """Get the size of the action space for environment

        Returns
        -------
        num_actions : int
            action space size
        """
        return len(self.action_space)

    def get_minimum_actions(self):
        """Get the minimum possible actions required to exploit all sensitive hosts from the
        initial state

        Returns
        -------
        minimum_actions : int
            minumum possible actions
        """
        return self.network.get_minimal_steps()

    def get_best_possible_score(self):
        """Get the best score possible for this environment, assuming action cost of 1 and each
        sensitive host is exploitable from any other connected subnet.

        The theoretical best score is where the agent only exploits a single host in each subnet
        that is required to reach sensitive hosts along the shortest bath in network graph, and
        exploits the two sensitive hosts (i.e. the minial steps)

        Returns
        -------
        max_score : float
            theoretical max score
        """
        max_reward = self.network.get_total_sensitive_host_value()
        max_reward -= self.network.get_minimal_steps()
        return max_reward

    def _generate_initial_state(self):
        """Generate the initial state of the environment. Initial state is where no hosts have been
        compromised, only DMZ subnets are reachable and no information about services has been
        gained

        Returns
        -------
        initial_state : State
            the initial state of the environment
        """
        return State.generate_initial_state(self.network, self.service_map)

    def _action_traffic_permitted(self, action):
        """Checks whether an action is permitted in terms of firewall traffic and the target service,
        based on current set of compromised hosts on network.

        Arguments
        ---------
        action : Action
            the action performed

        Returns
        -------
        permitted : bool
            True if traffic is permitted for action, False otherwise
        """
        if not self.current_state.reachable(action.target):
            return False
        # We assume scannning uses alternative methods to work around firewall (e.g. UDP, ARP)
        if action.is_scan():
            return True
        service = action.service
        dest = action.target[0]
        for src in self.compromised_subnets:
            if self.network.traffic_permitted(src, dest, service):
                return True
        return False

    def _update_state(self, action, success, services):
        """Updates the current state of environment based on if action was successful and the gained
        service info

        Arguments
        ---------
        action : Action
            the action performed
        success : bool
            whether action was successful
        services : dict
            service info gained from action
        """
        target = action.target
        if action.is_scan() or (not action.is_scan() and success):
            # 1. scan or successful exploit, all service info gained for target
            for srv, present in services.items():
                self.current_state.update_service(target, srv, present)
            if not action.is_scan():
                # successful exploit so host compromised
                self.current_state.set_compromised(target)
                self.compromised_subnets.add(target[0])
                self._update_reachable(action.target)
        # 2. unsuccessful exploit, targeted service may or may not be present so do nothing

    def _update_reachable(self, compromised_m):
        """Updates the reachable status of hosts on network, based on current state and newly
        exploited host

        Arguments
        ---------
        compromised_m : (int, int)
            compromised host address
        """
        comp_subnet = compromised_m[0]
        for m in self.address_space:
            if self.current_state.reachable(m):
                continue
            m_subnet = m[0]
            if self.network.subnets_connected(comp_subnet, m_subnet):
                self.current_state.set_reachable(m)

    def _is_goal(self):
        """Check if the current state is the goal state.
        The goal state is  when all sensitive hosts have been compromised

        Returns
        -------
        goal : bool
            True if goal state, otherwise False
        """
        for sensitive_m in self.network.get_sensitive_hosts():
            if not self.current_state.compromised(sensitive_m):
                # at least one sensitive host not compromised
                return False
        return True

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
