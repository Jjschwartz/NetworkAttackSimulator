import numpy as np
from cyber_attack_simulator.envs.network import Network
from cyber_attack_simulator.envs.action import Action
from cyber_attack_simulator.envs.state import State
from cyber_attack_simulator.envs.render import Viewer
import cyber_attack_simulator.envs.loader as loader


# Default reward when generating a network from paramaters
R_SENSITIVE = 1000.0
R_USER = 1000.0
# Default action costs
EXPLOIT_COST = 100.0
SCAN_COST = 100.0


class CyberAttackSimulatorEnv(object):
    """
    A simple simulated computer network with subnetworks and machines with
    different vulnerabilities.

    Properties:
    - current_state : the current knowledge the agent has observed
    - action_space : the set of all actions allowed for environment
    """

    rendering_modes = ["readable", "ASCI"]

    action_space = None
    current_state = None

    def __init__(self, config, exploit_cost=EXPLOIT_COST, scan_cost=SCAN_COST, exploit_probs=1.0):
        """
        Construct a new environment and network

        For deterministic exploits set exploit_probs=1.0
        For randomly generated probabilities set exploit_probs=None

        Arguments:
            dict config : network configuration
            float exploit_cost : cost of performing an exploit action
            float scan_cost : cost of performing a scan action
            None, int or list  exploit_probs :  success probability of exploits
        """
        self.config = config
        self.exploit_probs = exploit_probs
        self.seed = 1

        self.num_services = config["services"]
        self.network = Network(config)
        self.address_space = self.network.get_address_space()
        self.action_space = Action.generate_action_space(self.address_space, config["services"],
                                                         exploit_cost, scan_cost, exploit_probs)
        self.renderer = Viewer(self.network)
        self.init_state = self._generate_initial_state()
        self.compromised_subnets = None
        self.reset()

    @classmethod
    def from_file(cls, path, exploit_cost=EXPLOIT_COST, scan_cost=SCAN_COST, exploit_probs=1.0):
        """
        Construct a new Cyber Attack Simulator Environment from a config file.

        Arguments:
            str path : path to the config file
            float exploit_cost : cost of performing an exploit action
            float scan_cost : cost of performing a scan action
            None, int or list  exploit_probs :  success probability of exploits

        Returns:
            CyberAttackSimulatorEnv env : a new environment object
        """
        config = loader.load_config(path)
        return cls(config, exploit_cost, scan_cost, exploit_probs)

    @classmethod
    def from_params(cls, num_machines, num_services,
                    r_sensitive=R_SENSITIVE, r_user=R_USER,
                    exploit_cost=EXPLOIT_COST, scan_cost=SCAN_COST,
                    exploit_probs=1.0,
                    uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0,
                    restrictiveness=5,
                    seed=1):
        """
        Construct a new Cyber Attack Simulator Environment from a auto generated network based on
        number of machines and services.

        Arguments:
            int num_machines : number of machines to include in network (minimum is 3)
            int num_services : number of services to use in environment (minimum is 1)
            float r_sensitive : reward for sensitive subnet documents
            float r_user : reward for user subnet documents
            float exploit_cost : cost of performing an exploit action
            float scan_cost : cost of performing a scan action
            None, int or list  exploit_probs :  success probability of exploits
            bool uniform : whether to use uniform distribution of machine configs or corelated
                           machine configs
            float alpha_H : (only used when uniform=False), scaling/concentration parameter for
                            controlling corelation between machine configurations (must be > 0)
            float alpha_V : (only used when uniform=False) scaling/concentration parameter for
                            controlling corelation between services across machine configurations
                            (must be > 0)
            float lambda_V : (only used when uniform=False) parameter for controlling average
                             number of services running per machine configuration (must be > 0)
            int restrictiveness : max number of services allowed to pass through firewalls between
                                  zones
            int seed : random number generator seed

        Returns:
            CyberAttackSimulatorEnv env : a new environment object
        """
        config = loader.generate_config(num_machines, num_services,
                                        r_sensitive, r_user,
                                        uniform, alpha_H, alpha_V, lambda_V,
                                        restrictiveness, seed)
        return cls(config, exploit_cost, scan_cost, exploit_probs)

    def reset(self):
        """
        Reset the state of the environment and returns the initial state.

        Returns:
            State initial_state : the initial state of the environment
        """
        self.current_state = self.init_state.copy()
        self.compromised_subnets = set([loader.INTERNET])
        return self.current_state.copy()

    def step(self, action):
        """
        Run one step of the environment using action.

        Arguments:
            Action action : Action object from action_space

        Returns:
            State obs : current state of environment known by agent
            float reward : reward from performing action
            bool done : whether the episode has ended or not
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
        # update current state in place, then return copy since State object is muteable
        obs = self.current_state.copy()
        return obs, reward, done

    def render(self, mode="ASCI"):
        """
        Render current state.

        See render module for more details on modes and symbols.

        If mode = ASCI:
            Machines displayed in rows, with one row for each subnet and
            machines displayed in order of id within subnet

        Arguments:
            str mode : rendering mode
        """
        if mode == "ASCI":
            self.renderer.render_asci(self.current_state)
        elif mode == "readable":
            self.renderer.render_readable(self.current_state)
        else:
            print("Please choose correct render mode: {0}".format(self.rendering_modes))

    def render_episode(self, episode, width=7, height=7):
        """
        Render an episode as sequence of network graphs, where an episode is a sequence of
        (state, action, reward, done) tuples generated from interactions with environment.

        Arguments:
            list episode : list of (State, Action, reward, done) tuples
            int width : width of GUI window
            int height : height of GUI window
        """
        self.renderer.render_episode(episode)

    def render_network_graph(self, initial_state=True, axes=None, show=False):
        """
        Render a plot of network as a graph with machines as nodes arranged into subnets and
        showing connections between subnets

        Arguments:
            bool initial_state : whether to render current or initial state of network
            Axes axes : matplotlib axes to plot graph on, or None to plot on new axes
            bool show : whether to display plot, or simply setup plot and showing plot can be
                        handled elsewhere by user
        """
        state = self.init_state if initial_state else self.current_state
        self.renderer.render_graph(state, axes, show)

    def get_state_size(self):
        return self.init_state.get_state_size()

    def get_num_actions(self):
        return len(self.action_space)

    def _generate_initial_state(self):
        """
        Generate the initial state of the environment. Initial state is where no machines have been
        compromised, only DMZ subnets are reachable and no information about services has been
        gained

        Returns:
            State initial_state : the initial state of the environment
        """
        return State.generate_initial_state(self.network, self.num_services)

    def _action_traffic_permitted(self, action):
        """
        Checks whether an action is permitted in terms of firewall traffic and the target service,
        based on current set of compromised machines on network.

        Arguments:
            Action action : the action performed

        Returns:
            bool permitted : True if traffic is permitted for action, False otherwise
        """
        if not self.current_state.reachable(action.target):
            return False
        # TODO: scan should only return info on allowed service traffic
        if action.is_scan():
            return True
        service = action.service
        dest = action.target[0]
        for src in self.compromised_subnets:
            if self.network.traffic_permitted(src, dest, service):
                return True
        return False

    def _update_state(self, action, success, services):
        """
        Updates the current state of environment based on if action was successful and the gained
        service info

        Arguments:
            Action action : the action performed
            bool success : whether action was successful
            list services : service info gained from action
        """
        target = action.target
        if action.is_scan() or (not action.is_scan() and success):
            # 1. scan or successful exploit, all service info gained for target
            for s in range(len(services)):
                self.current_state.update_service(target, s, services[s])
            if not action.is_scan():
                # successful exploit so machine compromised
                self.current_state.set_compromised(target)
                self.compromised_subnets.add(target[0])
                self._update_reachable(action.target)
        # 2. unsuccessful exploit, targeted service may or may not be present so do nothing

    def _update_reachable(self, compromised_m):
        """
        Updates the reachable status of machines on network, based on current state and newly
        exploited machine

        Arguments:
            (int, int) compromised_m : compromised machine address
        """
        comp_subnet = compromised_m[0]
        for m in self.address_space:
            if self.current_state.reachable(m):
                continue
            m_subnet = m[0]
            if self.network.subnets_connected(comp_subnet, m_subnet):
                self.current_state.set_reachable(m)

    def _is_goal(self):
        """
        Check if the current state is the goal state.
        The goal state is  when all sensitive machines have been compromised

        Returns:
            bool goal : True if goal state, otherwise False
        """
        for sensitive_m in self.network.get_sensitive_machines():
            if not self.current_state.compromised(sensitive_m):
                # at least one sensitive machine not compromised
                return False
        return True

    def __str__(self):
        output = "Environment: "
        output += "Subnets = {}, ".format(self.network.subnets)
        output += "Services = {}, ".format(self.num_services)
        output += "Exploit Probs = {}".format(self.exploit_probs)
        return output

    def outfile_name(self):
        """
        Generate name for environment for use when writing to a file.

        Output format:
            <list of size of each subnet>_<number of services>_<det or stoch>
        """
        output = "{}_".format(self.network.subnets)
        output += "{}_".format(self.num_services)
        if self.exploit_probs is None or type(self.exploit_probs) is list:
            deterministic = "stoch"
        else:
            deterministic = "det" if self.exploit_probs == 1.0 else False
        output += "{}".format(deterministic)
        return output
