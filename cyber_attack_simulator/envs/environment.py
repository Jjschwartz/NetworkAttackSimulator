from enum import Enum
from collections import OrderedDict
import copy
import sys
import numpy as np
from cyber_attack_simulator.envs.network import Network
from cyber_attack_simulator.envs.action import Action
from cyber_attack_simulator.envs.state import State
from cyber_attack_simulator.envs.render import Viewer


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

    def __init__(self, config, exploit_probs=1.0, static=True):
        """
        Construct a new environment and network

        For deterministic exploits set exploit_probs=1.0
        For randomly generated probabilities set exploit_probs=None

        Arguments:
            dict config : network configuration
            None, int or list  exploit_probs :  success probability of exploits
            bool static : whether the network changes each episode or not
        """
        self.config = config
        self.exploit_probs = exploit_probs
        self.static = static
        self.seed = 1
        self.reset_count = 0

        self.num_services = config["services"]
        self.network = Network(config, self.seed)
        self.address_space = self.network.get_address_space()
        self.action_space = Action.generate_action_space(
            self.address_space, config["services"], exploit_probs)
        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns the initial observation.

        Returns:
            dict obs : the intial observation of the network environment
        """
        if not self.static:
            # generate new network using different seed
            self.network.generate_network(self.seed + self.reset_count)

        obs = OrderedDict()
        sensitive_machines = self.network.get_sensitive_machines()
        for m in self.address_space:
            # initially the status of services on machine are unknown
            service_info = np.full(self.num_services, Service.unknown,
                                   Service)
            compromised = False
            sensitive = False
            reachable = False
            if self.network.subnet_exposed(m[0]):
                reachable = True
            if m in sensitive_machines:
                sensitive = True
            obs[m] = {"service_info": service_info, "compromised": compromised,
                      "sensitive": sensitive, "reachable": reachable}
        self.current_state = State(obs)
        self.reset_count += 1
        return copy.deepcopy(self.current_state)

    def step(self, action):
        """
        Run one step of the environment using action.

        Arguments:
            Action action : Action object from action_space

        Returns:
            dict observation : agent's observation of the network
            float reward : reward from performing action
            bool done : whether the episode has ended or not
            dict info : contains extra info useful for debugging or
                visualization
        """
        if not self.current_state.reachable(action.target):
            return self.current_state, 0 - action.cost, False, {}

        # Non-determinism
        if np.random.random_sample() > action.prob:
            return self.current_state, 0 - action.cost, False, {}

        success, value, services = self.network.perform_action(action)

        value = 0 if self.current_state.compromised(action.target) else value
        self._update_state(action, success, services)
        done = self.is_goal()
        reward = value - action.cost
        return copy.deepcopy(self.current_state), reward, done, {}

    def _update_state(self, action, success, services):
        """
        Updates the current state of network state based on if action was
        successful and the gained service info

        Arguments:
            Action action : the action performed
            bool success : whether action was successful
            list services : service info gained from action
        """
        target = action.target
        if action.is_scan() or (not action.is_scan() and success):
            # 1. scan or successful exploit, all service info gained for target
            for s in range(len(services)):
                new_state = Service.present if services[s] else Service.absent
                self.current_state.update_service(target, s, new_state)
            if not action.is_scan():
                # successful exploit so machine compromised
                self.current_state.set_compromised(target)
                self._update_reachable(action.target)
        else:
            # 2. unsuccessful exploit, targeted service is absent
            self.current_state.update_service(target, action.service,
                                              Service.absent)

    def _update_reachable(self, compromised_m):
        """
        Updates the reachable status of machines on network, based on current
        state and newly exploited machine

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

    def is_goal(self):
        """
        Check if the current state is the goal state.
        The goal state is  when all sensitive documents have been collected
        (i.e. all rewarded machines have been compromised)

        Returns:
            bool goal : True if goal state, otherwise False
        """
        for sensitive_m in self.network.get_sensitive_machines():
            if not self.current_state.compromised(sensitive_m):
                # at least one sensitive machine not compromised
                return False
        return True

    def render(self, mode="ASCI"):
        """
        Render current state.

        If mode = ASCI:
            Machines displayed in rows, with one row for each subnet and
            machines displayed in order of id within subnet

            Key, for each machine:
                C   sensitive & compromised
                R   sensitive & reachable
                S   sensitive
                c   compromised
                r   reachable
                o   non-of-above

        Arguments:
            str mode : rendering mode
        """
        if mode == "ASCI":
            self._render_asci()
        elif mode == "readable":
            self._render_readable()
        else:
            print("Please choose correct render mode: {0}".format(
                self.rendering_modes))

    def _render_asci(self):
        outfile = sys.stdout

        subnets = [[], [], []]
        for m in self.address_space:
            subnets[m[0]].append(self._get_machine_symbol(m))

        max_row_size = max([len(x) for x in subnets])
        min_row_size = min([len(x) for x in subnets])

        output = "-----------------------------"
        for i, row in enumerate(subnets):
            output += "\nsubnet {0}: ".format(i)
            output += " " * ((max_row_size - len(row)) // 2)
            for col in row:
                output += col
            output += "\n"
            if i < len(subnets) - 1:
                n_spaces = (max_row_size - min_row_size) // 2
                output += " " * (len("subnet X: ") + n_spaces) + "|"
        output += "-----------------------------\n\n"

        outfile.write(output)

    def _get_machine_symbol(self, m):
        if self.current_state.sensitive(m):
            if self.current_state.compromised(m):
                symbol = "C"
            elif self.current_state.reachable(m):
                symbol = "R"
            else:
                symbol = "S"
        elif self.current_state.compromised(m):
            symbol = "c"
        elif self.current_state.reachable(m):
            symbol = "r"
        else:
            symbol = "o"
        return symbol

    def _render_readable(self):
        output = ""
        for m in self.address_space:
            output += "Machine = " + str(m) + " =>\n"

            output += "\tServices:\n"
            for s in range(self.num_services):
                service_state = self.current_state.service_state(m, s)
                output += "\t\t{0} = {1}".format(s, str(service_state))
                output += "\n"

            output += "\treachable: {0}\n".format(
                self.current_state.reachable(m))
            output += "\tcompromised: {0}\n".format(
                self.current_state.compromised(m))
            output += "\tsensitive: {0}\n".format(
                self.current_state.sensitive(m))
        sys.stdout.write(output)

    def optimal_num_actions(self):
        """
        Return optimal number of actions to reach goal, assuming deterministic
        actions.

        Also, assumes that for environments where the number of services is 2
        or greater the optimal actions are scan -> exploit

        Returns:
            int n: optimal number of actions to reach goal
        """
        num_machines = len(self.address_space)
        num_subnets = np.ceil((num_machines - 2) / 5)
        user_depth = np.floor(np.log2(num_subnets)) + 1
        if self.num_services > 1:
            return 2 + 2 + 2 * user_depth
        # only 1 service so only requires 1 action per subnet
        return 1 + 1 + user_depth

    def render_episode(self, episode):
        """
        Render an episode as sequence of network graphs

        Arguments:
            list episode : ordered list of (State, Action, reward) tuples from
                each timestep taken during episode
        """
        Viewer(episode, self.network)

    def __str__(self):
        output = "Environment: "
        output += "Subnets = {}, ".format(self.network.subnets)
        output += "Services = {}, ".format(self.num_services)
        if self.exploit_probs is None or type(self.exploit_probs) is list:
            deterministic = False
        else:
            deterministic = True if self.exploit_probs == 1.0 else False
        output += "Deterministic = {}, ".format(deterministic)
        output += "Static = {}".format(self.static)
        return output

    def outfile_name(self):
        output = "{}_".format(self.network.subnets)
        output += "{}_".format(self.num_services)
        if self.exploit_probs is None or type(self.exploit_probs) is list:
            deterministic = "stoch"
        else:
            deterministic = "det" if self.exploit_probs == 1.0 else False
        output += "{}".format(deterministic)
        return output


class Service(Enum):
    """
    Possible states for a service running on a machine, from the agents point
    of view

    Possible service state observations:
    0. unknown : the service may or may not be running on machine
    1. present : the service is running on the machine
    2. absent : the service is not running on the machine
    """
    unknown = 0
    present = 1
    absent = 2

    def __str__(self):
        if self.value == 0:
            return "unknown"
        if self.value == 1:
            return "present"
        return "absent"
