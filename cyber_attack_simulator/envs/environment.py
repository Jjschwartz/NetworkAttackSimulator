from enum import Enum
from collections import OrderedDict
import copy
import sys
import numpy as np
from cyber_attack_simulator.envs.network import Network
import cyber_attack_simulator.envs.network as network
from cyber_attack_simulator.envs.action import Action

R_SENSITIVE = 9000.0
R_USER = 5000.0


class CyberAttackSimulatorEnv(object):
    """
    A simple simulated computer network with subnetworks and machines with
    different vulnerabilities.

    Properties:
    - current_state : the current knowledge the agent has observed. This is
        defined by a dictionary of the known status of each machine on the
        network.

        Status for each machine is defined by:
            1. service_info : list of ServiceState, for each service
            2. compromised : True/False
            3. sensitive : True/False (whether machine has sensitive info)
            4. reachable : True/False (whether machine is currently reachable)

    - action_space : the set of all actions allowed for environment
    """

    action_space = None
    current_state = None

    def __init__(self, num_machines, num_services):
        """
        Initialize a new environment and network with the specified number
        of exploits and machines

        Arguments:
            int num_machines : number of machines to include in network
                (minimum is 3)
            int num_services : number of services to use in environment
                (minimum is 1)
        """
        assert 0 < num_services
        assert 2 < num_machines
        self.num_services = num_services
        self.num_machines = num_machines
        self.network = Network(num_machines, num_services)
        self.address_space = self.network.get_address_space()
        self.action_space = Action.generate_action_space(
            self.address_space, self.num_services)
        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns the initial observation.

        Returns:
            dict obs : the intial observation of the network environment
        """
        obs = {}
        reward_machines = self.network.get_reward_machines()
        for m in self.address_space:
            # initially the status of services on machine are unknown
            service_info = np.full(self.num_services, ServiceState.unknown,
                                   ServiceState)
            compromised = False
            sensitive = False
            reachable = False
            if m[0] == network.EXPOSED:
                reachable = True
            if m in reward_machines:
                sensitive = True
            obs[m] = {"service_info": service_info, "compromised": compromised,
                      "sensitive": sensitive, "reachable": reachable}
        self.current_state = OrderedDict(sorted(obs.items()))
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
        if not self._is_reachable(action.target):
            return self.current_state, 0 - action.cost, False, {}

        success, value, services = self.network.perform_action(action)

        value = 0 if self._already_rewarded(action.target) else value
        self._update_state(action, success, services)
        done = self.is_goal()
        reward = value - action.cost
        return copy.deepcopy(self.current_state), reward, done, {}

    def _is_reachable(self, target):
        """
        Checks if a given target machine is reachable

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool reachable : True if reachable, otherwise False
        """
        return self.current_state[target]["reachable"]

    def _already_rewarded(self, m):
        """
        Checks if the reward, if any, from exploting a machine has already
        been collected (i.e. sensitive documents have already been collected).

        Returns:
            bool recieved : True if reward was already recieved from this
                machine, otherwise False
        """
        return self.current_state[m]["compromised"]

    def _update_state(self, action, success, services):
        """
        Updates the current state of network state based on if action was
        successful and the gained service info

        Arguments:
            Action action : the action performed
            bool success : whether action was successful
            list services : service info gained from action
        """
        m_service_info = self.current_state[action.target]["service_info"]
        if action.is_scan() or (not action.is_scan() and success):
            # 1. scan or successful exploit, all service info gained for target
            for s in range(len(services)):
                if services[s]:
                    m_service_info[s] = ServiceState.present
                else:
                    m_service_info[s] = ServiceState.absent
            if not action.is_scan():
                # successful exploit so machine compromised
                self.current_state[action.target]["compromised"] = True
                self._update_reachable(action.target)
        else:
            # 2. unsuccessful exploit, targeted service is absent
            m_service_info[action.service] = ServiceState.absent

    def _update_reachable(self, compromised_m):
        """
        Updates the reachable status of machines on network, based on current
        state and newly exploited machine

        Arguments:
            (int, int) compromised_m : compromised machine address
        """
        comp_subnet = compromised_m[0]
        for m in self.address_space:
            if self.current_state[m]["reachable"]:
                continue
            m_subnet = m[0]
            if self.network.subnets_connected(comp_subnet, m_subnet):
                self.current_state[m]["reachable"] = True

    def is_goal(self):
        """
        Check if the current state is the goal state.
        The goal state is  when all sensitive documents have been collected
        (i.e. all rewarded machines have been compromised)

        Returns:
            bool goal : True if goal state, otherwise False
        """
        for m in self.network.get_reward_machines():
            if not self.current_state[m]["compromised"]:
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
        outfile = sys.stdout

        subnets = [[], [], []]
        for m, v in self.current_state.items():
            subnets[m[0]].append(self._get_machine_symbol(v))

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

    def _get_machine_symbol(self, m_state):
        if m_state["sensitive"]:
            if m_state["compromised"]:
                symbol = "C"
            elif m_state["reachable"]:
                symbol = "R"
            else:
                symbol = "S"
        elif m_state["compromised"]:
            symbol = "c"
        elif m_state["reachable"]:
            symbol = "r"
        else:
            symbol = "o"
        return symbol


class ServiceState(Enum):
    """
    Possible states for a service running on a machine, from the agents point
    of view

    Possible service state observations:
    1. present : the service is running on the machine
    2. absent : the service is not running on the machine
    3. unknown : the service may or may not be running on machine
    """
    present = 1
    absent = 2
    unknown = 3

    def __str__(self):
        if self.value == 1:
            return "present"
        if self.value == 2:
            return "absent"
        return "unknown"
