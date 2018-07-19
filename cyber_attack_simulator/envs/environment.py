from enum import Enum
from collections import OrderedDict
import copy
import numpy as np
from cyber_attack_simulator.envs.network import Network
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

    def __init__(self, num_machines, num_exploits):
        """
        Initialize a new environment and network with the specified number
        of exploits and machines

        Arguments:
            int num_machines : number of machines to include in network
                (minimum is 3)
            int num_exploits : number of exploits (and hence services) to use
                in environment (minimum is 1)
        """
        assert 0 < num_exploits
        assert 2 < num_machines
        self.num_exploits = num_exploits
        self.num_machines = num_machines
        self.network = Network(num_machines, num_exploits)
        self._generate_action_space()
        self.reset()

    def _generate_action_space(self):
        """
        Generate the action space for the environment which consists of for
        each machine in the network, an exploit for each service (i.e.
        num_exploits) and a scan
        """
        machines = self.network.get_machines()
        temp = set()
        for m in machines:
            temp.add(Action(m, "scan", None))
            for s in range(self.num_exploits):
                temp.add(Action(m, "exploit", s))
        self.action_space = temp

    def reset(self):
        """
        Reset the state of the environment and returns the initial observation.

        Returns:
            dict obs : the intial observation of the network environment
        """
        obs = {}
        machines = self.network.get_machines()
        for m in machines:
            # initially the status of services on machine are unknown
            service_info = np.full(self.num_exploits, ServiceState.unknown,
                                   ServiceState)
            compromised = False
            if m[0] == 1:
                # machine on subnet, has no sensitive info
                sensitive = False
                # only machines on exposed subnet are reachable at start
                reachable = True
            else:
                # every 10th machine has sensitive info
                sensitive = m[1] % 10 == 0
                reachable = False
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

    def render(self):
        """
        Renders the environment and outputs it to stdout
        """
        for m in self.current_state.keys():
            print("Machine: {0} {1}".format(m, self.current_state[m]))

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
        # if machine on subnet 1 is compromised then subnets 1 and 2 are
        # reachable
        if compromised_m[0] == 1:
            for m in self.current_state.keys():
                self.current_state[m]["reachable"] = True
        # otherwise all machines all already reachable
        # TODO update for when more than 3 subnets

    def is_goal(self):
        """
        Check if the current state is the goal state. Where the goal
        state is defined as when all sensitive documents have been collected
        (i.e. all machines containing sensitive documents have been
        compromised)

        Returns:
            bool goal : True if goal state, otherwise False
        """
        for m in self.network.get_sensitive_machines():
            if not self.current_state[m]["compromised"]:
                return False
        return True


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
