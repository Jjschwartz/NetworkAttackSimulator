import numpy as np


EXPLOIT_COST = 100.0
SCAN_COST = 100.0


class Action(object):
    """
    An action in the environment.

    An action is defined by 3 properties:
        1. type: either exploit or scan
        2. service: which service the exploit is targeting, where services are
            identified by a number from 0 upto num_exploits in environment.
            This is None for scan type actions.
        3. target: the machine to launch action against. The machine is defined
            by the (subnet, machine_id) tuple
    """

    def __init__(self, target, type="scan", service=None, prob=1.0):
        """
        Initialize a new action

        Arguments:
            tuple target : address of target
            str type : either "scan" or "exploit"
            int service : the target service for an exploit
            float prob : probability of success for a given action
        """
        self.target = target
        self.type = type
        self.service = service
        self.prob = prob

    def is_scan(self):
        return self.type == "scan"

    @property
    def cost(self):
        """
        The cost of performing the given action
        """
        return SCAN_COST if self.is_scan() else EXPLOIT_COST

    def __str__(self):
        return ("Action: target=" + str(self.target[0]) + "." +
                str(self.target[1]) + ", type=" + self.type + ", service=" +
                str(self.service))

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        elif self.target != other.target or self.type != other.type:
            return False
        else:
            return self.service == other.service

    def __lt__(self, other):
        """
        Sort by target, then type (scan then exploit), then service
        """
        if self.target == other.target:
            if self.type == other.type:
                if self.is_scan():
                    return True
                return self.service < other.service
            else:
                return self.is_scan()
        else:
            return self.target < other.target

    @staticmethod
    def generate_action_space(address_space, num_services, exploit_probs=1.0):
        """
        Generate the action space for the environment

        Success probabilities of each exploit are determined as follows:
            - None - probabilities generated randomly from uniform distribution
            - single-float - probability of each exploit is set to value
            - list of float - probability of each exploit is set to
                corresponding value in list

        For deterministic exploits set exploit_probs=1.0

        Arguments:
            list address_space : list of addresses for each machine in network
            int num_services : number of possible services running on machines
            None, int or list  exploit_probs :  success probability of exploits

        Returns:
            list action_space : list of actions
        """
        if exploit_probs is None:
            exploit_probs = np.random.random_sample(num_services)
        elif type(exploit_probs) is list:
            if len(exploit_probs) == num_services:
                raise ValueError("Lengh of exploit probability list must be "
                                 + "same as number of services")
            for e in exploit_probs:
                if e <= 0.0 or e > 1.0:
                    raise ValueError("Exploit probabilities must be > 0.0 "
                                     + "and <=1.0")
        else:
            if exploit_probs <= 0.0 or exploit_probs > 1.0:
                raise ValueError("Exploit probabilities must be > 0.0 and "
                                 + "<=1.0")
            exploit_probs = [exploit_probs] * num_services

        action_space = []
        for address in address_space:
            # add scan
            scan = Action(address, "scan")
            action_space.append(scan)
            for service in range(num_services):
                prob = exploit_probs[service]
                exploit = Action(address, "exploit", service, prob)
                action_space.append(exploit)
        return action_space
