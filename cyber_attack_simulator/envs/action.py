import numpy as np


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

    def __init__(self, target, cost, type="scan", service=None, prob=1.0):
        """
        Initialize a new action

        Arguments:
            (int, int) target : address of target
            float cost : cost of performing action
            str type : either "scan" or "exploit"
            int service : the target service for an exploit
            float prob : probability of success for a given action
        """
        self.target = target
        self.cost = cost
        self.type = type
        self.service = service
        self.prob = prob

    def is_scan(self):
        return self.type == "scan"

    def __str__(self):
        return ("Action: target={0}, cost={1}, type={2}, service={3}".format(
                    self.target, self.cost, self.type, self.service))

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
            return self.cost == other.cost and self.service == other.service

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
    def generate_action_space(address_space, num_services, exploit_cost, scan_cost,
                              exploit_probs=1.0):
        """
        Generate the action space for the environment

        Success probabilities of each exploit are determined as follows:
            - None - probabilities generated randomly from uniform distribution
            - "mixed" - probabilities randomly chosen from distribution of low: 0.2,
                med: 0.5 and high: 0.8 with probability of level based on attack complexity
                distribution of top 10 vulnerabilities in 2017.
            - single-float - probability of each exploit is set to value
            - list of float - probability of each exploit is set to
                corresponding value in list

        For deterministic exploits set exploit_probs=1.0

        Arguments:
            list address_space : list of addresses for each machine in network
            int num_services : number of possible services running on machines
            float exploit_cost : cost of performing an exploit action
            float scan_cost : cost of performing a scan action
            mixed exploit_probs :  success probability of exploits

        Returns:
            list action_space : list of actions
        """
        if exploit_probs is None:
            exploit_probs = np.random.random_sample(num_services)
        elif exploit_probs == 'mixed':
            # success probability of low, med, high attack complexity
            if num_services == 1:
                # for case where only 1 service ignore low probability actions
                # since could lead to unnecessarily long attack paths
                levels = [0.5, 0.8]
                probs = [0.5, 0.5]
            else:
                levels = [0.2, 0.5, 0.8]
                probs = [0.2, 0.4, 0.4]
            exploit_probs = np.random.choice(levels, num_services, p=probs)
        elif type(exploit_probs) is list:
            if len(exploit_probs) == num_services:
                raise ValueError("Lengh of exploit probability list must equal number of services")
            for e in exploit_probs:
                if e <= 0.0 or e > 1.0:
                    raise ValueError("Exploit probabilities must be > 0.0 and <=1.0")
        else:
            if exploit_probs <= 0.0 or exploit_probs > 1.0:
                raise ValueError("Exploit probabilities must be > 0.0 and <=1.0")
            exploit_probs = [exploit_probs] * num_services

        action_space = []
        for address in address_space:
            scan = Action(address, scan_cost, "scan")
            action_space.append(scan)
            for service in range(num_services):
                prob = exploit_probs[service]
                exploit = Action(address, exploit_cost, "exploit", service, prob)
                action_space.append(exploit)
        return action_space

    @staticmethod
    def load_action_space(address_space, service_exploits, scan_cost):
        """
        Load the action space for the environment from service exploits list

        Arguments:
            list address_space : list of addresses for each machine in network
            dict service_exploits : map from service name to (prob, cost) tuple
            float scan_cost : cost of performing a scan action

        Returns:
            list action_space : list of actions
        """
        action_space = []
        for address in address_space:
            scan = Action(address, scan_cost, "scan")
            action_space.append(scan)
            for service, val in service_exploits.items():
                prob = val[0]
                cost = val[1]
                exploit = Action(address, cost, "exploit", service, prob)
                action_space.append(exploit)
        return action_space
