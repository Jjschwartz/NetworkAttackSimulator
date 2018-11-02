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
