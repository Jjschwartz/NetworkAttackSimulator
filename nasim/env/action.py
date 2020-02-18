import math


class Action:
    """The base abstract action class in the environment.

    All

    An action is defined by 3 properties:
        1. type: either exploit or scan
        2. service: which service the exploit is targeting, where services are
            identified by a number from 0 upto num_exploits in environment.
            This is None for scan type actions.
        3. target: the machine to launch action against. The machine is defined
            by the (subnet, machine_id) tuple
    """

    def __init__(self, target, cost, prob=1.0):
        """
        Arguments
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float
            probability of success for a given action
        """
        self.target = target
        self.cost = cost
        self.prob = prob

    def is_scan(self):
        raise NotImplementedError

    def __str__(self):
        return (f"{self.__class__.__name}: target={self.target}, "
                f"cost={self.cost:.4f}, prob={self.prob:.4f}")

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, type(self)):
            return False
        elif self.target != other.target:
            return False
        return (math.isclose(self.cost, other.cost)
                and math.isclose(self.prob, other.prob))

    @staticmethod
    def load_action_space(scenario):
        """Load the action space for the environment from exploits list

        Arguments
        ---------
        scenario : Scenario
            scenario description object

        Returns
        -------
        action_space : list
            list of actions
        """
        action_space = []
        for address in scenario.address_space:
            scan = ServiceScan(address, scenario.scan_cost)
            action_space.append(scan)
            for e_name, e_def in scenario.exploits.items():
                exploit = Exploit(address, **e_def)
                action_space.append(exploit)
        return action_space


class Exploit(Action):

    def __init__(self, target, cost, service, prob=1.0):
        """
        Arguments
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        service : str
            the target service
        prob : float
            probability of success
        """
        super().__init__(target, cost, prob)
        self.service = service

    def is_scan(self):
        return False

    def __str__(self):
        return super().__str__() + f"service={self.service}"

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service


class ServiceScan(Action):

    def is_scan(self):
        return True

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.is_scan() == other.is_scan()
