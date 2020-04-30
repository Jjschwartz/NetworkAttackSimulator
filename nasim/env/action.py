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

    def __init__(self, name, target, cost, prob=1.0):
        """
        Arguments
        ---------
        name : str
            name of action
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float
            probability of success for a given action
        """
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob

    def is_exploit(self):
        return False

    def is_scan(self):
        return False

    def is_service_scan(self):
        return False

    def is_os_scan(self):
        return False

    def is_subnet_scan(self):
        return False

    def __str__(self):
        return (f"{self.__class__.__name__}: name={self.name}, "
                f"target={self.target}, cost={self.cost:.2f}, "
                f"prob={self.prob:.2f}")

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, type(self)):
            return False
        elif self.target != other.target:
            return False
        elif not (math.isclose(self.cost, other.cost)
                  and math.isclose(self.prob, other.prob)):
            return False
        return (self.is_scan() == other.is_scan()
                and self.is_service_scan() == other.is_service_scan()
                and self.is_os_scan() == other.is_os_scan())

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
            action_space.append(ServiceScan("service_scan",
                                            address,
                                            scenario.service_scan_cost))
            action_space.append(OSScan("os_scan",
                                       address,
                                       scenario.os_scan_cost))
            action_space.append(SubnetScan("subnet_scan",
                                           address,
                                           scenario.subnet_scan_cost))
            for e_name, e_def in scenario.exploits.items():
                exploit = Exploit(e_name, address, **e_def)
                action_space.append(exploit)
        return action_space


class Exploit(Action):

    def __init__(self, name, target, cost, service, os=None, prob=1.0):
        """
        Arguments
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        service : str
            the target service
        os : str, optional
            the target OS of exploit, if None then exploit works for all OS
            (default=None)
        prob : float, optional
            probability of success (default=1.0)
        """
        super().__init__(name, target, cost, prob)
        self.os = os
        self.service = service

    def is_exploit(self):
        return True

    def __str__(self):
        return super().__str__() + f", os={self.os}, service={self.service}"

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service and self.os == other.os


class ServiceScan(Action):

    def is_scan(self):
        return True

    def is_service_scan(self):
        return True


class OSScan(Action):

    def is_scan(self):
        return True

    def is_os_scan(self):
        return True


class SubnetScan(Action):

    def is_scan(self):
        return True

    def is_subnet_scan(self):
        return True
