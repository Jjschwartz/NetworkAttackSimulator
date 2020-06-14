""" Action related classes for the NASim environment.

This module contains the different action classes that are used
to define implement actions within a NASim environment.

Every action inherits from the base Action class, which defines
some common attributes and functions. Different types of actions
are implemented as subclasses of the Action class.

Action types implemented:

- Exploit
- ServiceScan
- OSScan
- SubnetScan

Additionally, it also contains the ActionResult dataclass for storing
the results of performing an action.
"""

import math


class Action:
    """The base abstract action class in the environment

    There are multiple types of actions (e.g. exploit, scan, etc.), but every
    action has some common attributes.

    ...

    Attributes
    ----------
    name : str
        the name of action
    target : (int, int)
        the (subnet, host) address of target of the action. The target of the
        action could be the address of a host that the action is being used
        against (e.g. for exploits or targeted scans) or could be the host that
        the action is being executed on (e.g. for subnet scans).
    cost : float
        the cost of performing the action
    prob : float
        the success probability of the action. This is the probability that
        the action works given that it's preconditions are met. E.g. a remote
        exploit targeting a host that you cannot communicate with will always
        fail. For deterministic actions this will be 1.0.
    """

    def __init__(self, name, target, cost, prob=1.0):
        """
        Parameters
        ---------
        name : str
            name of action
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        """
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob

    def is_exploit(self):
        """Check if action is an exploit

        Returns
        -------
        bool
            True if action is exploit, otherwise False
        """
        return type(self, Exploit)

    def is_scan(self):
        """Check if action is a scan

        Returns
        -------
        bool
            True if action is scan, otherwise False
        """
        return type(self, ServiceScan, OSScan, SubnetScan)

    def is_service_scan(self):
        """Check if action is a service scan

        Returns
        -------
        bool
            True if action is service scan, otherwise False
        """
        return type(self, ServiceScan)

    def is_os_scan(self):
        """Check if action is an OS scan

        Returns
        -------
        bool
            True if action is an OS scan, otherwise False
        """
        return type(self, OSScan)

    def is_subnet_scan(self):
        """Check if action is a subnet scan

        Returns
        -------
        bool
            True if action is a subnet scan, otherwise False
        """
        return type(self, SubnetScan)

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
        """Load the action space for the environment from scenario

        Parameters
        ---------
        scenario : Scenario
            scenario description

        Returns
        -------
        list
            list of actions for environment
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
    """An Exploit action in the environment

    Inherits from the base Action Class. It overrides the is_exploit() method
    and adds some additional attributes.

    ...


    Attributes
    ----------
    service : str
        the service targeted by exploit
    os : str
        the OS targeted by exploit. If None then exploit works for all OSs.
    """

    def __init__(self, name, target, cost, service, os=None, prob=1.0):
        """
        Parameters
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

    def __str__(self):
        return super().__str__() + f", os={self.os}, service={self.service}"

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service and self.os == other.os


class ServiceScan(Action):
    """A Service Scan action in the environment

    Inherits from the base Action Class.
    """
    pass


class OSScan(Action):
    """An OS Scan action in the environment

    Inherits from the base Action Class.
    """
    pass


class SubnetScan(Action):
    """A Subnet Scan action in the environment

    Inherits from the base Action Class.
    """
    pass


class ActionResult:
    """A dataclass for storing the results of an Action.

    These results are then used to update the full state and observation.

    ...

    Attributes
    ----------
    success : bool
        True if exploit/scan was successful, False otherwise
    value : float
        value gained from action. Is the value of the host if successfuly
        exploited, otherwise 0
    services : dict
        services identified by action.
    os : dict
        OS identified by action
    discovered : dict
        host addresses discovered by action
    connection_error : bool
        True if action failed due to connection error (e.g. could
        not reach target)
    """

    def __init__(self, success, value=0.0, services=None, os=None,
                 discovered=None, connection_error=False):
        """
        Parameters
        ----------
        success : bool
            True if exploit/scan was successful, False otherwise
        value : float, optional
            value gained from action (default=0.0)
        services : dict, optional
            services identified by action (default=None)
        os : dict, optional
            OS identified by action (default=None)
        discovered : dict, optional
            host addresses discovered by action (default=None)
        connection_error : bool, optional
            True if action failed due to connection error (default=None)
        """
        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.discovered = {} if discovered is None else discovered
        self.connection_error = connection_error

    def info(self):
        """Get results as dict

        Returns
        -------
        dict
            action results information
        """
        return dict(
            success=self.success,
            value=self.value,
            services=self.services,
            os=self.os,
            discovered=self.discovered,
            connection_error=self.connection_error
        )

    def __str__(self):
        output = ["ActionObservation:"],
        for k, v in self.info().items():
            output.append(f"  {k}={v}")
        return "\n".join(output)
