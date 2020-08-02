"""Action related classes for the NASim environment.

This module contains the different action classes that are used
to implement actions within a NASim environment, along within the
different ActionSpace classes, and the ActionResult class.

Notes
-----

**Actions:**

Every action inherits from the base :class:`Action` class, which defines
some common attributes and functions. Different types of actions
are implemented as subclasses of the Action class.

Action types implemented:

- :class:`Exploit`
- :class:`ServiceScan`
- :class:`OSScan`
- :class:`SubnetScan`

**Action Spaces:**

There are two types of action spaces, depending on if you are using flat actions or not:

- :class:`FlatActionSpace`
- :class:`ParameterisedActionSpace`

"""

import math
import numpy as np
from gym import spaces


def load_action_list(scenario):
    """Load list of actions for environment for given scenario """
    action_list = []
    for address in scenario.address_space:
        action_list.append(ServiceScan(address,
                                       scenario.service_scan_cost))
        action_list.append(OSScan(address,
                                  scenario.os_scan_cost))
        action_list.append(SubnetScan(address,
                                      scenario.subnet_scan_cost))
        for e_name, e_def in scenario.exploits.items():
            exploit = Exploit(e_name, address, **e_def)
            action_list.append(exploit)
    return action_list


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

    def __init__(self, name, target, cost, prob=1.0, **kwargs):
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
        return isinstance(self, Exploit)

    def is_scan(self):
        """Check if action is a scan

        Returns
        -------
        bool
            True if action is scan, otherwise False
        """
        return isinstance(self, (ServiceScan, OSScan, SubnetScan))

    def is_service_scan(self):
        """Check if action is a service scan

        Returns
        -------
        bool
            True if action is service scan, otherwise False
        """
        return isinstance(self, ServiceScan)

    def is_os_scan(self):
        """Check if action is an OS scan

        Returns
        -------
        bool
            True if action is an OS scan, otherwise False
        """
        return isinstance(self, OSScan)

    def is_subnet_scan(self):
        """Check if action is a subnet scan

        Returns
        -------
        bool
            True if action is a subnet scan, otherwise False
        """
        return isinstance(self, SubnetScan)

    def is_noop(self):
        """Check if action is a do nothing action.

        Returns
        -------
        bool
            True if action is a noop action, otherwise False
        """
        return isinstance(self, NoOp)

    def __str__(self):
        return (f"{self.__class__.__name__}: "
                f"target={self.target}, "
                f"cost={self.cost:.2f}, "
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

    def __init__(self,
                 name,
                 target,
                 cost,
                 service,
                 os=None,
                 prob=1.0,
                 **kwargs):
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
        return (f"{self.__class__.__name__}: name={self.name}, "
                f"target={self.target}, cost={self.cost:.2f}, "
                f"prob={self.prob:.2f}, os={self.os}, service={self.service}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service and self.os == other.os


class ServiceScan(Action):
    """A Service Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self, target, cost, prob=1.0, **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        """
        super().__init__("service_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         **kwargs)


class OSScan(Action):
    """An OS Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self, target, cost, prob=1.0, **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        """
        super().__init__("os_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         **kwargs)


class SubnetScan(Action):
    """A Subnet Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self, target, cost, prob=1.0, **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        """
        super().__init__("subnet_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         **kwargs)


class NoOp(Action):
    """A do nothing action in the environment

    Inherits from the base Action Class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="noop", target=(1, 0), cost=0, prob=1.0)


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

    def __init__(self,
                 success,
                 value=0.0,
                 services=None,
                 os=None,
                 discovered=None,
                 connection_error=False):
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


class FlatActionSpace(spaces.Discrete):
    """Flat Action space for NASim environment.

    Inherits and implements the gym.spaces.Discrete action space

    ...

    Attributes
    ----------
    n : int
        the number of actions in the action space
    actions : list of Actions
        the list of the Actions in the action space
    """

    def __init__(self, scenario):
        """
        Parameters
        ---------
        scenario : Scenario
            scenario description
        """
        self.actions = load_action_list(scenario)
        super().__init__(len(self.actions))

    def get_action(self, action_idx):
        """Get Action object corresponding to action idx

        Parameters
        ----------
        action_idx : int
            the action idx

        Returns
        -------
        Action
            Corresponding Action object
        """
        assert isinstance(action_idx, int), \
            ("When using flat action space, action must be an integer"
             f" or an Action object: {action_idx} is invalid")
        return self.actions[action_idx]


class ParameterisedActionSpace(spaces.MultiDiscrete):
    """A parameterised action space for NASim environment.

    Inherits and implements the gym.spaces.MultiDiscrete action space, where
    each dimension corresponds to a different action parameter.

    The action parameters (in order) are:

    0. Action Type = [0, 3]
       where 0=Exploit, 1=ServiceScan, 2=OSScan, 3=SubnetScan
    1. Subnet = [0, #subnets-1]
       -1 since we don't include the internet subnet
    2. Host = [0, max subnets size-1]
    3. Service = [0, #services]
       Note, this is only important for exploits
    4. OS = [0, #OS+1]
       Where 0=None.
       Note, this is only important for exploits.

    ...

    Attributes
    ----------
    nvec : Numpy.Array
        vector of the of the size of each parameter
    actions : list of Actions
        the list of all the Actions in the action space
    """

    action_types = [Exploit, ServiceScan, OSScan, SubnetScan]

    def __init__(self, scenario):
        """
        Parameters
        ----------
        scenario : Scenario
            scenario description
        """
        self.scenario = scenario
        self.actions = load_action_list(scenario)

        nvec = [
            len(self.action_types),
            len(self.scenario.subnets)-1,
            max(self.scenario.subnets),
            self.scenario.num_services,
            self.scenario.num_os+1
        ]

        super().__init__(nvec)

    def get_action(self, action_vec):
        """Get Action object corresponding to action vector.

        Parameters
        ----------
        action_vector : list of ints or tuple of ints or Numpy.Array
            the action vector

        Returns
        -------
        Action
            Corresponding Action object

        Notes
        -----
        1. if host# specified in action vector is greater than
           the number of hosts in the specified subnet, then host#
           will be changed to host# % subnet size.
        2. if action is an exploit and parameters do not match
           any exploit definition in the scenario description then
           a NoOp action is returned with 0 cost.
        """
        assert isinstance(action_vec, (list, tuple, np.ndarray)), \
            ("When using parameterised action space, action must be an Action"
             f" object, a list or a numpy array: {action_vec} is invalid")
        a_class = self.action_types[action_vec[0]]
        # need to add one to subnet to account for Internet subnet
        subnet = action_vec[1]+1
        host = action_vec[2] % self.scenario.subnets[subnet]

        target = (subnet, host)

        if not (a_class == Exploit):
            # can ignore other action parameters
            kwargs = self._get_scan_action_def(a_class)
            return a_class(target=target, **kwargs)

        # action is exploit, so have to make sure it is valid choice
        # and get constant params (name, cost, prob)
        service = self.scenario.services[action_vec[3]]
        os = None if action_vec[4] == 0 else self.scenario.os[action_vec[4]-1]
        e_def = self._get_exploit_def(service, os)
        if e_def is None:
            return NoOp()
        return a_class(target=target, **e_def)

    def _get_scan_action_def(self, a_class):
        """Get the constants for scan actions definitions """
        if a_class == ServiceScan:
            return {"cost": self.scenario.service_scan_cost}
        elif a_class == OSScan:
            return {"cost": self.scenario.os_scan_cost}
        elif a_class == SubnetScan:
            return {"cost": self.scenario.subnet_scan_cost}
        else:
            raise TypeError(f"Not implemented for Action class {a_class}")

    def _get_exploit_def(self, service, os):
        """Check if exploit parameters are valid """
        e_map = self.scenario.exploit_map
        if service not in e_map:
            return None
        if os not in e_map[service]:
            return None
        return e_map[service][os]
