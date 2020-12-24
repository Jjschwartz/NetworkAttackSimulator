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
- :class:`PrivilegeEscalation`
- :class:`ServiceScan`
- :class:`OSScan`
- :class:`SubnetScan`
- :class:`ProcessScan`
- :class:`NoOp`

**Action Spaces:**

There are two types of action spaces, depending on if you are using flat
actions or not:

- :class:`FlatActionSpace`
- :class:`ParameterisedActionSpace`

"""

import math
import numpy as np
from gym import spaces

from .utils import AccessLevel


def load_action_list(scenario):
    """Load list of actions for environment for given scenario

    Parameters
    ----------
    scenario : Scenario
        the scenario

    Returns
    -------
    list
        list of all actions in environment
    """
    action_list = []
    for address in scenario.address_space:
        action_list.append(
            ServiceScan(address, scenario.service_scan_cost)
        )
        action_list.append(
            OSScan(address, scenario.os_scan_cost)
        )
        action_list.append(
            SubnetScan(address, scenario.subnet_scan_cost)
        )
        action_list.append(
            ProcessScan(address, scenario.process_scan_cost)
        )
        for e_name, e_def in scenario.exploits.items():
            exploit = Exploit(e_name, address, **e_def)
            action_list.append(exploit)
        for pe_name, pe_def in scenario.privescs.items():
            privesc = PrivilegeEscalation(pe_name, address, **pe_def)
            action_list.append(privesc)
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
    req_access : AccessLevel,
        the required access level to perform action. For for on host actions
        (i.e. subnet scan, process scan, and privilege escalation) this will
        be the access on the target. For remote actions (i.e. service scan,
        os scan, and exploits) this will be the access on a pivot host (i.e.
        a compromised host that can reach the target).
    """

    def __init__(self,
                 name,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
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
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        assert 0 <= prob <= 1.0
        self.name = name
        self.target = target
        self.cost = cost
        self.prob = prob
        self.req_access = req_access

    def is_exploit(self):
        """Check if action is an exploit

        Returns
        -------
        bool
            True if action is exploit, otherwise False
        """
        return isinstance(self, Exploit)

    def is_privilege_escalation(self):
        """Check if action is privilege escalation action

        Returns
        -------
        bool
            True if action is privilege escalation action, otherwise False
        """
        return isinstance(self, PrivilegeEscalation)

    def is_scan(self):
        """Check if action is a scan

        Returns
        -------
        bool
            True if action is scan, otherwise False
        """
        return isinstance(self, (ServiceScan, OSScan, SubnetScan, ProcessScan))

    def is_remote(self):
        """Check if action is a remote action

        A remote action is one where the target host is a remote host (i.e. the
        action is not performed locally on the target)

        Returns
        -------
        bool
            True if action is remote, otherwise False
        """
        return isinstance(self, (ServiceScan, OSScan, Exploit))

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

    def is_process_scan(self):
        """Check if action is a process scan

        Returns
        -------
        bool
            True if action is a process scan, otherwise False
        """
        return isinstance(self, ProcessScan)

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
                f"prob={self.prob:.2f}, "
                f"req_access={self.req_access}")

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.target != other.target:
            return False
        if not (math.isclose(self.cost, other.cost)
                and math.isclose(self.prob, other.prob)):
            return False
        return self.req_access == other.req_access


class Exploit(Action):
    """An Exploit action in the environment

    Inherits from the base Action Class.

    ...

    Attributes
    ----------
    service : str
        the service targeted by exploit
    os : str
        the OS targeted by exploit. If None then exploit works for all OSs.
    access : int
        the access level gained on target if exploit succeeds.
    """

    def __init__(self,
                 name,
                 target,
                 cost,
                 service,
                 os=None,
                 access=0,
                 prob=1.0,
                 req_access=AccessLevel.USER,
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
        access : int, optional
            the access level gained on target if exploit succeeds (default=0)
        prob : float, optional
            probability of success (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.os = os
        self.service = service
        self.access = access

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"service={self.service}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.service == other.service \
            and self.os == other.os \
            and self.access == other.access


class PrivilegeEscalation(Action):
    """A privilege escalation action in the environment

    Inherits from the base Action Class.

    ...

    Attributes
    ----------
    process : str
        the process targeted by the privilege escalation. If None the action
        works independent of a process
    os : str
        the OS targeted by privilege escalation. If None then action works
        for all OSs.
    access : int
        the access level resulting from privilege escalation action
    """

    def __init__(self,
                 name,
                 target,
                 cost,
                 access,
                 process=None,
                 os=None,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        access : int
            the access level resulting from the privilege escalation
        process : str, optional
            the target process, if None the action does not require a process
            to work (default=None)
        os : str, optional
            the target OS of privilege escalation action, if None then action
            works for all OS (default=None)
        prob : float, optional
            probability of success (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__(name=name,
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access)
        self.access = access
        self.os = os
        self.process = process

    def __str__(self):
        return (f"{super().__str__()}, os={self.os}, "
                f"process={self.process}, access={self.access}")

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.process == other.process \
            and self.os == other.os \
            and self.access == other.access


class ServiceScan(Action):
    """A Service Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("service_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class OSScan(Action):
    """An OS Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("os_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class SubnetScan(Action):
    """A Subnet Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("subnet_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class ProcessScan(Action):
    """A Process Scan action in the environment

    Inherits from the base Action Class.
    """

    def __init__(self,
                 target,
                 cost,
                 prob=1.0,
                 req_access=AccessLevel.USER,
                 **kwargs):
        """
        Parameters
        ---------
        target : (int, int)
            address of target
        cost : float
            cost of performing action
        prob : float, optional
            probability of success for a given action (default=1.0)
        req_access : AccessLevel, optional
            the required access level to perform action
            (default=AccessLevel.USER)
        """
        super().__init__("process_scan",
                         target=target,
                         cost=cost,
                         prob=prob,
                         req_access=req_access,
                         **kwargs)


class NoOp(Action):
    """A do nothing action in the environment

    Inherits from the base Action Class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="noop",
                         target=(1, 0),
                         cost=0,
                         prob=1.0,
                         req_access=AccessLevel.NONE)


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
    processes : dict
        processes identified by action
    access : dict
        access gained by action
    discovered : dict
        host addresses discovered by action
    connection_error : bool
        True if action failed due to connection error (e.g. could
        not reach target)
    permission_error : bool
        True if action failed due to a permission error (e.g. incorrect access
        level to perform action)
    undefined_error : bool
        True if action failed due to an undefined error (e.g. random exploit
        failure)
    newly_discovered : dict
        host addresses discovered for the first time by action
    """

    def __init__(self,
                 success,
                 value=0.0,
                 services=None,
                 os=None,
                 processes=None,
                 access=None,
                 discovered=None,
                 connection_error=False,
                 permission_error=False,
                 undefined_error=False,
                 newly_discovered=None):
        """
        Parameters
        ----------
        success : bool
            True if exploit/scan was successful, False otherwise
        value : float, optional
            value gained from action (default=0.0)
        services : dict, optional
            services identified by action (default=None={})
        os : dict, optional
            OS identified by action (default=None={})
        processes : dict, optional
            processes identified by action (default=None={})
        access : dict, optional
            access gained by action (default=None={})
        discovered : dict, optional
            host addresses discovered by action (default=None={})
        connection_error : bool, optional
            True if action failed due to connection error (default=False)
        permission_error : bool, optional
            True if action failed due to a permission error (default=False)
        undefined_error : bool, optional
            True if action failed due to an undefined error (default=False)
        newly_discovered : dict, optional
            host addresses discovered for first time by action (default=None)
        """
        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.processes = {} if processes is None else processes
        self.access = {} if access is None else access
        self.discovered = {} if discovered is None else discovered
        self.connection_error = connection_error
        self.permission_error = permission_error
        self.undefined_error = undefined_error
        if newly_discovered is not None:
            self.newly_discovered = newly_discovered
        else:
            self.newly_discovered = {}

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
            processes=self.processes,
            access=self.access,
            discovered=self.discovered,
            connection_error=self.connection_error,
            permission_error=self.permission_error,
            newly_discovered=self.newly_discovered
        )

    def __str__(self):
        output = ["ActionObservation:"]
        for k, val in self.info().items():
            output.append(f"  {k}={val}")
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

    0. Action Type = [0, 5]

       Where:

         0=Exploit,

         1=PrivilegeEscalation,

         2=ServiceScan,

         3=OSScan,

         4=SubnetScan,

         5=ProcessScan,

    1. Subnet = [0, #subnets-1]

       -1 since we don't include the internet subnet

    2. Host = [0, max subnets size-1]
    3. OS = [0, #OS+1]

       Where 0=None.

    4. Service = [0, #services]
    5. Process = [0, #processes+1]

       Where 0=None.

    Note that OS, Service and Process are only importand for exploits and
    privilege escalation actions.

    ...

    Attributes
    ----------
    nvec : Numpy.Array
        vector of the of the size of each parameter
    actions : list of Actions
        the list of all the Actions in the action space
    """

    action_types = [
        Exploit,
        PrivilegeEscalation,
        ServiceScan,
        OSScan,
        SubnetScan,
        ProcessScan
    ]

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
            self.scenario.num_os+1,
            self.scenario.num_services,
            self.scenario.num_processes
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

        if a_class not in (Exploit, PrivilegeEscalation):
            # can ignore other action parameters
            kwargs = self._get_scan_action_def(a_class)
            return a_class(target=target, **kwargs)

        os = None if action_vec[3] == 0 else self.scenario.os[action_vec[3]-1]

        if a_class == Exploit:
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            service = self.scenario.services[action_vec[4]]
            a_def = self._get_exploit_def(service, os)
        else:
            # privilege escalation
            # have to make sure it is valid choice
            # and also get constant params (name, cost, prob, access)
            proc = self.scenario.processes[action_vec[5]]
            a_def = self._get_privesc_def(proc, os)

        if a_def is None:
            return NoOp()
        return a_class(target=target, **a_def)

    def _get_scan_action_def(self, a_class):
        """Get the constants for scan actions definitions """
        if a_class == ServiceScan:
            cost = self.scenario.service_scan_cost
        elif a_class == OSScan:
            cost = self.scenario.os_scan_cost
        elif a_class == SubnetScan:
            cost = self.scenario.subnet_scan_cost
        elif a_class == ProcessScan:
            cost = self.scenario.process_scan_cost
        else:
            raise TypeError(f"Not implemented for Action class {a_class}")
        return {"cost": cost}

    def _get_exploit_def(self, service, os):
        """Check if exploit parameters are valid """
        e_map = self.scenario.exploit_map
        if service not in e_map:
            return None
        if os not in e_map[service]:
            return None
        return e_map[service][os]

    def _get_privesc_def(self, proc, os):
        """Check if privilege escalation parameters are valid """
        pe_map = self.scenario.privesc_map
        if proc not in pe_map:
            return None
        if os not in pe_map[proc]:
            return None
        return pe_map[proc][os]
