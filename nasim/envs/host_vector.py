""" This module contains the HostVector class.

This is the main class for storing and updating the state of a single host
in the NASim environment.
"""

import numpy as np

from .utils import AccessLevel
from .action import ActionResult


class HostVector:
    """ A Vector representation of a single host in NASim.

    Each host is represented as a vector (1D numpy array) for efficiency and to
    make it easier to use with deep learning agents. The vector is made up of
    multiple features arranged in a consistent way.

    Features in the vector, listed in order, are:

    1. subnet address - one-hot encoding with length equal to the number
                        of subnets
    2. host address - one-hot encoding with length equal to the maximum number
                      of hosts in any subnet
    3. compromised - bool
    4. reachable - bool
    5. discovered - bool
    6. value - float
    7. discovery value - float
    8. access - int
    9. OS - bool for each OS in scenario (only one OS has value of true)
    10. services running - bool for each service in scenario
    11. processes running - bool for each process in scenario

    Notes
    -----
    - The size of the vector is equal to:

        #subnets + max #hosts in any subnet + 6 + #OS + #services + #processes.

    - Where the +6 is for compromised, reachable, discovered, value,
      discovery_value, and access features
    - The vector is a float vector so True/False is actually represented as
      1.0/0.0.

    """

    # class properties that are the same for all hosts
    # these are set when calling vectorize method
    # the bounds on address space (used for one hot encoding of host address)
    address_space_bounds = None
    # number of OS in scenario
    num_os = None
    # map from OS name to its index in host vector
    os_idx_map = {}
    # number of services in scenario
    num_services = None
    # map from service name to its index in host vector
    service_idx_map = {}
    # number of processes in scenario
    num_processes = None
    # map from process name to its index in host vector
    process_idx_map = {}
    # size of state for host vector (i.e. len of vector)
    state_size = None

    # vector position constants
    # to be initialized
    _subnet_address_idx = 0
    _host_address_idx = None
    _compromised_idx = None
    _reachable_idx = None
    _discovered_idx = None
    _value_idx = None
    _discovery_value_idx = None
    _access_idx = None
    _os_start_idx = None
    _service_start_idx = None
    _process_start_idx = None

    def __init__(self, vector):
        self.vector = vector

    @classmethod
    def vectorize(cls, host, address_space_bounds, vector=None):
        if cls.address_space_bounds is None:
            cls._initialize(
                address_space_bounds, host.services, host.os, host.processes
            )

        if vector is None:
            vector = np.zeros(cls.state_size, dtype=np.float32)
        else:
            assert len(vector) == cls.state_size

        vector[cls._subnet_address_idx + host.address[0]] = 1
        vector[cls._host_address_idx + host.address[1]] = 1
        vector[cls._compromised_idx] = int(host.compromised)
        vector[cls._reachable_idx] = int(host.reachable)
        vector[cls._discovered_idx] = int(host.discovered)
        vector[cls._value_idx] = host.value
        vector[cls._discovery_value_idx] = host.discovery_value
        vector[cls._access_idx] = host.access
        for os_num, (os_key, os_val) in enumerate(host.os.items()):
            vector[cls._get_os_idx(os_num)] = int(os_val)
        for srv_num, (srv_key, srv_val) in enumerate(host.services.items()):
            vector[cls._get_service_idx(srv_num)] = int(srv_val)
        host_procs = host.processes.items()
        for proc_num, (proc_key, proc_val) in enumerate(host_procs):
            vector[cls._get_process_idx(proc_num)] = int(proc_val)
        return cls(vector)

    @classmethod
    def vectorize_random(cls, host, address_space_bounds, vector=None):
        hvec = cls.vectorize(host, vector)
        # random variables
        for srv_num in cls.service_idx_map.values():
            srv_val = np.random.randint(0, 2)
            hvec.vector[cls._get_service_idx(srv_num)] = srv_val

        chosen_os = np.random.choice(list(cls.os_idx_map.values()))
        for os_num in cls.os_idx_map.values():
            hvec.vector[cls._get_os_idx(os_num)] = int(os_num == chosen_os)

        for proc_num in cls.process_idx_map.values():
            proc_val = np.random.randint(0, 2)
            hvec.vector[cls._get_process_idx(proc_num)] = proc_val
        return hvec

    @property
    def compromised(self):
        return self.vector[self._compromised_idx]

    @compromised.setter
    def compromised(self, val):
        self.vector[self._compromised_idx] = int(val)

    @property
    def discovered(self):
        return self.vector[self._discovered_idx]

    @discovered.setter
    def discovered(self, val):
        self.vector[self._discovered_idx] = int(val)

    @property
    def reachable(self):
        return self.vector[self._reachable_idx]

    @reachable.setter
    def reachable(self, val):
        self.vector[self._reachable_idx] = int(val)

    @property
    def address(self):
        return (
            self.vector[self._subnet_address_idx_slice()].argmax(),
            self.vector[self._host_address_idx_slice()].argmax()
        )

    @property
    def value(self):
        return self.vector[self._value_idx]

    @property
    def discovery_value(self):
        return self.vector[self._discovery_value_idx]

    @property
    def access(self):
        return self.vector[self._access_idx]

    @access.setter
    def access(self, val):
        self.vector[self._access_idx] = int(val)

    @property
    def services(self):
        services = {}
        for srv, srv_num in self.service_idx_map.items():
            services[srv] = self.vector[self._get_service_idx(srv_num)]
        return services

    @property
    def os(self):
        os = {}
        for os_key, os_num in self.os_idx_map.items():
            os[os_key] = self.vector[self._get_os_idx(os_num)]
        return os

    @property
    def processes(self):
        processes = {}
        for proc, proc_num in self.process_idx_map.items():
            processes[proc] = self.vector[self._get_process_idx(proc_num)]
        return processes

    def is_running_service(self, srv):
        srv_num = self.service_idx_map[srv]
        return bool(self.vector[self._get_service_idx(srv_num)])

    def is_running_os(self, os):
        os_num = self.os_idx_map[os]
        return bool(self.vector[self._get_os_idx(os_num)])

    def is_running_process(self, proc):
        proc_num = self.process_idx_map[proc]
        return bool(self.vector[self._get_process_idx(proc_num)])

    def perform_action(self, action):
        """Perform given action against this host

        Arguments
        ---------
        action : Action
            the action to perform

        Returns
        -------
        HostVector
            the resulting state of host after action
        ActionObservation
            the result from the action
        """
        next_state = self.copy()
        if action.is_service_scan():
            result = ActionResult(True, 0, services=self.services)
            return next_state, result

        if action.is_os_scan():
            return next_state, ActionResult(True, 0, os=self.os)

        if action.is_exploit():
            if self.is_running_service(action.service) and \
               (action.os is None or self.is_running_os(action.os)):
                # service and os is present so exploit is successful
                value = 0
                next_state.compromised = True
                if not self.access == AccessLevel.ROOT:
                    # ensure a machine is not rewarded twice
                    # and access doesn't decrease
                    next_state.access = action.access
                    if action.access == AccessLevel.ROOT:
                        value = self.value

                result = ActionResult(
                    True,
                    value=value,
                    services=self.services,
                    os=self.os,
                    access=action.access
                )
                return next_state, result

        # following actions are on host so require correct access
        if not self.compromised and action.req_access <= self.access:
            result = ActionResult(False, 0, permission_error=True)
            return next_state, result

        if action.is_process_scan():
            result = ActionResult(
                True, 0, access=self.access, processes=self.processes
            )
            return next_state, result

        if action.is_privilege_escalation():
            has_proc = (
                action.process is None
                or self.is_running_process(action.process)
            )
            has_os = (
                action.os is None or self.is_running_os(action.os)
            )
            if has_proc and has_os:
                # host compromised and proc and os is present
                # so privesc is successful
                value = 0.0
                if not self.access == AccessLevel.ROOT:
                    # ensure a machine is not rewarded twice
                    # and access doesn't decrease
                    next_state.access = action.access
                    if action.access == AccessLevel.ROOT:
                        value = self.value
                result = ActionResult(
                    True,
                    value=value,
                    processes=self.processes,
                    os=self.os,
                    access=action.access
                )
                return next_state, result

        # action failed due to host config not meeting preconditions
        return next_state, ActionResult(False, 0)

    def observe(self,
                address=False,
                compromised=False,
                reachable=False,
                discovered=False,
                access=False,
                value=False,
                discovery_value=False,
                services=False,
                processes=False,
                os=False):
        obs = np.zeros(self.state_size, dtype=np.float32)
        if address:
            subnet_slice = self._subnet_address_idx_slice()
            host_slice = self._host_address_idx_slice()
            obs[subnet_slice] = self.vector[subnet_slice]
            obs[host_slice] = self.vector[host_slice]
        if compromised:
            obs[self._compromised_idx] = self.vector[self._compromised_idx]
        if reachable:
            obs[self._reachable_idx] = self.vector[self._reachable_idx]
        if discovered:
            obs[self._discovered_idx] = self.vector[self._discovered_idx]
        if value:
            obs[self._value_idx] = self.vector[self._value_idx]
        if discovery_value:
            v = self.vector[self._discovery_value_idx]
            obs[self._discovery_value_idx] = v
        if access:
            obs[self._access_idx] = self.vector[self._access_idx]
        if os:
            idxs = self._os_idx_slice()
            obs[idxs] = self.vector[idxs]
        if services:
            idxs = self._service_idx_slice()
            obs[idxs] = self.vector[idxs]
        if processes:
            idxs = self._process_idx_slice()
            obs[idxs] = self.vector[idxs]
        return obs

    def readable(self):
        return self.get_readable(self.vector)

    def copy(self):
        vector_copy = np.copy(self.vector)
        return HostVector(vector_copy)

    def numpy(self):
        return self.vector

    @classmethod
    def _initialize(cls, address_space_bounds, services, os_info, processes):
        cls.os_idx_map = {}
        cls.service_idx_map = {}
        cls.process_idx_map = {}
        cls.address_space_bounds = address_space_bounds
        cls.num_os = len(os_info)
        cls.num_services = len(services)
        cls.num_processes = len(processes)
        cls._update_vector_idxs()
        for os_num, (os_key, os_val) in enumerate(os_info.items()):
            cls.os_idx_map[os_key] = os_num
        for srv_num, (srv_key, srv_val) in enumerate(services.items()):
            cls.service_idx_map[srv_key] = srv_num
        for proc_num, (proc_key, proc_val) in enumerate(processes.items()):
            cls.process_idx_map[proc_key] = proc_num

    @classmethod
    def _update_vector_idxs(cls):
        cls._subnet_address_idx = 0
        cls._host_address_idx = cls.address_space_bounds[0]
        cls._compromised_idx = (
            cls._host_address_idx + cls.address_space_bounds[1]
        )
        cls._reachable_idx = cls._compromised_idx + 1
        cls._discovered_idx = cls._reachable_idx + 1
        cls._value_idx = cls._discovered_idx + 1
        cls._discovery_value_idx = cls._value_idx + 1
        cls._access_idx = cls._discovery_value_idx + 1
        cls._os_start_idx = cls._access_idx + 1
        cls._service_start_idx = cls._os_start_idx + cls.num_os
        cls._process_start_idx = cls._service_start_idx + cls.num_services
        cls.state_size = cls._process_start_idx + cls.num_processes

    @classmethod
    def _subnet_address_idx_slice(cls):
        return slice(cls._subnet_address_idx, cls._host_address_idx)

    @classmethod
    def _host_address_idx_slice(cls):
        return slice(cls._host_address_idx, cls._compromised_idx)

    @classmethod
    def _get_service_idx(cls, srv_num):
        return cls._service_start_idx+srv_num

    @classmethod
    def _service_idx_slice(cls):
        return slice(cls._service_start_idx, cls._process_start_idx)

    @classmethod
    def _get_os_idx(cls, os_num):
        return cls._os_start_idx+os_num

    @classmethod
    def _os_idx_slice(cls):
        return slice(cls._os_start_idx, cls._service_start_idx)

    @classmethod
    def _get_process_idx(cls, proc_num):
        return cls._process_start_idx+proc_num

    @classmethod
    def _process_idx_slice(cls):
        return slice(cls._process_start_idx, cls.state_size)

    @classmethod
    def get_readable(cls, vector):
        readable_dict = dict()
        hvec = cls(vector)
        readable_dict["Address"] = hvec.address
        readable_dict["Compromised"] = bool(hvec.compromised)
        readable_dict["Reachable"] = bool(hvec.reachable)
        readable_dict["Discovered"] = bool(hvec.discovered)
        readable_dict["Value"] = hvec.value
        readable_dict["Discovery Value"] = hvec.discovery_value
        readable_dict["Access"] = hvec.access
        for os_name in cls.os_idx_map:
            readable_dict[f"{os_name}"] = hvec.is_running_os(os_name)
        for srv_name in cls.service_idx_map:
            readable_dict[f"{srv_name}"] = hvec.is_running_service(srv_name)
        for proc_name in cls.process_idx_map:
            readable_dict[f"{proc_name}"] = hvec.is_running_process(proc_name)

        return readable_dict

    @classmethod
    def reset(cls):
        """Resets any class variables.

        This is used to avoid errors when changing scenarios within a single
        python session
        """
        cls.address_space_bounds = None

    def __repr__(self):
        return f"Host: {self.address}"

    def __hash__(self):
        return hash(str(self.vector))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, HostVector):
            return False
        return np.array_equal(self.vector, other.vector)
