import numpy as np

from .action_obs import ActionObservation


class HostVector:

    # class properties that are the same for all hosts
    # these are set when calling vectorize method
    num_os = None
    os_idx_map = {}
    num_services = None
    service_idx_map = {}
    state_size = None

    # vector position constants
    _subnet_address_idx = 0
    _host_address_idx = _subnet_address_idx+1
    _compromised_idx = _host_address_idx+1
    _reachable_idx = _compromised_idx+1
    _discovered_idx = _reachable_idx+1
    _value_idx = _discovered_idx+1
    _discovery_value_idx = _value_idx+1
    _service_start_idx = _discovery_value_idx+1
    # to be initialized
    _os_start_idx = None

    def __init__(self, vector):
        self.vector = vector

    @classmethod
    def vectorize(cls, host, vector=None):
        cls.num_os = len(host.os)
        cls.num_services = len(host.services)
        cls.state_size = cls._service_start_idx + cls.num_os + cls.num_services
        cls._os_start_idx = cls._service_start_idx + cls.num_services

        if vector is None:
            vector = np.zeros(cls.state_size, dtype=np.float32)
        else:
            assert len(vector) == cls.state_size

        vector[cls._subnet_address_idx] = host.address[0]
        vector[cls._host_address_idx] = host.address[1]
        vector[cls._compromised_idx] = int(host.compromised)
        vector[cls._reachable_idx] = int(host.reachable)
        vector[cls._discovered_idx] = int(host.discovered)
        vector[cls._value_idx] = host.value
        vector[cls._discovery_value_idx] = host.discovery_value
        for srv_num, (srv_key, srv_val) in enumerate(host.services.items()):
            cls.service_idx_map[srv_key] = srv_num
            vector[cls._get_service_idx(srv_num)] = int(srv_val)
        for os_num, (os_key, os_val) in enumerate(host.os.items()):
            cls.os_idx_map[os_key] = os_num
            vector[cls._get_os_idx(os_num)] = int(os_val)
        return cls(vector)

    @classmethod
    def vectorize_random(cls, host, vector=None):
        hvec = cls.vectorize(host, vector)
        # random variables
        for srv_num in cls.service_idx_map.values():
            srv_val = np.random.randint(0, 2)
            hvec.vector[cls._get_service_idx(srv_num)] = srv_val

        chosen_os = np.random.choice(list(cls.os_idx_map.values()))
        for os_num in cls.os_idx_map.values():
            hvec.vector[cls._get_os_idx(os_num)] = int(os_num == chosen_os)
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
        return (self.vector[self._subnet_address_idx],
                self.vector[self._host_address_idx])

    @property
    def value(self):
        return self.vector[self._value_idx]

    @property
    def discovery_value(self):
        return self.vector[self._discovery_value_idx]

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

    def is_running_service(self, srv):
        srv_num = self.service_idx_map[srv]
        return self.vector[self._get_service_idx(srv_num)]

    def is_running_os(self, os):
        os_num = self.os_idx_map[os]
        return self.vector[self._get_os_idx(os_num)]

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
            return next_state, ActionObservation(True,
                                                 0,
                                                 services=self.services)

        if action.is_os_scan():
            return next_state, ActionObservation(True, 0, os=self.os)

        if self.is_running_service(action.service) and \
           (action.os is None or self.is_running_os(action.os)):
            # service and os is present so exploit is successful
            value = 0
            if not self.compromised:
                # to ensure a machine is not rewarded twice
                value = self.value
                next_state.compromised = True
            return next_state, ActionObservation(True,
                                                 value,
                                                 services=self.services,
                                                 os=self.os)
        # service absent, exploit fails
        return next_state, ActionObservation(False, 0)

    def observe(self,
                address=False,
                compromised=False,
                reachable=False,
                discovered=False,
                value=False,
                discovery_value=False,
                services=False,
                os=False):
        obs = np.zeros(self.state_size, dtype=np.float32)
        if address:
            i = self._subnet_address_idx
            obs[i] = self.vector[i]
            obs[self._host_address_idx] = self.vector[self._host_address_idx]
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
        if services:
            idxs = self._service_idx_slice()
            obs[idxs] = self.vector[idxs]
        if os:
            idxs = self._os_idx_slice()
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
    def _get_service_idx(cls, srv_num):
        return cls._service_start_idx+srv_num

    @classmethod
    def _service_idx_slice(cls):
        return slice(cls._service_start_idx, cls._os_start_idx)

    @classmethod
    def _get_os_idx(cls, os_num):
        return cls._os_start_idx+os_num

    @classmethod
    def _os_idx_slice(cls):
        return slice(cls._os_start_idx, cls.state_size)

    @classmethod
    def get_readable(cls, vector):
        readable_dict = dict()
        readable_dict["Address"] = (int(vector[cls._subnet_address_idx]),
                                    int(vector[cls._host_address_idx]))
        readable_dict["Compromised"] = bool(vector[cls._compromised_idx])
        readable_dict["Reachable"] = bool(vector[cls._reachable_idx])
        readable_dict["Discovered"] = bool(vector[cls._discovered_idx])
        readable_dict["Value"] = vector[cls._value_idx]
        readable_dict["Discovery Value"] = vector[cls._discovery_value_idx]
        for srv_num in range(cls.num_services):
            v = bool(vector[cls._get_service_idx(srv_num)])
            readable_dict[f"srv_{srv_num}"] = v
        for os_num in range(cls.num_os):
            v = bool(vector[cls._get_os_idx(os_num)])
            readable_dict[f"os_{os_num}"] = v
        return readable_dict

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
