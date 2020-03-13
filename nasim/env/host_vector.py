import numpy as np


class HostVector:

    def __init__(self, host):
        self.num_os = len(host.os)
        self.num_services = len(host.services)
        self.state_size = self._service_start_idx + self.num_services + self.num_os
        self.vector = self._vectorize(host)

    def _vectorize(self, host):
        vector = np.zeros(self.state_size, dtype=np.float32)
        vector[self._compromised_idx] = int(host._compromised)
        vector[self._reachable_idx] = int(host._reachable)
        vector[self._discovered_idx] = int(host._discovered)
        vector[self._value_idx] = host.value
        for srv_num, srv_val in enumerate(host.services.values()):
            vector[self._get_service_idx(srv_num)] = int(srv_val)
        for os_num, os_val in enumerate(host.os.values()):
            vector[self._get_os_idx(os_num)] = int(os_val)
        return vector

    def set_compromised(self, val):
        self.vector[self._compromised_idx] = int(val)

    def set_reachable(self, val):
        self.vector[self._reachable_idx] = int(val)

    def set_discovered(self, val):
        self.vector[self._discovered_idx] = int(val)

    def observe(self, compromised=False, reachable=False, discovered=False,
                value=False, services=False, os=False):
        obs = np.zeros(self.state_size, dtype=np.float32)
        if compromised:
            obs[self._compromised_idx] = self.vector[self._compromised_idx]
        if reachable:
            obs[self._reachable_idx] = self.vector[self._reachable_idx]
        if discovered:
            obs[self._discovered_idx] = self.vector[self._discovered_idx]
        if value:
            obs[self._value_idx] = self.vector[self._value_idx]
        if services:
            obs[self._service_idx_slice] = self.vector[self._service_idx_slice]
        if os:
            obs[self._os_idx_slice] = self.vector[self._os_idx_slice]
        return obs

    @property
    def _compromised_idx(self):
        return 0

    @property
    def _reachable_idx(self):
        return self._compromised_idx+1

    @property
    def _discovered_idx(self):
        return self._reachable_idx+1

    @property
    def _value_idx(self):
        return self._discovered_idx+1

    @property
    def _service_start_idx(self):
        return self._value_idx+1

    def _get_service_idx(self, srv_num):
        return self._service_start_idx+srv_num

    @property
    def _service_idx_slice(self):
        return slice(self._service_start_idx, self._os_start_idx)

    @property
    def _os_start_idx(self):
        return self._service_start_idx+self.num_services

    def _get_os_idx(self, os_num):
        return self._os_start_idx+os_num

    @property
    def _os_idx_slice(self):
        return slice(self._os_start_idx, self.state_size)

    def readable(self):
        return self.get_readable(self.vector)

    def get_readable(self, vector):
        readable_dict = dict()
        readable_dict["Compromised"] = bool(vector[self._compromised_idx])
        readable_dict["Reachable"] = bool(vector[self._reachable_idx])
        readable_dict["Discovered"] = bool(vector[self._discovered_idx])
        readable_dict["Value"] = vector[self._value_idx]
        for srv_num in range(self.num_services):
            readable_dict[f"srv_{srv_num}"] = bool(vector[self._get_service_idx(srv_num)])
        for os_num in range(self.num_os):
            readable_dict[f"os_{os_num}"] = bool(vector[self._get_os_idx(os_num)])
        return readable_dict
