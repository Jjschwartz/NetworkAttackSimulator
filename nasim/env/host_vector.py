import numpy as np


class HostVector:

    def __init__(self, host):
        self.num_os = len(host.os)
        self.num_services = len(host.services)
        self.state_size = 3 + self.num_services + self.num_os
        self.vector = self._vectorize(host)

    def _vectorize(self, host):
        vector = np.zeros(self.state_size, dtype=np.float32)
        vector[self._compromised_idx] = int(host._compromised)
        vector[self._reachable_idx] = int(host._reachable)
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

    @property
    def _compromised_idx(self):
        return 0

    @property
    def _reachable_idx(self):
        return self._compromised_idx+1

    @property
    def _value_idx(self):
        return self._reachable_idx+1

    @property
    def _service_start_idx(self):
        return self._value_idx+1

    def _get_service_idx(self, srv_num):
        return self._service_start_idx+srv_num

    @property
    def _os_start_idx(self):
        return self._service_start_idx+self.num_services

    def _get_os_idx(self, os_num):
        return self._os_start_idx+os_num
