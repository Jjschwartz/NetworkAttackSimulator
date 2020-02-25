import numpy as np

from .utils import OneHotBool, ServiceState


class HostState:
    """A class that represents the current state of a host

    Properties
    ----------
    host_num : int
        the host number
    addr : (int, int)
        address of host
    compromised : OneHotBool
        whether host has been compromised or not
    reachable : OneHotBool
        whether host is reachable or not
    services : dict
        with service name as key and ServiceState as value
    os : dict
        with each os as key and whether the host is running as value, including
        an entry for None (i.e. no obs for OS)
    """

    def __init__(self, host_num, addr, compromised, reachable, services, os):
        self.host_num = host_num
        self.addr = addr
        self._compromised = compromised
        self._reachable = reachable
        self._services = services
        self._os = os
        self._vector = self._vectorize()

    def _vectorize(self):
        vector = np.zeros(self.vector_size, dtype=np.int16)
        vector[self._compromised_idx + self._compromised] = 1
        vector[self._reachable_idx + self._reachable] = 1
        for i, srv_val in enumerate(self._services.values()):
            vector[self._service_idx(i) + srv_val] = 1
        for i, os_val in enumerate(self._os.values()):
            if os_val:
                vector[self._os_idx[i]] = 1
        return vector

    @property
    def compromised(self):
        return self._compromised

    def update_compromised(self, new_val):
        pass

    @property
    def vector_size(self):
        # 2*OneHotBool for controlled and reachable
        return 2*len(OneHotBool) + len(self._services)*len(ServiceState) + len(self._os)

    @property
    def _compromised_idx(self):
        return 0

    @property
    def _reachable_idx(self):
        return self._compromised_idx + 3

    @property
    def _service_start_idx(self):
        return self._reachable_idx + 3

    def _service_idx(self, srv_num):
        return self._service_start_idx + 3*srv_num

    @property
    def _os_start_idx(self):
        return self._service_start_idx + 3*len(self._services)

    def _os_idx(self, os_num):
        return self._os_start_idx + os_num
