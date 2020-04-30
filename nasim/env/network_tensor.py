import numpy as np

from .host_vector import HostVector


class NetworkTensor:

    # class variables
    host_to_idx_map = None

    def __init__(self, network_tensor):
        """
        Arguments
        ---------
        network_tensor : np.Array
            the network as a tensor
        """
        self.tensor = network_tensor

    @classmethod
    def tensorize(cls, network):
        h0 = network.hosts[(1, 0)]
        h0_vector = HostVector.vectorize(h0)
        tensor = np.zeros((len(network.hosts),
                          h0_vector.state_size),
                          dtype=np.float32)

        if cls.host_to_idx_map is None:
            cls.host_to_idx_map = dict()
            for host_num, host_addr in enumerate(network.hosts):
                cls.host_to_idx_map[host_addr] = host_num

        for host_addr, host in network.hosts.items():
            host_num = cls.host_to_idx_map[host_addr]
            HostVector.vectorize(host, tensor[host_num])
        return NetworkTensor(tensor)

    @classmethod
    def tensorize_random(cls, network):
        h0 = network.hosts[(1, 0)]
        h0_vector = HostVector.vectorize_random(h0)
        tensor = np.zeros((len(network.hosts),
                          h0_vector.state_size),
                          dtype=np.float32)

        if cls.host_to_idx_map is None:
            cls.host_to_idx_map = dict()
            for host_num, host_addr in enumerate(network.hosts):
                cls.host_to_idx_map[host_addr] = host_num

        for host_addr, host in network.hosts.items():
            host_num = cls.host_to_idx_map[host_addr]
            HostVector.vectorize_random(host, tensor[host_num])
        return NetworkTensor(tensor)

    @classmethod
    def get_host_idx(cls, host_addr):
        return cls.host_to_idx_map[host_addr]

    @property
    def hosts(self):
        hosts = []
        for host_addr in self.host_to_idx_map:
            hosts.append((host_addr, self.get_host(host_addr)))
        return hosts

    def copy(self):
        new_tensor = np.copy(self.tensor)
        return NetworkTensor(new_tensor)

    def update_host(self, host_addr, host_vector):
        host_idx = self.host_to_idx_map[host_addr]
        self.tensor[host_idx] = host_vector.vector

    def get_host(self, host_addr):
        host_idx = self.host_to_idx_map[host_addr]
        return HostVector(self.tensor[host_idx])

    def get_host_and_idx(self, host_addr):
        host_idx = self.host_to_idx_map[host_addr]
        return host_idx, HostVector(self.tensor[host_idx])

    def host_reachable(self, host_addr):
        return self.get_host(host_addr).reachable

    def host_compromised(self, host_addr):
        return self.get_host(host_addr).compromised

    def host_discovered(self, host_addr):
        return self.get_host(host_addr).discovered

    def set_host_compromised(self, host_addr):
        self.get_host(host_addr).compromised = True

    def set_host_reachable(self, host_addr):
        self.get_host(host_addr).reachable = True

    def set_host_discovered(self, host_addr):
        self.get_host(host_addr).discovered = True

    def get_host_value(self, host_address):
        return self.hosts[host_address].get_value()

    def host_is_running_service(self, host_addr, service):
        return self.get_host(host_addr).is_running_service(service)

    def host_is_running_os(self, host_addr, os):
        return self.get_host(host_addr).is_running_os(os)

    def get_total_host_value(self):
        total_value = 0
        for host_addr in self.host_to_idx_map:
            host = self.get_host(host_addr)
            total_value += host.value
        return total_value

    def state_size(self):
        return self.tensor.size

    def get_readable(self):
        host_obs = []
        for host_addr in self.host_to_idx_map:
            host = self.get_host(host_addr)
            readable_dict = host.readable()
            host_obs.append(readable_dict)
        return host_obs

    def __str__(self):
        output = "\n--- Network Tensor ---\n"
        output += "Hosts:\n"
        for host in self.hosts:
            output += str(host) + "\n"
        return output

    def __hash__(self):
        return hash(str(self.tensor))

    def __eq__(self, other):
        return np.array_equal(self.tensor, other.tensor)
