import numpy as np

from .host_vector import HostVector
from .observation import Observation


class State:
    """A state in the NASim Environment.

    Each row in the state tensor represents the state of a single host on the
    network. For details on host the state a single host is represented see
    :class:`HostVector`

    ...

    Attributes
    ----------
    tensor : numpy.Array
        tensor representation of the state of network
    host_num_map : dict
        mapping from host address to host number (this is used
        to map host address to host row in the network tensor)
    """

    def __init__(self, network_tensor, host_num_map):
        """
        Parameters
        ----------
        state_tensor : np.Array
            the tensor representation of the network state
        host_num_map : dict
            mapping from host address to host number (this is used
            to map host address to host row in the network tensor)
        """
        self.tensor = network_tensor
        self.host_num_map = host_num_map

    @classmethod
    def tensorize(cls, network):
        h0 = network.hosts[(1, 0)]
        h0_vector = HostVector.vectorize(h0, network.address_space_bounds)
        tensor = np.zeros(
            (len(network.hosts), h0_vector.state_size),
            dtype=np.float32
        )
        for host_addr, host in network.hosts.items():
            host_num = network.host_num_map[host_addr]
            HostVector.vectorize(
                host, network.address_space_bounds, tensor[host_num]
            )
        return cls(tensor, network.host_num_map)

    @classmethod
    def generate_initial_state(cls, network):
        state = cls.tensorize(network)
        return network.reset(state)

    @classmethod
    def generate_random_initial_state(cls, network):
        h0 = network.hosts[(1, 0)]
        h0_vector = HostVector.vectorize_random(
            h0, network.address_space_bounds
        )
        tensor = np.zeros(
            (len(network.hosts), h0_vector.state_size),
            dtype=np.float32
        )
        for host_addr, host in network.hosts.items():
            host_num = network.host_num_map[host_addr]
            HostVector.vectorize_random(
                host, network.address_space_bounds, tensor[host_num]
            )
        state = cls(tensor, network.host_num_map)
        # ensure host state set correctly
        return network.reset(state)

    @classmethod
    def from_numpy(cls, s_array, state_shape, host_num_map):
        if s_array.shape != state_shape:
            s_array = s_array.reshape(state_shape)
        return State(s_array, host_num_map)

    @property
    def hosts(self):
        hosts = []
        for host_addr in self.host_num_map:
            hosts.append((host_addr, self.get_host(host_addr)))
        return hosts

    def copy(self):
        new_tensor = np.copy(self.tensor)
        return State(new_tensor, self.host_num_map)

    def get_initial_observation(self, fully_obs):
        """Get the initial observation of network.

        Returns
        -------
        Observation
            an observation object
        """
        obs = Observation(self.shape())
        if fully_obs:
            obs.from_state(self)
            return obs

        for host_addr, host in self.hosts:
            if not host.reachable:
                continue
            host_obs = host.observe(reachable=True,
                                    discovered=True)
            host_idx = self.get_host_idx(host_addr)
            obs.update_from_host(host_idx, host_obs)
        return obs

    def get_observation(self, action, action_result, fully_obs):
        """Get observation given last action and action result

        Parameters
        ----------
        action : Action
            last action performed
        action_result : ActionResult
            observation from performing action
        fully_obs : bool
            whether problem is fully observable or not

        Returns
        -------
        Observation
            an observation object
        """
        obs = Observation(self.shape())
        obs.from_action_result(action_result)
        if fully_obs:
            obs.from_state(self)
            return obs

        if not action_result.success:
            # action failed so no observation
            return obs

        t_idx, t_host = self.get_host_and_idx(action.target)
        obs_kwargs = dict(
            address=True,       # must be true for success
            compromised=False,
            reachable=True,     # must be true for success
            discovered=True,    # must be true for success
            value=False,
            # discovery_value=False,    # this is only added as needed
            services=False,
            os=False
        )
        if not action.is_scan():
            # exploit action, so get all observations for host
            obs_kwargs["compromised"] = True
            obs_kwargs["value"] = True
            obs_kwargs["services"] = True
            obs_kwargs["os"] = True
        elif action.is_service_scan():
            obs_kwargs["services"] = True
        elif action.is_os_scan():
            obs_kwargs["os"] = True
        elif action.is_subnet_scan():
            for host_addr, discovered in action_result.discovered.items():
                if not discovered:
                    continue
                d_idx, d_host = self.get_host_and_idx(host_addr)
                d_obs = d_host.observe(discovery_value=True,
                                       **obs_kwargs)
                obs.update_from_host(d_idx, d_obs)
            # this is for target host (where scan was performed on)
            obs_kwargs["compromised"] = True
        else:
            raise NotImplementedError(f"Action {action} not implemented")
        target_obs = t_host.observe(**obs_kwargs)
        obs.update_from_host(t_idx, target_obs)
        return obs

    def shape_flat(self):
        return self.numpy_flat().shape

    def shape(self):
        return self.tensor.shape

    def numpy_flat(self):
        return self.tensor.flatten()

    def numpy(self):
        return self.tensor

    def update_host(self, host_addr, host_vector):
        host_idx = self.host_num_map[host_addr]
        self.tensor[host_idx] = host_vector.vector

    def get_host(self, host_addr):
        host_idx = self.host_num_map[host_addr]
        return HostVector(self.tensor[host_idx])

    def get_host_idx(self, host_addr):
        return self.host_num_map[host_addr]

    def get_host_and_idx(self, host_addr):
        host_idx = self.host_num_map[host_addr]
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
        for host_addr in self.host_num_map:
            host = self.get_host(host_addr)
            total_value += host.value
        return total_value

    def state_size(self):
        return self.tensor.size

    def get_readable(self):
        host_obs = []
        for host_addr in self.host_num_map:
            host = self.get_host(host_addr)
            readable_dict = host.readable()
            host_obs.append(readable_dict)
        return host_obs

    def __str__(self):
        output = "\n--- State ---\n"
        output += "Hosts:\n"
        for host in self.hosts:
            output += str(host) + "\n"
        return output

    def __hash__(self):
        return hash(str(self.tensor))

    def __eq__(self, other):
        return np.array_equal(self.tensor, other.tensor)
