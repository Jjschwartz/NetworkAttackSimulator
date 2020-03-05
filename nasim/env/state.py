import numpy as np

from .observation import Observation


class State:
    """A state in the network attack simulator environment.

    This class mainly acts as a wrapper to the Environment Network class and provides
    functionality for getting the state in friendly format (i.e. as numpy array).
    """

    def __init__(self, network):
        """
        Arguments
        ---------
        network : Network
            the network object for the environment
        """
        self.network = network
        self._tensor = self._tensorize()

    def _tensorize(self):
        """Create a numpy tensor version of state """
        h0 = self.network.hosts[(1, 0)]
        tensor = np.zeros((len(self.network.hosts), h0.state_size), dtype=np.float32)
        for host_num, host in enumerate(self.network.hosts.values()):
            tensor[host_num] = host.numpy()
        return tensor

    def reset(self):
        self._tensor = self._tensorize()

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

        for host_addr in self.network.hosts:
            h_idx, host = self.get_host_and_idx(host_addr)
            if self.network.subnet_public(host_addr[0]):
                host_obs = host.observe(reachable=True,
                                        discovered=True)
                obs.update_from_host(h_idx, host_obs)
        return obs

    def get_observation(self, action, action_obs, fully_obs):
        """Get observation given last action

        Arguments
        ---------
        action : Action
            last action performed
        action_obs : ActionObservation
            observation from performing action
        fully_obs : bool
            whether problem is fully observable or not

        Returns
        -------
        Observation
            an observation object
        """
        obs = Observation(self.shape())
        if fully_obs:
            obs.from_state(self)
            return obs

        if not action_obs.success:
            # action failed so no observation
            return obs

        target_idx, target_host = self.get_host_and_idx(action.target)
        if not action.is_scan():
            # exploit action, so get all observations for host
            target_obs = target_host.observe(compromised=True,
                                             reachable=True,
                                             discovered=True,
                                             value=True,
                                             services=True,
                                             os=True)
        elif action.is_service_scan():
            target_obs = target_host.observe(reachable=True,
                                             discovered=True,
                                             services=True)
        elif action.is_os_scan():
            target_obs = target_host.observe(reachable=True,
                                             discovered=True,
                                             os=True)
        elif action.is_subnet_scan():
            for host_addr, discovered in action_obs.discovered.items():
                if discovered:
                    d_idx, d_host = self.get_host_and_idx(host_addr)
                    d_obs = d_host.observe(reachable=True,
                                           discovered=True)
                    obs.update_from_host(d_idx, d_obs)
            target_obs = target_host.observe(compromised=True,
                                             reachable=True,
                                             discovered=True)
        else:
            raise NotImplementedError(f"Action {action} not implemented")
        obs.update_from_host(target_idx, target_obs)
        return obs

    def update(self, host_addr):
        """Updates the state tensor with latest state of given host

        Arguments
        ---------
        host_addr : (int, int)
            the address of host to update state for
        """
        host_idx, host = self.get_host_and_idx(host_addr)
        self._tensor[host_idx] = host.numpy()

    def get_host_and_idx(self, host_addr):
        for host_idx, host in enumerate(self.network.hosts.values()):
            if host.address == host_addr:
                return host_idx, host
        raise AssertionError(f"Host Address '{host_addr}' invalid. Bad format or not in network.")

    def flat_size(self):
        """Return the size of state in terms of the flattened state.

        Returns
        -------
        int
            size of flattened state
        """
        return self.network.state_size()

    def flat_shape(self):
        """Return the shape of state in terms of the flattened state.

        Returns
        -------
        (int, int)
            shape of flattened state
        """
        return self.numpy_flat().shape

    def shape(self):
        """Return the shape of state in terms of the unflattened (tensor) state.

        Returns
        -------
        (int, int)
            shape of flattened state
        """
        return self._tensor.shape

    def numpy_flat(self):
        """Returns state as a 1D numpy array.

        Returns
        -------
        ndarray
            ID numpy array representation of state
        """
        return self._tensor.flatten()

    def numpy_2D(self):
        """Returns state as a 2D numpy array, with each column being the state of a host.

        Returns
        -------
        ndarray
            numpy array representation of state
        """
        return self._tensor

    @staticmethod
    def generate_initial_state(network):
        """Create the initial state of the environment.

        Arguments
        ---------
        network : Network
            the environment network object

        Returns
        -------
        State
            the initial state of the environment
        """
        initial_state = State(network)
        return initial_state

    def __str__(self):
        return str(self.network)
