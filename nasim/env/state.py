from .observation import Observation
from .network_tensor import NetworkTensor


class State:
    """A state in the NASim Environment.

    This class mainly acts as a wrapper to the Environment Network class and
    provides functionality for getting the state in friendly format
    (i.e. as numpy array).

    ...
    Attributes
    ----------
    network_state : NetworkTensor
        tensor representation of the state of network
    """

    def __init__(self, network_state):
        """
        Parameters
        ----------
        network_state : NetworkTensor
            the current network state
        """
        self.network_state = network_state

    @classmethod
    def tensorize(cls, network):
        """Create a new state from a Network object """
        network_tensor = NetworkTensor.tensorize(network)
        return State(network_tensor)

    def copy(self):
        network_state_copy = self.network_state.copy()
        return State(network_state_copy)

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

        for host_addr, host in self.network_state.hosts:
            if not host.reachable:
                continue
            host_obs = host.observe(reachable=True,
                                    discovered=True)
            host_idx = self.network_state.get_host_idx(host_addr)
            obs.update_from_host(host_idx, host_obs)
        return obs

    def get_observation(self, action, action_obs, fully_obs):
        """Get observation given last action

        Parameters
        ----------
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
        obs.from_action_obs(action_obs)
        if fully_obs:
            obs.from_state(self)
            return obs

        if not action_obs.success:
            # action failed so no observation
            return obs

        t_idx, t_host = self.network_state.get_host_and_idx(action.target)
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
            for host_addr, discovered in action_obs.discovered.items():
                if not discovered:
                    continue
                d_idx, d_host = self.network_state.get_host_and_idx(host_addr)
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

    def flat_size(self):
        return self.network_state.state_size()

    def flat_shape(self):
        return self.numpy_flat().shape

    def shape(self):
        return self.numpy_2D().shape

    def numpy_flat(self):
        return self.numpy_2D().flatten()

    def numpy_2D(self):
        return self.network_state.tensor

    def get_readable(self):
        host_obs = self.network_state.get_readable()
        return host_obs

    @classmethod
    def generate_initial_state(cls, network):
        state = cls.tensorize(network)
        return network.reset(state)

    @classmethod
    def generate_random_initial_state(cls, network):
        network_tensor = NetworkTensor.tensorize_random(network)
        state = State(network_tensor)
        # ensure host state set correctly
        return network.reset(state)

    def __str__(self):
        return str(self.network_state)

    def __hash__(self):
        return hash(self.network_state)

    def __eq__(self, other):
        return self.network_state == other.network_state
