import numpy as np


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

    def update(self, host_addr):
        """Updates the state tensor with latest state of given host

        Arguments
        ---------
        host_addr : (int, int)
            the address of host to update state for
        """
        for host_num, host in enumerate(self.network.hosts.values()):
            if host.address == host_addr:
                self._tensor[host_num] = host.numpy()
                break

    def get_state_size(self):
        """Return the size of state in terms of the flattened state.

        Returns
        -------
        int
            size of flattened state
        """
        return self.network.state_size()

    def get_state_shape(self):
        """Return the shape of state in terms of the flattened state.

        Returns
        -------
        (int, int)
            shape of flattened state
        """
        return self.numpy().shape

    def numpy(self):
        """Returns state as a 1D numpy array.

        Returns
        -------
        ndarray
            numpy array representation of state
        """
        return self._tensor.flatten()

    @staticmethod
    def generate_initial_mdp_state(network):
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
