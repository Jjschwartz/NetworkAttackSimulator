import numpy as np


class Observation:
    """An observation for the network attack simulator.

    This is returned by the step function

    --------

    An observation contains for each host
    1. compromised : 0 or 1
        whether the action resulted in the host becoming compromised (1) or not (0)
    2. reachable : 0 or 1
        whether the action resulted in the host becoming reachable (1) or not (0)
    3. discovered : 0 or 1
        whether the action resulted in the host becoming discovered (1) or not (0)
    4. value : float
        observed value of host
    5. for each service :
        i. present : 0 or 1
            whether the service was observed to be present or not
    6. for each os :
        i. present : 0 or 1
            whether the os was observed to be present or not
    """

    def __init__(self, state_shape):
        """
        Arguments
        ---------
        state_shape : (int, int)
            2D shape of the state (i.e. num_hosts, host_vector_size)
        """
        self.tensor = np.zeros(state_shape, dtype=np.float32)

    def from_state(self, state):
        """Copy observation from state (i.e. fully observable observation)

        Arguments
        ---------
        state : State
            the state object to copy from
        """
        self.tensor[:] = state.tensor

    def update_from_host(self, host_idx, host_obs_vector):
        """Update the observation using given host observation vector """
        self.tensor[host_idx][:] = host_obs_vector

    def numpy_flat(self):
        """Returns observation as a 1D numpy array.

        Returns
        -------
        ndarray
            1D numpy array representation of observation
        """
        return self.tensor.flatten()

    def numpy_2D(self):
        """Returns observation as a 2D numpy array, with each column being
        the observation of a host.

        Returns
        -------
        ndarray
            numpy array representation of observation
        """
        return self.tensor
