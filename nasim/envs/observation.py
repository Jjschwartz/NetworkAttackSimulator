import numpy as np

from .host_vector import HostVector


class Observation:
    """An observation for NASim.

    Each observation is a 2D tensor with a row for each host and an additional
    row containing auxiliary observations. Each host row is a host_vector (for
    details see :class:`HostVector`) while the auxiliary
    row contains non-host specific observations (see Notes section).

    ...

    Attributes
    ----------
    obs_shape : (int, int)
        the shape of the observation
    aux_row : int
        the row index for the auxiliary row
    tensor : numpy.ndarray
        2D Numpy array storing the observation

    Notes
    -----
    The auxiliary row is the final row in the observation tensor and has the
    following features (in order):

    1. Action success - True (1) or False (0)
        indicates whether the action succeeded or failed
    2. Connection error - True (1) or False (0)
        indicates whether there was a connection error or not

    Since the number of features in the auxiliary row is less than the number
    of features in each host row, the remainder of the row is all zeros.
    """

    # obs vector positions for auxiliary observations
    _success_idx = 0
    _conn_error_idx = _success_idx+1

    def __init__(self, state_shape):
        """
        Parameters
        ----------
        state_shape : (int, int)
            2D shape of the state (i.e. num_hosts, host_vector_size)
        """
        self.obs_shape = (state_shape[0]+1, state_shape[1])
        self.aux_row = self.obs_shape[0]-1
        self.tensor = np.zeros(self.obs_shape, dtype=np.float32)

    @staticmethod
    def get_space_bounds(scenario):
        value_bounds = scenario.host_value_bounds
        discovery_bounds = scenario.host_discovery_value_bounds
        obs_low = min(0, value_bounds[0], discovery_bounds[0])
        obs_high = max(0,
                       value_bounds[1],
                       discovery_bounds[1],
                       len(scenario.subnets),
                       max(scenario.subnets))
        return (obs_low, obs_high)

    @classmethod
    def from_numpy(cls, o_array, state_shape):
        obs = cls(state_shape)
        if o_array.shape != (state_shape[0]+1, state_shape[1]):
            o_array = o_array.reshape(state_shape[0]+1, state_shape[1])
        obs.tensor = o_array
        return obs

    def from_state(self, state):
        self.tensor[:self.aux_row] = state.tensor

    def from_action_result(self, action_result):
        success = int(action_result.success)
        self.tensor[self.aux_row][self._success_idx] = success
        con_err = int(action_result.connection_error)
        self.tensor[self.aux_row][self._conn_error_idx] = con_err

    def from_state_and_action(self, state, action_result):
        self.from_state(state)
        self.from_action_result(action_result)

    def update_from_host(self, host_idx, host_obs_vector):
        self.tensor[host_idx][:] = host_obs_vector

    @property
    def success(self):
        """Whether the action succeded or not

        Returns
        -------
        bool
            True if the action succeeded, otherwise False
        """
        return bool(self.tensor[self.aux_row][self._success_idx])

    @property
    def connection_error(self):
        """Whether there was a connection error or not

        Returns
        -------
        bool
            True if there was a connection error, otherwise False
        """
        return bool(self.tensor[self.aux_row][self._conn_error_idx])

    def shape_flat(self):
        """Get the flat (1D) shape of the Observation.

        Returns
        -------
        (int, )
            the flattened shape of observation
        """
        return self.numpy_flat().shape

    def shape(self):
        """Get the (2D) shape of the observation

        Returns
        -------
        (int, int)
            the 2D shape of the observation
        """
        return self.obs_shape

    def numpy_flat(self):
        """Get the flattened observation tensor

        Returns
        -------
        numpy.ndarray
            the flattened (1D) observation tenser
        """
        return self.tensor.flatten()

    def numpy(self):
        """Get the observation tensor

        Returns
        -------
        numpy.ndarray
            the (2D) observation tenser
        """
        return self.tensor

    def get_readable(self):
        host_obs = []
        for host_idx in range(self.obs_shape[0]-1):
            host_obs_vec = self.tensor[host_idx]
            readable_dict = HostVector.get_readable(host_obs_vec)
            host_obs.append(readable_dict)

        aux_obs = {
            "Success": self.success,
            "Connection Error": self.connection_error
        }
        return host_obs, aux_obs

    def __str__(self):
        return str(self.tensor)

    def __eq__(self, other):
        return np.array_equal(self.tensor, other.tensor)

    def __hash__(self):
        return hash(str(self.tensor))
