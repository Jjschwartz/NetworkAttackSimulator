import numpy as np

from .host_vector import HostVector
from .network_tensor import NetworkTensor


class Observation:
    """An observation for NASim.

    This is returned by the step function after every action is performed.
    Each observation is a 2D tensor with a row for each host and an additional
    row containing auxiliary observations.
    Each host row is a host_vector.
    For details see :py:class: nasim.env.host_vector.HostVector

    The auxiliary row is the final row in the observation tensor and has the
    following features (in order):
    1. Action success - True/False
        whether the action succeeded or failed
    2. Connection error - True/False
        whether there was a connection error or not

    Since the number of features in the auxiliary row is less than the number
    of features in each host row, the remainder of the row is all zeros.

    1. compromised : 0 or 1
        whether the action resulted in the host becoming compromised (1)
        or not (0)
    2. reachable : 0 or 1
        whether the action resulted in the host becoming reachable (1)
        or not (0)
    3. discovered : 0 or 1
        whether the action resulted in the host becoming discovered (1)
        or not (0)
    4. value : float
        observed value of host
    5. for each service :
        i. present : 0 or 1
            whether the service was observed to be present or not
    6. for each os :
        i. present : 0 or 1
            whether the os was observed to be present or not
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

    def from_state(self, state):
        self.tensor[:self.aux_row] = state.network_state.tensor

    def from_action_obs(self, action_obs):
        self.tensor[self.aux_row][self._success_idx] = int(action_obs.success)
        con_err = int(action_obs.connection_error)
        self.tensor[self.aux_row][self._conn_error_idx] = con_err

    def from_state_and_action(self, state, action_obs):
        self.from_state(state)
        self.from_action_obs(action_obs)

    def update_from_host(self, host_idx, host_obs_vector):
        """Update the observation using given host observation vector """
        self.tensor[host_idx][:] = host_obs_vector

    @property
    def success(self):
        return bool(self.tensor[self.aux_row][self._success_idx])

    @property
    def connection_error(self):
        return bool(self.tensor[self.aux_row][self._conn_error_idx])

    def numpy_flat(self):
        return self.tensor.flatten()

    def numpy_2D(self):
        return self.tensor

    def get_readable(self):
        host_obs = []
        for host_idx in NetworkTensor.host_to_idx_map.values():
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
