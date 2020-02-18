import numpy as np
from copy import deepcopy

# keys
COMPROMISED_KEY = "compromised"
REACHABLE_KEY = "reachable"

# positions
COMPROMISED = 0
REACHABLE = 1

# values for possible service knowledge states
UNKNOWN = 0     # service may or may not be running on host
PRESENT = 1     # service is running on the host
ABSENT = 2      # service not running on the host


class State:
    """A state in the network attack simulator environment.

    Properties:
    - OrderedDict obs : a dictionary with the address of each host as the
        key and values being a list of values of state variables

    State variables:
    - Defined by :
        1. compromised : True/False
        2. reachable : True/False (whether host is currently reachable)
        3. service_info : list of knowledge for each service (UNKNOWN, PRESENT, ABSENT)

    Main methods:
    - reachable : whether a given host is reachable
    - compromised : whether a given host is compromised
    - service_state : get the knowledge state for a given service and host
    """
    # class variable that maps host address to vector position
    host_indices = {}
    # class variable that maps service ID to vector position
    service_indices = {}
    # class variable
    m_state_size = 0

    def __init__(self, obs):
        """
        Arguments
        ---------
        obs : dict
            dict with the host address key and dict of host state as values
        """
        self._obs = obs
        self._vector = self._vectorize()

    def _vectorize(self):
        """Convert observation map into a 1D numpy array.
        Vector contains compromised, reachable, srv1, .. , srvN in order of host index
        """
        M = len(self._obs)
        vector = np.zeros(M * State.m_state_size, dtype=np.int8)
        for m, m_state in self._obs.items():
            m_i = self._get_host_index(m)
            for k, v in m_state.items():
                # int(v) since compromised and reachable are bools in map
                if k == COMPROMISED_KEY:
                    vector[m_i + COMPROMISED] = int(v)
                elif k == REACHABLE_KEY:
                    vector[m_i + REACHABLE] = int(v)
                else:
                    srv_i = self._get_service_index(k)
                    vector[m_i + srv_i] = v
        return vector

    def _get_host_index(self, m):
        """Get the index of a host in state vector"""
        return State.host_indices[m] * State.m_state_size

    def _get_service_index(self, srv):
        """Get the index of a service in state vector"""
        return State.service_indices[srv] + REACHABLE + 1

    def reachable(self, host_addr):
        """Checks if a given target host is reachable

        Arguments
        ---------
        host_addr : (int, int)
            the host address

        Returns
        -------
        bool
            True if reachable
        """
        return self._obs[host_addr][REACHABLE_KEY]

    def compromised(self, host_addr):
        """Checks if a given target host is compromised

        Arguments
        ---------
        target : (int, int)
            the host address

        Returns
        -------
        bool
            True if compromised
        """
        return self._obs[host_addr][COMPROMISED_KEY]

    def service_state(self, host_addr, service):
        """Get the service state for a given target host and service

        Arguments
        ---------
        host_addr : (int, int)
            the target host address
        service : int
            the service number

        Returns
        -------
        int
            state of service
        """
        return self._obs[host_addr][service]

    def update_service(self, host_addr, service, present):
        """Update a service on the specified target hosts

        Arguments
        ---------
        host_addr : (int, int)
            the target host address
        service : int or str
            the service ID
        present : bool
            whether service is present or absent
        """
        t_index = self._get_host_index(host_addr)
        s_index = self._get_service_index(service)
        if present:
            self._obs[host_addr][service] = PRESENT
            self._vector[t_index + s_index] = PRESENT
        else:
            self._obs[host_addr][service] = ABSENT
            self._vector[t_index + s_index] = ABSENT

    def set_compromised(self, host_addr):
        """Set the target host state as compromised

        Arguments
        ---------
        host_addr : (int, int)
            the target host address
        """
        self._obs[host_addr][COMPROMISED_KEY] = True
        t_index = self._get_host_index(host_addr)
        self._vector[t_index + COMPROMISED] = int(True)

    def set_reachable(self, host_addr):
        """Set the target host state as reachable

        Arguments
        ---------
        host_addr : (int, int)
            the target host address
        """
        self._obs[host_addr][REACHABLE_KEY] = True
        t_index = self._get_host_index(host_addr)
        self._vector[t_index + REACHABLE] = int(True)

    def copy(self):
        """Return a copy of the state

        Returns
        -------
        State
            a copy of this state
        """
        obs_copy = dict()
        for m, v in self._obs.items():
            obs_copy[m] = deepcopy(v)
        return State(obs_copy)

    def flatten(self):
        """Get a copy of the state as a 1D numpy array.

        The state of each host is stacked on top of each other in order of address.

        Returns
        -------
        ndarray
            state as a 1D numpy array
        """
        return self._vector.copy()

    def get_hashable(self):
        """Return a copy of the state in an efficient hashable form.

        Returns
        -------
        str
            state vector as hashable string
        """
        return self._vector.tostring()

    def get_state_size(self):
        """Return the size of state in terms of the flattened state.

        Returns
        -------
        int
            size of flattened state
        """
        return self._vector.shape[0]

    def __str__(self):
        return str(self._obs)

    def __hash__(self):
        return hash(self._vector.tostring())

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        if len(self._obs) != len(other._obs):
            return False
        return np.array_equal(self._vector, other._vector)

    @staticmethod
    def generate_initial_state(network, services):
        """Generate the initial state of the environment. Initial state is where
        no hosts have been compromised, only public subnets are reachable and no
        information about services has been gained

        Arguments
        ---------
        network : Network
            the environment network object
        services : list
            list of services

        Returns
        -------
        State
            the initial state of the environment
        """
        obs = dict()

        for i, srv in enumerate(services):
            State.service_indices[srv] = i

        State.m_state_size = len(services) + 2

        for i, m in enumerate(network.address_space):
            # host state is a mapping from service, or status var to value
            # also index which is used when vectorizing state
            State.host_indices[m] = i
            host_state = dict()
            host_state[COMPROMISED_KEY] = False
            if network.subnet_public(m[0]):
                host_state[REACHABLE_KEY] = True
            else:
                host_state[REACHABLE_KEY] = False
            for srv in services:
                host_state[srv] = UNKNOWN
            obs[m] = host_state
        initial_state = State(obs)
        return initial_state
