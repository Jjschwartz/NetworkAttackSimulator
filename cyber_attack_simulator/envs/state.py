import numpy as np
from collections import OrderedDict
from copy import deepcopy

# keys
COMPROMISED_KEY = "compromised"
REACHABLE_KEY = "reachable"

# positions
COMPROMISED = 0
REACHABLE = 1

# values for possible service knowledge states
UNKNOWN = 0     # service may or may not be running on machine
PRESENT = 1     # service is running on the machine
ABSENT = 2      # service not running on the machine


class State(object):
    """
    A state in the cyber attack simulator environment.

    Properties:
        - OrderedDict obs : a dictionary with the address of each machine as the
            key and values being a list of values of state variables

    State variables:
        - Defined by :
            1. compromised : True/False
            2. reachable : True/False (whether machine is currently reachable)
            3. service_info : list of knowledge for each service (UNKNOWN, PRESENT, ABSENT)

    Main methods:
        - reachable : whether a given machine is reachable
        - compromised : whether a given machine is compromised
        - service_state : get the knowledge state for a given service and machine
    """
    # class variable that maps machine address to vector position
    machine_indices = {}
    # class variable that maps service ID to vector position
    service_indices = {}
    # class variable
    m_state_size = 0

    def __init__(self, obs):
        """
        Initialize new state object.

        Arguments:
            OrderedDict obs : a dictionary with the address of each machine as the
                key and values a dictionary of machine state
        """
        self._obs = obs
        self._vector = self._vectorize()

    def _vectorize(self):
        """
        Convert observation map into a 1D vector.
        Vector contains compromised, reachable, srv1, .. , srvN in order of machine index

        Returns:
            numpy state_vector : state as a 1D numpy array
        """
        M = len(self._obs)
        vector = np.zeros(M * State.m_state_size, dtype=np.int8)
        for m, m_state in self._obs.items():
            m_i = self._get_machine_index(m)
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

    def _get_machine_index(self, m):
        """
        Get the index of a machine in state vector

        Arguments:
            (int, int) m : the machine address

        Returns:
            int index : the index of the machine
        """
        return State.machine_indices[m] * State.m_state_size

    def _get_service_index(self, srv):
        """
        Get the index of a service in state vector

        Arguments:
            int or str srv : the service ID

        Returns:
            int index : the index of the service
        """
        return State.service_indices[srv] + REACHABLE + 1

    def reachable(self, target):
        """
        Checks if a given target machine is reachable

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool reachable : True if reachable
        """
        return self._obs[target][REACHABLE_KEY]

    def compromised(self, target):
        """
        Checks if a given target machine is compromised

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool compromised : True if compromised
        """
        return self._obs[target][COMPROMISED_KEY]

    def service_state(self, target, service):
        """
        Get the service state for a given target machine and service

        Arguments:
            (int, int) target : the target machine address
            int service : the service number

        Returns
            int service_state : state of service
        """
        return self._obs[target][service]

    def update_service(self, target, service, present):
        """
        Update a service on the specified target machines

        Arguments:
            (int, int) target : the target machine address
            int or str service : the service ID
            bool present : whether service is present or absent
        """
        t_index = self._get_machine_index(target)
        s_index = self._get_service_index(service)
        if present:
            self._obs[target][service] = PRESENT
            self._vector[t_index + s_index] = PRESENT
        else:
            self._obs[target][service] = ABSENT
            self._vector[t_index + s_index] = ABSENT

    def set_compromised(self, target):
        """
        Set the target machine state as compromised

        Arguments:
            (int, int) target : the target machine address
        """
        self._obs[target][COMPROMISED_KEY] = True
        t_index = self._get_machine_index(target)
        self._vector[t_index + COMPROMISED] = int(True)

    def set_reachable(self, target):
        """
        Set the target machine state as reachable

        Arguments:
            (int, int) target : the target machine address
        """
        self._obs[target][REACHABLE_KEY] = True
        t_index = self._get_machine_index(target)
        self._vector[t_index + REACHABLE] = int(True)

    def copy(self):
        """
        Return a copy of the state

        Returns:
            State copy : a copy of this state
        """
        obs_copy = OrderedDict()
        for m, v in self._obs.items():
            obs_copy[m] = deepcopy(v)
        return State(obs_copy)

    def flatten(self):
        """
        Return a copy of the state as a 1D numpy array, with the state of each
        machine in state stacked on top of each other in order of address.

        Returns:
            ndarray flattened : state as a 1D numpy array
        """
        return self._vector.copy()

    def get_hashable(self):
        """
        Return a copy of the state in an efficient hashable form.

        Returns:
            str hashable : state vector as hashable string
        """
        return self._vector.tostring()

    def get_state_size(self):
        """
        Return the size of state in terms of the flattened state.

        Returns:
            int state_size : size of flattened state
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
    def generate_initial_state(network, exploitable_services):
        """
        Generate the initial state of the environment. Initial state is where no machines have been
        compromised, only DMZ subnets are reachable and no information about services has been
        gained

        Arguments:
            Network network : the environment network object
            dict service_map : map of exploitable service IDs to index

        Returns:
            State initial_state : the initial state of the environment
        """
        obs = OrderedDict()

        for srv, i in exploitable_services.items():
            State.service_indices[srv] = i

        State.m_state_size = len(exploitable_services) + 2

        for i, m in enumerate(network.get_address_space()):
            # machine state is a mapping from service, or status var to value
            # also index which is used when vectorizing state
            State.machine_indices[m] = i
            machine_state = OrderedDict()
            machine_state[COMPROMISED_KEY] = False
            if network.subnet_exposed(m[0]):
                machine_state[REACHABLE_KEY] = True
            else:
                machine_state[REACHABLE_KEY] = False
            for srv in exploitable_services.keys():
                machine_state[srv] = UNKNOWN
            obs[m] = machine_state
        initial_state = State(obs)
        return initial_state
