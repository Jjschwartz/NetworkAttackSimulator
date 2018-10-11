import numpy as np
from collections import OrderedDict

# index of each state variable in state list
COMPROMISED = 0
REACHABLE = 1
SERVICE_INFO = 2

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
        - next_state : get next state given an action
    """

    def __init__(self, obs):
        """
        Initialize new state object.

        Arguments:
            OrderedDict obs : a dictionary with the address of each machine as the
                key and values being a list of values of state variables
        """
        self._obs = obs

    def reachable(self, target):
        """
        Checks if a given target machine is reachable

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool reachable : True if reachable
        """
        return self._obs[target][REACHABLE]

    def compromised(self, target):
        """
        Checks if a given target machine is compromised

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool compromised : True if compromised
        """
        return self._obs[target][COMPROMISED]

    def service_state(self, target, service):
        """
        Get the service state for a given target machine and service

        Arguments:
            (int, int) target : the target machine address
            int service : the service number

        Returns
            int service_state : state of service
        """
        return self._obs[target][SERVICE_INFO + service]

    def update_service(self, target, service, present):
        """
        Update a service on the specified target machines

        Arguments:
            (int, int) target : the target machine address
            int service : the service number
            bool present : whether service is present or absent
        """
        if present:
            self._obs[target][SERVICE_INFO + service] = PRESENT
        else:
            self._obs[target][SERVICE_INFO + service] = ABSENT

    def set_compromised(self, target):
        """
        Set the target machine state as compromised

        Arguments:
            (int, int) target : the target machine address
        """
        self._obs[target][COMPROMISED] = 1

    def set_reachable(self, target):
        """
        Set the target machine state as reachable

        Arguments:
            (int, int) target : the target machine address
        """
        self._obs[target][REACHABLE] = 1

    def copy(self):
        """
        Return a copy of the state

        Returns:
            State copy : a copy of this state
        """
        obs_copy = OrderedDict()
        for m, v in self._obs.items():
            machine_state_copy = np.copy(v)
            obs_copy[m] = machine_state_copy
        return State(obs_copy)

    def flatten(self):
        """
        Return a copy of the state as a 1D numpy array, with the state of each
        machine in state stacked on top of each other in order of address.

        Returns:
            ndarray flattened : state as a 1D numpy array
        """
        return np.concatenate(list(self._obs.values()))

    def get_state_size(self):
        """
        Return the size of state in terms of the flattened state.

        Returns:
            int state_size : size of flattened state
        """
        return self.flatten().shape[0]

    def __str__(self):
        return str(self._obs)

    def __hash__(self):
        # We can assume address space doesn't change so only need to hash values
        # Also using an OrderedDict so order is stable
        hash_list = []
        for v in self._obs.values():
            hash_list.append(hash(v.tostring()))
        return hash(tuple(hash_list))

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        if len(self._obs) != len(other._obs):
            return False
        for m, v in self._obs.items():
            other_v = other._obs.get(m)
            if other_v is None:
                return False
            if not np.array_equal(v, other_v):
                return False
        return True

    @staticmethod
    def generate_initial_state(network, num_services):
        """
        Generate the initial state of the environment. Initial state is where no machines have been
        compromised, only DMZ subnets are reachable and no information about services has been
        gained

        Arguments:
            Network network : the environment network object
            int num_services : number of services running in environment

        Returns:
            State initial_state : the initial state of the environment
        """
        obs = OrderedDict()
        for m in network.get_address_space():
            # one vector for each machine, with compromised and reachable as first two values
            # and service knowledge state for each service as other values
            machine_state = np.full(num_services + SERVICE_INFO, UNKNOWN)
            machine_state[COMPROMISED] = 0
            machine_state[REACHABLE] = 0
            if network.subnet_exposed(m[0]):
                machine_state[REACHABLE] = 1
            obs[m] = machine_state
        initial_state = State(obs)
        return initial_state
