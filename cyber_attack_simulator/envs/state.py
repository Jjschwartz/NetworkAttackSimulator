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
            ServiceState state : state of service
        """
        return self._obs[target][SERVICE_INFO][service]

    def update_service(self, target, service, present):
        """
        Update a service on the specified target machines

        Arguments:
            (int, int) target : the target machine address
            int service : the service number
            bool present : whether service is present or absent
        """
        if present:
            self._obs[target][SERVICE_INFO][service] = PRESENT
        else:
            self._obs[target][SERVICE_INFO][service] = ABSENT

    def set_compromised(self, target):
        """
        Set the target machine state as compromised

        Arguments:
            (int, int) target : the target machine address
        """
        self._obs[target][COMPROMISED] = True

    def set_reachable(self, target):
        """
        Set the target machine state as reachable

        Arguments:
            (int, int) target : the target machine address
        """
        self._obs[target][REACHABLE] = True

    def copy(self):
        """
        Return a copy of the state

        Returns:
            State copy : a copy of this state
        """
        obs_copy = OrderedDict()
        for k, v in self._obs.items():
            service_info_copy = np.copy(v[SERVICE_INFO])
            obs_copy[k] = [v[COMPROMISED], v[REACHABLE], service_info_copy]
        return State(obs_copy)

    def __str__(self):
        return str(self._obs)

    def __hash__(self):
        # We can assume address space doesn't change so only need to hash values
        # Also using an OrderedDict so order is stable
        hash_list = []
        for v in self._obs.values():
            hash_list.append(v[COMPROMISED])
            hash_list.append(v[REACHABLE])
            hash_list.append(hash(v[SERVICE_INFO].tostring()))
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
            if (v[COMPROMISED] != other_v[COMPROMISED] or v[REACHABLE] != other_v[REACHABLE]):
                return False
            if not np.array_equal(v[SERVICE_INFO], other_v[SERVICE_INFO]):
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
            service_info = np.full(num_services, UNKNOWN)
            compromised = False
            reachable = False
            if network.subnet_exposed(m[0]):
                reachable = True
            obs[m] = [compromised, reachable, service_info]
        initial_state = State(obs)
        return initial_state
