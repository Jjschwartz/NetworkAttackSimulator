import numpy as np
from collections import OrderedDict


# index of each state variable in state list
COMPROMISED = 0
REACHABLE = 1
SERVICE_INFO = 2


class State(object):
    """
    A state in the cyber attack simulator environment.

    Properties:
        - OrderedDict obs : a dictionary with the address of each machine as the
            key and values being a list of values of state variables

    State variables:
        - Defined by :
            1. service_info : list of ServiceState, for each service
            2. compromised : True/False
            3. reachable : True/False (whether machine is currently reachable)

    Main methods:
        - reachable : whether machine is reachable
        - compromised : whether machine is compromised
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

    def update_service(self, target, service, new_service_state):
        """
        Update a service on the specified target machines

        Arguments:
            (int, int) target : the target machine address
            int service : the service number
            ServiceState new_service_state : new service state
        """
        self._obs[target][SERVICE_INFO][service] = new_service_state

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
