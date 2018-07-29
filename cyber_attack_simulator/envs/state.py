import numpy as np


class State(object):
    """
    A state in the cyber attack simulator environmentself.

    Properties:
        - dict obs : a dictionary with the address of each machine as the
            key and values being the observed state of the machine

    State of a machine:
        - Defined by :
            1. service_info : list of ServiceState, for each service
            2. compromised : True/False
            3. sensitive : True/False (whether machine has sensitive info)
            4. reachable : True/False (whether machine is currently reachable)

    Main methods:
        - reachable : whether machine is reachable
        - compromised : whether machine is compromised
        - next_state : get next state given an action
    """

    def __init__(self, obs):
        self._obs = obs

    def get_machines(self):
        """
        Get all machines in state

        Returns:
            list machines : list of machine addresses in state
        """
        return sorted(list(self._obs.keys()))

    def reachable(self, target):
        """
        Checks if a given target machine is reachable

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool reachable : True if reachable
        """
        return self._obs[target]["reachable"]

    def compromised(self, target):
        """
        Checks if a given target machine is compromised

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool compromised : True if compromised
        """
        return self._obs[target]["compromised"]

    def sensitive(self, target):
        """
        Checks if a given target machine has sensitive documents (i.e. a
        reward)

        Arguments:
            (int, int) target : the machine address

        Returns:
            bool sensitive : True if machine has sensitive docs
        """
        return self._obs[target]["sensitive"]

    def service_state(self, target, service):
        """
        Get the service state for a given target machine and service

        Arguments:
            (int, int) target : the target machine address
            int service : the service number

        Returns
            ServiceState state : state of service
        """
        return self._obs[target]["service_info"][service]

    def update_service(self, target, service, new_state):
        """
        Update a service on the specified target machines

        Arguments:
            (int, int) target : the target machine address
            int service : the service number
            ServiceState new_state : new service state
        """
        self._obs[target]["service_info"][service] = new_state

    def set_compromised(self, target):
        self._obs[target]["compromised"] = True

    def set_reachable(self, target):
        self._obs[target]["reachable"] = True

    def __str__(self):
        return str(self._obs)

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        if len(self._obs) != len(other._obs):
            return False
        for m, v in self._obs.items():
            if m not in other._obs:
                return False

            other_v = other._obs[m]
            if (v["compromised"] != other_v["compromised"]
                or v["reachable"] != other_v["reachable"]
                    or v["sensitive"] != other_v["sensitive"]):
                return False
            if not np.array_equal(v["service_info"], other_v["service_info"]):
                return False
        return True
