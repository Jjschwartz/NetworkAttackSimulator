import numpy as np


class Machine(object):
    """
    A single machine in the network.

    Properties:
    - address : address of machine as a (subnet, id) tuple
    - services : a dictionary of services and bools indicating if they are
        active on this machine.
    - value : the reward gained from exploiting machine
    """

    def __init__(self, address, services, value=0.0):
        """
        Initialize a machine.

        Arguments:
            tuple address : address of machine as (subnet, id)
            list services: a list of ordered bools indicating which services
                are present/absent
            float value : value of the machine
        """
        self.address = address
        self._services = services
        self._value = value

    def perform_action(self, action):
        """
        Attempt to perform given action against this machine

        Arguments:
            Action exploit : the exploit Action

        Returns:
            bool success : True if exploit/scan was successful, False otherwise
            float value : value gained from action. Is the value of the
                machine if successfuly exploited, otherwise 0
            list services : the list of services identified by action.
        """
        if action.is_scan():
            return True, 0, self._services.copy()
        elif self._services[action.service]:
            # service is present so exploit is successful
            return True, self._value, self._services.copy()
        else:
            # service absent, exploit fails
            return False, 0, np.asarray([])

    def __str__(self):
        return ("Machine: " + str(self.address) + ", " + str(self._value))

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) != type(other):
            return False
        if self.address != other.address:
            return False
        if self._value != other._value:
            return False
        return np.equal(self._services, other._services).all()
