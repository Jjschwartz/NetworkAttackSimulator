import numpy as np


class Machine(object):
    """
    A single machine in the network.

    Properties:
    - subnet_id : subnetwork machine belongs to
    - id : machine number withing subnet
    - services : a dictionary of services and bools indicating if they are
        active on this machine.

    The combination of subnet and id are enough to uniquely identify a machine
    within a network (similar to an IP address). E.g. machine with subnet=1 and
    id=3, would have a network address 1.3.
    """

    def __init__(self, subnet_id, id, services, value=0.0):
        """
        Initialize a machine.

        Arguments:
        int subnet_id : the subnetwork this machine belongs to
        int id : machine number withing subnet
        list services: a list of ordered bools indicating which services are
            present/absent
        float value : value of the machine
        """
        self._subnet_id = subnet_id
        self._id = id
        self._services = services
        self._value = value

    def perform_action(self, action):
        """
        Attempt to perform given action against this machine

        Arguments:
        Action exploit : the exploit Action

        Returns:
        bool success : True if exploit/scan was successful, False otherwise
        float value : value gained from action. Is the value of the machine if
            successfuly exploited, otherwise 0 if unsuccessful or scan.
        list services : the list of services identified by action. This is
            self.services if exploit was successful or scan, otherwise an empty
            list
        """
        if action.is_scan():
            return True, 0, self._services.copy()
        elif self._services[action.service]:
            # service is present so exploit is successful
            return True, self._value, self._services.copy()
        else:
            # service absent, exploit fails
            return False, 0, np.asarray([])

    @property
    def address(self):
        """
        The unique address of this machine, defined by tuple (subnet, id)
        """
        return (self._subnet_id, self._id)

    def __str__(self):
        return ("Machine: " + str(self._subnet_id) + "." + str(self._id)
                + ", " + str(self._value))

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        elif self._subnet_id != other._subnet_id or self._id != other._id:
            return False
        elif self._value != other._value:
            return False
        else:
            return (self._services == other._services).all()
