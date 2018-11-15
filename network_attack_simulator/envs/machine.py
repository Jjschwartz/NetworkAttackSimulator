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
            list or dict services: an ordered list of bools or a (service_name, bool) dictionary
                indicating which services are present/absent
            float value : value of the machine
        """
        self.address = address
        if type(services) == list:
            self._services = self._convert_services_list_to_map(services)
        else:
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
            dict services : the dict of services identified by action.
        """
        if action.is_scan():
            return True, 0, self._services.copy()
        elif self._services[action.service]:
            # service is present so exploit is successful
            return True, self._value, self._services.copy()
        else:
            # service absent, exploit fails
            return False, 0, {}

    def get_value(self):
        """
        Returns value of this machine

        Returns:
            float value : value of this machine
        """
        return self._value

    def _convert_services_list_to_map(self, services):
        service_map = {}
        for srv, val in enumerate(services):
            service_map[srv] = val
        return service_map

    def __str__(self):
        output = "Machine: {\n"
        output += "\taddress: {}\n".format(self.address)
        output += "\tvalue: {}\n".format(self._value)
        output += "\tservices: {\n"
        if type(self._services) is list:
            for i, val in enumerate(self._services):
                output += "\t\t{}: {}\n".format(i, val)
        else:
            for name, val in self._services.items():
                output += "\t\t{}: {}\n".format(name, val)
        output += "\t}\n"
        output += "}"
        return output

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
