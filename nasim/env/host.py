

class Host:
    """A single host in the network.

    Properties:
    - address : address of host as a (subnet, id) tuple
    - services : a dictionary of services and bools indicating if they are
        active on this host.
    - value : the reward gained from exploiting host
    """

    def __init__(self, address, services, value=0.0):
        """
        Arguments
        ---------
        address : (int, int)
            address of host as (subnet, id)
        services: dict
            a (service_name, bool) dictionary indicating which services
            are present/absent
        value : float, optional
            value of the host (default=0.0)
        """
        self.address = address
        self.services = services
        self.value = value

    def perform_action(self, action):
        """Perform given action against this host

        Arguments
        ---------
        action : Action
            the action to perform

        Returns
        -------
        success : bool
            True if exploit/scan was successful, False otherwise
        value : float
            value gained from action. Is the value of the host if successfuly
            exploited, otherwise 0
        services : dict
            the dict of services identified by action.
        """
        if action.is_scan():
            return True, 0, self.services.copy()
        elif self.services[action.service]:
            # service is present so exploit is successful
            return True, self.value, self.services.copy()
        # service absent, exploit fails
        return False, 0, {}

    def get_value(self):
        """Returns value of this host

        Returns
        -------
        float
            value of this host
        """
        return self.value

    def __str__(self):
        output = ["Host: {"]
        output.append(f"\taddress: {self.address}")
        output.append(f"\tvalue: {self.value}")
        output.append("\tservices: {")
        if type(self.services) is list:
            for i, val in enumerate(self.services):
                output.append(f"\t\t{i}: {val}")
        else:
            for name, val in self.services.items():
                output.append(f"\t\t{name}: {val}")
        output.append("\t}")
        output.append("}")
        return "\n".join(output)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Host):
            return False
        if self.address != other.address:
            return False
        if self.value != other.value:
            return False
        return self.services == other.services
