from .host_vector import HostVector


class Host:
    """A single host in the network.

    Properties
    ----------
    address : (int, int)
        address of host as a (subnet, id) tuple
    os : str
        the OS of the host
    services : dict
        a dictionary of services and bools indicating if they are
        active on this host.
    value : float
        the reward gained from exploiting host
    compromised : bool, optional
        whether host has been compromised or not
    reachable : bool
        whether host is reachable by attacker or not
    """

    def __init__(self, address, os, services, value=0.0, compromised=False, reachable=False):
        """
        Arguments
        ---------
        address : (int, int)
            address of host as (subnet, id)
        os : dict
            A os_name: bool dictionary indicating which OS the host is runinng
        services: dict
            a (service_name, bool) dictionary indicating which services
            are present/absent
        value : float, optional
            value of the host (default=0.0)
        compromised : bool, optional
            whether host has been compromised or not (default=False)
        reachable : bool, optional
            whether host is reachable by attacker or not (default=False)
        """
        self.address = address
        self.os = os
        self.services = services
        self.value = value
        self._compromised = compromised
        self._reachable = reachable
        self._vector = HostVector(self)

    @property
    def compromised(self):
        return self._compromised

    @compromised.setter
    def compromised(self, val):
        self._compromised = val
        self._vector.set_compromised(val)

    @property
    def reachable(self):
        return self._reachable

    @reachable.setter
    def reachable(self, val):
        self._reachable = val
        self._vector.set_reachable(val)

    @property
    def state_size(self):
        return self._vector.state_size

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
        os : dict
            the dict of OS identified by action
        """
        if action.is_service_scan():
            return True, 0, self.services, {}
        if action.is_os_scan():
            return True, 0, {}, self.os
        # action is an exploit
        if self.services[action.service] and (action.os is None or self.os[action.os]):
            # service and os is present so exploit is successful
            value = 0
            if not self.compromised:
                # to ensure a machine is not rewarded twice
                value = self.value
                self.compromised = True
            return True, value, self.services, self.os
        # service absent, exploit fails
        return False, 0, {}, {}

    def service_present(self, service):
        """Returns whether host is running a service or not.

        Arguments
        ---------
        service : str
            name of service

        Returns
        -------
        bool
            True if host is running service
        """
        return self.services[service]

    def is_running_os(self, os):
        """Returns whether host is running a given OS or not.

        Arguments
        ---------
        os : str
            name of os

        Returns
        -------
        bool
            True if host is running OS
        """
        return self.os[os]

    def get_value(self):
        """Returns value of this host

        Returns
        -------
        float
            value of this host
        """
        return self.value

    def numpy(self):
        """Return numpy array representation of host

        Returns
        -------
        ndarray
            numpy representation of host
        """
        return self._vector.vector

    def update_vector(self):
        self._vector = HostVector(self)

    def __str__(self):
        output = ["Host: {"]
        output.append(f"\taddress: {self.address}")
        output.append(f"\tcompromised: {self._compromised}")
        output.append(f"\treachable: {self._reachable}")
        output.append(f"\tvalue: {self.value}")
        output.append("\tservices: {")
        for name, val in self.services.items():
            output.append(f"\t\t{name}: {val}")
        output.append("\t}")
        output.append("\tOS: {")
        for os_name, val in self.os.items():
            output.append(f"\t\t{os_name}: {val}")
        output.append("\t}")
        output.append("}")
        return "\n".join(output)

    def __repr__(self):
        return f"Host: {self.address}"

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
        return self.services == other.services and self.os == other.os
