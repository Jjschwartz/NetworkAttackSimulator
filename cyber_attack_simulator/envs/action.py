
EXPLOIT_COST = 10.0
SCAN_COST = 10.0


class Action(object):
    """
    An action in the environment.

    An action is defined by 3 properties:
        1. type: either exploit or scan
        2. service: which service the exploit is targeting, where services are
            identified by a number from 0 upto num_exploits in environment.
            This is None for scan type actions.
        3. target: the machine to launch action against. The machine is defined
            by the (subnet, machine_id) tuple
    """

    def __init__(self, target, type="scan", service=None):
        """
        Initialize a new action
        """
        self.target = target
        self.type = type
        self.service = service

    def is_scan(self):
        return self.type == "scan"

    @property
    def cost(self):
        """
        The cost of performing the given action
        """
        return SCAN_COST if self.is_scan() else EXPLOIT_COST

    def __str__(self):
        return ("Action: target=" + str(self.target[0]) + "." +
                str(self.target[1]) + ", type=" + self.type + ", service=" +
                str(self.service))

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        elif self.target != other.target or self.type != other.type:
            return False
        else:
            return self.service == other.service
