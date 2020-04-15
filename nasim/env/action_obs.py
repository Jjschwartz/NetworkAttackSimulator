

class ActionObservation:

    def __init__(self, success, value=0.0, services=None, os=None,
                 discovered=None):
        """
        Arguments
        ---------
        success : bool
            True if exploit/scan was successful, False otherwise
        value : float
            value gained from action. Is the value of the host if successfuly
            exploited, otherwise 0
        services : dict
            services identified by action.
        os : dict
            OS identified by action
        discovered : dict
            host addresses discovered by action
        """
        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.discovered = {} if discovered is None else discovered

    def __str__(self):
        output = ["ActionObservation:",
                  f"  Success={self.success}",
                  f"  Value={self.value}",
                  f"  Services={self.services}",
                  f"  OS={self.os}",
                  f"  Discovered={self.discovered}"]
        return "\n".join(output)

    def info(self):
        return dict(
            success=self.success,
            value=self.value,
            services=self.services,
            os=self.os,
            discovered=self.discovered
        )
