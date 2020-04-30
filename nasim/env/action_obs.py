

class ActionObservation:

    def __init__(self, success, value=0.0, services=None, os=None,
                 discovered=None, connection_error=False):
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
        connection_error : bool
            True if action failed due to connection error (e.g. could
            not reach target)
        """
        self.success = success
        self.value = value
        self.services = {} if services is None else services
        self.os = {} if os is None else os
        self.discovered = {} if discovered is None else discovered
        self.connection_error = connection_error

    def __str__(self):
        output = ["ActionObservation:"],
        for k, v in self.info().items():
            output.append(f"  {k}={v}")
        return "\n".join(output)

    def info(self):
        return dict(
            success=self.success,
            value=self.value,
            services=self.services,
            os=self.os,
            discovered=self.discovered,
            connection_error=self.connection_error
        )
