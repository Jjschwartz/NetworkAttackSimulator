import enum


class OneHotBool(enum.IntEnum):
    NONE = 0
    TRUE = 1
    FALSE = 2

    @staticmethod
    def from_bool(b):
        if b:
            return OneHotBool.TRUE
        return OneHotBool.FALSE

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class ServiceState(enum.IntEnum):
    # values for possible service knowledge states
    UNKNOWN = 0     # service may or may not be running on host
    PRESENT = 1     # service is running on the host
    ABSENT = 2      # service not running on the host

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
