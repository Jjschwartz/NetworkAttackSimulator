"""
This module contains functionality for loading network configurations from yaml
files and also for generating configurations based on number of machines and
services in network using standard formula.
"""
import yaml
import numpy as np

# dictionary of valid key names and value types for config file
VALID_CONFIG_KEYS = {"subnets": list,
                     "topology": list,
                     "services": int,
                     "sensitive_machines": list}

# Constants for generating network
USER_SUBNET_SIZE = 5
EXPOSED = 0
SENSITIVE = 1


def generate_config(num_machines, num_services, r_sensitive, r_user):
    """
    Generate the network configuration based on standard formula.

    Arguments:
        int num_machines : number of machines to include in network
            (minimum is 3)
        int num_exploits : number of exploits (and hence services) to use
            in environment (minimum is 1)
        float r_sensitive : reward for sensitive subnet documents
        float r_user : reward for user subnet documents

    Returns:
        dict config : network configuration as dictionary
    """
    assert 0 < num_services
    assert 2 < num_machines
    assert 0 < r_sensitive and 0 < r_user
    config = {}
    subnets = generate_subnets(num_machines)
    config["subnets"] = subnets
    config["topology"] = generate_topology(subnets)
    config["services"] = num_services
    s_machines = generate_sensitive_machines(subnets, r_sensitive, r_user)
    config["sensitive_machines"] = s_machines
    return config


def load_config(file_name):
    """
    Load the network configuration from file

    Raises errors if file invalid.

    Arguments:
        str file_name : path and name of config file

    Returns:
        dict config : network configuration as dictionary
    """
    config = load_yaml_file(file_name)
    try:
        check_config_valid(config)
        return config
    except (KeyError, TypeError, ValueError) as exc:
        raise exc


def generate_subnets(num_machines):
    """
    Generate list of subnet sizes

    Argument:
        int num_machines : number of machines in network

    Returns:
        list subnets : list of number of machines in each subnet
    """
    # exposed (0) and sensitive (1) subnets both contain 1 machine
    subnets = [1, 1]
    # remainder of machines go into user subnet tree
    num_full_user_subnets = ((num_machines - 2) // USER_SUBNET_SIZE)
    subnets += [USER_SUBNET_SIZE] * num_full_user_subnets
    if ((num_machines - 2) % USER_SUBNET_SIZE) != 0:
        subnets.append((num_machines - 2) % USER_SUBNET_SIZE)
    return subnets


def generate_topology(subnets):
    """
    Generate the topology of the network, defining the connectivity between
    subnets.

    Arguments:
        list subnets : list of subnet sizes

    Returns:
        2D matrix topology : an adjacency matrix of subnets
    """
    num_subnets = len(subnets)
    # extra column for public connectivity
    topology = np.zeros((num_subnets, num_subnets + 1))
    # exposed subnet is connected to sensitive and first user subnet and also
    # to public
    for row in range(3):
        for col in range(4):
            if row > 0 and col == 0:
                continue
            topology[row][col] = 1
    if num_subnets == 3:
        return topology
    # all other subnets are part of user binary tree
    for row in range(2, num_subnets):
        # subnet connected to itself
        topology[row][row + 1] = 1
        # position in tree
        pos = row - 2
        if pos > 0:
            parent = ((pos - 1) // 2) + 3
            topology[row][parent] = 1
        child_left = ((2 * pos) + 1) + 3
        child_right = ((2 * pos) + 2) + 3
        if child_left < num_subnets + 1:
            topology[row][child_left] = 1
        if child_right < num_subnets + 1:
            topology[row][child_right] = 1
    return topology


def generate_sensitive_machines(subnets, r_sensitive, r_user):
    """
    Generate list of senstive machines.

    Arguments:
        list subnets : list of subnet sizes
        float r_sensitive : value for sensitive subnet information
        float r_user : value for user subnet information

    Returns:
        list sensitive_machines : list of [subnetID, machineID, value] lists
    """
    sensitive_machines = []
    # first sensitive machine is first machine in SENSITIVE network
    sensitive_machines.append([SENSITIVE, 0, r_sensitive])
    # second sensitive machine is last machine on last USER network
    sensitive_machines.append([len(subnets) - 1, subnets[-1] - 1, r_user])
    return sensitive_machines


def load_yaml_file(file_name):
    """
    Load yaml file.

    Raises error if file not found or invalid yaml file.

    Arguments:
        str file_name : path and name of config file
    """
    with open(file_name, "r") as stream:
        try:
            config = yaml.load(stream)
            return config
        except yaml.YAMLError as exc:
            raise exc


def check_config_valid(config):
    """
    Checks if a config dictionary is valid.

    Raises relevant error if config not valid.

    Arguments:
        dict config : the config dictionary to check
    """
    # 0. check correct number of keys
    if len(config) != len(VALID_CONFIG_KEYS):
        raise KeyError("Incorrect number of config file keys: {0} != {1}"
                       .format(len(config), len(VALID_CONFIG_KEYS)))

    # 1. check keys are valid and values are correct type
    for k, v in config.items():
        if k not in VALID_CONFIG_KEYS.keys():
            raise KeyError("{0} not a valid config file key".format(k))
        expected_type = VALID_CONFIG_KEYS[k]
        if type(v) is not expected_type:
            raise TypeError(("{0} invalid type for config file key '{1}': {2}"
                             " != {3}").format(v, k, type(v), expected_type))

    # 2. check subnets is valid list of positive ints
    subnets = config["subnets"]
    if len(subnets) < 1:
        raise ValueError("Subnets connot be empty list")
    for subnet_size in subnets:
        if type(subnet_size) is not int or subnet_size <= 0:
            raise ValueError("{0} invalid subnet size, must be positive int"
                             .format(subnet_size))

    # 3. check topology is valid adjacency matrix
    topology = config["topology"]
    if len(topology) != len(subnets):
        raise ValueError(("Number of rows in topology adjacency "
                         "matrix must equal number of subnets: {0} != {1}")
                         .format(len(topology), len(subnets)))
    for row in config["topology"]:
        if type(row) is not list:
            raise ValueError("topology must be 2D adjacency matrix "
                             "(i.e. list of lists)")
        if len(row) != len(subnets) + 1:
            raise ValueError(("Number of columns in topology "
                              "adjaceny matrix must equal number of subnets+1:"
                              " {0} != {1}").format(len(row), len(subnets)+1))
        for col in row:
            if type(col) is not int or (col != 1 and col != 0):
                raise ValueError(("Subnet_connections adjaceny matrix must "
                                  "contain only 1 (connected) or 0 (not "
                                  "connected): {0} invalid").format(col))

    # 4. check services is postive int
    services = config["services"]
    if services < 1:
        raise ValueError("{0} Invalid number of services, must be positive int"
                         .format(services))

    # 5. check sensitive_machines is valid list of (subnet, id, value) tuples
    sensitive_machines = config["sensitive_machines"]
    total_machines = sum(subnets)
    if len(sensitive_machines) < 1:
        raise ValueError(("Number of sensitive machines must be >= 1: {0} not"
                          " >= 1").format(len(sensitive_machines)))
    if len(sensitive_machines) > total_machines:
        raise ValueError(("Number of sensitive machines must be <= total "
                         "number of machines: {0} not <= {1}")
                         .format(len(sensitive_machines), total_machines))

    # 5.b sensitive machines must be valid address
    for m in sensitive_machines:
        if len(m) != 3:
            raise ValueError(("Invalid sensitive machine tuple: {0} != "
                             "(int, int, (int or float))").format(m))
        if type(m[0]) is not int or m[0] < 0 or m[0] >= len(subnets):
            raise ValueError(("Invalid sensitive machine tuple: subnet_id must"
                              " be a valid subnet: {0} != non-negative int "
                              "less than {1}").format(m[0], len(subnets)))
        if type(m[1]) is not int or m[1] < 0 or m[1] >= subnets[m[0]]:
            raise ValueError(("Invalid sensitive machine tuple: machine_id "
                              "must be a valid int: {0} != non-negative int "
                              "less than {1}").format(m[1], subnets[m[0]]))
        if (type(m[2]) is not int and type(m[2]) is not float) or m[2] <= 0:
            raise ValueError(("Invalid sensitive machine tuple: invalid value:"
                              " {0} != a positive int or float").format(m[2]))

    # 5.c sensitive machines must not contain duplicate addresses
    for i, m in enumerate(sensitive_machines):
        for j, n in enumerate(sensitive_machines):
            if i != j and m[0] == n[0] and m[1] == n[1]:
                raise(ValueError(("Sensitive machines list must not contain "
                                  "duplicate machine addresses: {0} == {1}")
                                 .format(m, n)))
