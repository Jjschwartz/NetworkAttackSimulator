"""
This module contains functionality for loading network configurations from yaml
files and also for generating configurations based on number of machines and
services in network using standard formula.
"""
import yaml
from collections import OrderedDict
from cyber_attack_simulator.envs.machine import Machine
from cyber_attack_simulator.envs.generator import get_machine_value

# dictionary of valid key names and value types for config file
VALID_CONFIG_KEYS = {"subnets": list,
                     "topology": list,
                     "sensitive_machines": list,
                     "num_services": int,
                     "service_exploits": dict,
                     "machine_configurations": dict,
                     "firewall": dict}

# Constants for generating network
INTERNET = 0


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
        parsed_config = parse_config(config)
        return parsed_config
    except (KeyError, TypeError, ValueError) as exc:
        raise exc


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
    if len(topology) != len(subnets) + 1:
        raise ValueError(("Number of rows in topology adjacency "
                         "matrix must equal number of subnets: {0} != {1}")
                         .format(len(topology), len(subnets) + 1))
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
    num_services = config["num_services"]
    if num_services < 1:
        raise ValueError("{0} Invalid number of services, must be positive int"
                         .format(num_services))

    # 5. check sensitive_machines is valid list of (subnet, id, value) tuples
    sensitive_machines = config["sensitive_machines"]
    num_machines = sum(subnets)
    if len(sensitive_machines) < 1:
        raise ValueError(("Number of sensitive machines must be >= 1: {0} not"
                          " >= 1").format(len(sensitive_machines)))
    if len(sensitive_machines) > num_machines:
        raise ValueError(("Number of sensitive machines must be <= total "
                         "number of machines: {0} not <= {1}")
                         .format(len(sensitive_machines), num_machines))

    # 5.b sensitive machines must be valid address
    for m in sensitive_machines:
        if len(m) != 3:
            raise ValueError(("Invalid sensitive machine tuple: {0} != "
                             "(int, int, (int or float))").format(m))
        if not is_valid_subnet_ID(m[0], subnets):
            raise ValueError(("Invalid sensitive machine tuple: subnet_id must"
                              " be a valid subnet: {0} != non-negative int "
                              "less than {1}").format(m[0], len(subnets) + 1))
        if not is_valid_machine_address(m[0], m[1], subnets):
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

    # 6 check service_exploits list is valid
    service_exploits = config["service_exploits"]
    if len(service_exploits) != num_services:
        raise ValueError("Service exploits dictionary must contain an entry for each service:"
                         " {} != {}".format(len(service_exploits), num_services))
    for e in service_exploits.values():
        if not is_valid_service_exploit(e):
            raise ValueError("{}. Service exploit must be a list of form: [probability, cost]."
                             .format(e))

    # 7 check machine_configurations
    machine_configs = config["machine_configurations"]
    if len(machine_configs) != num_machines:
        raise ValueError("Number of machine configurations must match the number of machines in "
                         "network: {} != {}".format(len(machine_configs), num_machines))
    if not has_all_machine_addresses(machine_configs, subnets):
        raise ValueError("Machine configurations must have no duplicates and have an address for "
                         "each machine on network.")
    for cfg in machine_configs.values():
        if not is_valid_machine_config(cfg, service_exploits):
            raise ValueError("Machine configurations must be at list, contain at least one "
                             "service exploit, contain only valid service exploits and contain "
                             "no duplicates: {} is invalid".format(cfg))

    # 8 check firewall is valid
    firewall = config["firewall"]
    if not contains_all_required_firewalls(firewall, topology):
        raise ValueError("Firewall dictionary must contain two entries for each subnet connection "
                         "in network (including from outside) as defined by network topology "
                         "matrix")
    for f in firewall.values():
        if not is_valid_firewall_setting(f, service_exploits):
            raise ValueError("Firewall setting must be a list, contain only service exploits and "
                             "contain no duplicates: {} is not valid".format(f))


def is_valid_subnet_ID(subnet_ID, subnets):
    """
    Check if given subnet_ID is a valid subnet given subnet list
    """
    if type(subnet_ID) is not int or subnet_ID < 1 or subnet_ID > len(subnets):
        return False
    return True


def is_valid_machine_address(subnet_ID, machine_ID, subnets):
    """
    Check given (subnet_ID, machine_ID) is a valid address given subnets list
    """
    if not is_valid_subnet_ID(subnet_ID, subnets):
        return False
    if type(machine_ID) is not int or machine_ID < 0 or machine_ID >= subnets[subnet_ID - 1]:
        return False
    return True


def is_valid_service_exploit(e):
    """
    Check whether service exploit is valid
    """
    if type(e) != list or len(e) != 2:
        return False
    if type(e[0]) != float or (type(e[1]) != float and type(e[1]) != int):
        return False
    if e[0] < 0 or e[0] > 1 or e[1] < 0:
        return False
    return True


def has_all_machine_addresses(addresses, subnets):
    """
    Check that list of (subnet_ID, machine_ID) tuples contains all addresses on network based
    on subnets list
    """
    for s_id, s_size in enumerate(subnets):
        for m in range(s_size):
            # +1 to s_id since first subnet is 1
            if str((s_id + 1, m)) not in addresses:
                return False
    return True


def is_valid_machine_config(cfg, service_exploits):
    """
    Check if a machine config is valid or not given the list of service exploits available

    N.B. each machine config must contain at least one of the services
    """
    if type(cfg) != list or len(cfg) == 0:
        return False
    for service in cfg:
        if service not in service_exploits:
            return False
    for i, x in enumerate(cfg):
        for j, y in enumerate(cfg):
            if i != j and x == y:
                return False
    return True


def contains_all_required_firewalls(firewall, topology):
    """
    Check that the list of firewall settings contains all necessary firewall settings
    """
    for src, row in enumerate(topology):
        for dest, col in enumerate(row):
            if src == dest:
                continue
            if col == 1 and (str((src, dest)) not in firewall or str((dest, src)) not in firewall):
                return False
    return True


def is_valid_firewall_setting(f, service_exploits):
    """
    Check that a given firewall setting is valid given the list of service exploits available
    """
    if type(f) != list:
        return False
    for service in f:
        if service not in service_exploits:
            return False
    for i, x in enumerate(f):
        for j, y in enumerate(f):
            if i != j and x == y:
                return False
    return True


def parse_config(config):
    """
    Parse config file so that it is in the correct form for the environment class
    """
    parsed_config = {}
    subnets = config["subnets"]
    subnets.insert(0, 1)
    parsed_config["subnets"] = subnets
    parsed_config["num_services"] = config["num_services"]
    parsed_config["topology"] = config["topology"]
    parsed_config["sensitive_machines"] = config["sensitive_machines"]
    parsed_config["machines"] = parse_machines(config["machine_configurations"],
                                               config["service_exploits"],
                                               config["sensitive_machines"])
    parsed_config["firewall"] = parse_firewall(config["firewall"])
    parsed_config["service_exploits"] = config["service_exploits"]

    return parsed_config


def parse_machines(machine_configurations, service_exploits, sensitive_machines):
    """
    Load the machine objects for the network

    Returns:
        dict machine : ordered dictionary of machines in network, with address as keys and machine
                objects as values
    """
    machines = OrderedDict()
    for address, services in machine_configurations.items():
        formatted_address = eval(address)
        cfg = construct_machine_config(services, service_exploits)
        value = get_machine_value(sensitive_machines, formatted_address)
        machines[formatted_address] = Machine(formatted_address, cfg, value)
    return machines


def construct_machine_config(m_services, service_exploits):
    """
    Construct numpy array of machine configuration
    """
    cfg = {}
    for service in service_exploits.keys():
        if service in m_services:
            cfg[service] = True
        else:
            cfg[service] = False
    return cfg


def parse_firewall(firewall):
    """
    Parse firewall into firewall dict
    """
    firewall_dict = {}
    for connect, v in firewall.items():
        firewall_dict[eval(connect)] = v
    return firewall_dict
