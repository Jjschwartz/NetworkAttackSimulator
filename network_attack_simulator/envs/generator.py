"""
This module contains functionality for generating network configurations based on number of
machines and services in network using standard formula.
"""
import numpy as np
from collections import OrderedDict
from network_attack_simulator.envs.machine import Machine

# Constants for generating network
USER_SUBNET_SIZE = 5
INTERNET = 0
DMZ = 1
SENSITIVE = 2
USER = 3


def generate_config(num_machines, num_services,
                    r_sensitive=10, r_user=10,
                    exploit_cost=1, exploit_probs=1.0,
                    uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0,
                    restrictiveness=5, seed=1):
    """
    Generate the network configuration based on standard formula.

    Machine Configuration distribution:
        1. if uniform=True
            => machine configurations are chosen uniformly at random from set of all valid
               possible configurations
        2. if uniform=False
            => machine configurations are chosen to be corelated (see below)

    CORRELATED CONFIGURATIONS:
    The distribution of configurations of each machine in the network are generated using a
    Nested Dirichlet Process, so that across the network machines will have corelated
    configurations (i.e. certain services/configurations will be more common across machines on
    the network), the degree of corelation is controlled by alpha_H and alpha_V, with lower
    values leading to greater corelation.

    lambda_V controls the average number of services running per machine. Higher values will
    mean more services (so more vulnerable) machines on average.

    EXPLOIT PROBABILITIES
    Success probabilities of each exploit are determined as follows:
        - None - probabilities generated randomly from uniform distribution
        - "mixed" - probabilities randomly chosen from distribution of low: 0.2,
            med: 0.5 and high: 0.8 with probability of level based on attack complexity
            distribution of top 10 vulnerabilities in 2017.
        - single-float - probability of each exploit is set to value
        - list of float - probability of each exploit is set to
            corresponding value in list

    For deterministic exploits set exploit_probs=1.0

    Arguments:
        int num_machines : number of machines to include in network
            (minimum is 3)
        int num_exploits : number of exploits (and hence services) to use
            in environment (minimum is 1)
        float r_sensitive : reward for sensitive subnet documents
        float r_user : reward for user subnet documents
        int exploit_cost : cost for an exploit
        mixed exploit_probs : success probability of exploits
        bool uniform : whether to use uniform distribution of machine configs or corelated
                       machine configs
        float alpha_H : (only used when uniform=False), scaling/concentration parameter for
                        controlling corelation between machine configurations (must be > 0)
        float alpha_V : (only used when uniform=False) scaling/concentration parameter for
                        controlling corelation between services across machine configurations
                        (must be > 0)
        float lambda_V : (only used when uniform=False) parameter for controlling average
                         number of services running per machine configuration (must be > 0)
        int restrictiveness : max number of services allowed to pass through firewalls between
                              zones
        int seed : random number generator seed

    Returns:
        dict config : network configuration as dictionary
    """
    assert 0 < num_services
    assert 2 < num_machines
    assert 0 < r_sensitive and 0 < r_user
    assert 0 < alpha_H and 0 < alpha_V and 0 < lambda_V
    assert 0 < restrictiveness
    np.random.seed(seed)
    config = {}
    subnets = generate_subnets(num_machines)
    config["subnets"] = subnets
    config["topology"] = generate_topology(subnets)
    config["num_services"] = num_services
    s_machines = generate_sensitive_machines(subnets, r_sensitive, r_user)
    config["sensitive_machines"] = s_machines
    machines = generate_machines(subnets, num_services, s_machines, uniform,
                                 alpha_H, alpha_V, lambda_V)
    config["machines"] = machines
    config['firewall'] = generate_firewall_map(subnets, config["topology"], num_services, machines,
                                               restrictiveness)
    config["service_exploits"] = generate_service_exploits(num_services, exploit_cost,
                                                           exploit_probs)
    return config


def generate_subnets(num_machines):
    """
    Generate list of subnet sizes

    Argument:
        int num_machines : number of machines in network

    Returns:
        list subnets : list of number of machines in each subnet
    """
    # Internet (0), DMZ (1) and sensitive (2) subnets both contain 1 machine
    subnets = [1, 1, 1]
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
    # including internet subnet
    num_subnets = len(subnets)
    topology = np.zeros((num_subnets, num_subnets))
    # DMZ subnet is connected to sensitive and first user subnet and also
    # to internet
    for row in range(USER + 1):
        for col in range(USER + 1):
            if row == INTERNET and col > DMZ:
                continue
            if row > DMZ and col == INTERNET:
                continue
            topology[row][col] = 1
    if num_subnets == USER + 1:
        return topology
    # all other subnets are part of user binary tree
    for row in range(USER, num_subnets):
        # subnet connected to itself
        topology[row][row] = 1
        # position in tree
        pos = row - USER
        if pos > 0:
            parent = ((pos - 1) // 2) + 3
            topology[row][parent] = 1
        child_left = ((2 * pos) + 1) + 3
        child_right = ((2 * pos) + 2) + 3
        if child_left < num_subnets:
            topology[row][child_left] = 1
        if child_right < num_subnets:
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


def generate_machines(subnets, num_services, sensitive_machines, uniform, alpha_H, alpha_V,
                      lambda_V):
    """
    Generate the machines on the network.

    Argument:
        list subnets : list of subnet sizes
        int num_services : number if services running in network
        list sensitive_machines : list of [subnetID, machineID, value] lists
        bool uniform : whether to use uniform distribution of machine configs or corelated
                       machine configs
        float alpha_H : scaling/concentration parameter for controlling corelation between
                        machine configurations
        float alpha_V : scaling/concentration parameter for controlling corelation between
                        services across machine configurations
        float lambda_V : parameter for controlling average number of services running per
                         machine configuration

    Returns:
        dict machine : ordered dictionary of machines in network, with
            address as keys and machine objects as values
    """
    machines = OrderedDict()

    if uniform:
        machine_config_set = possible_machine_configs(num_services)
        num_configs = len(machine_config_set)
    else:
        prev_configs = []
        prev_vuls = []
        host_num = 0

    for subnet, size in enumerate(subnets):
        if subnet == INTERNET:
            continue
        for m in range(size):
            if uniform:
                cfg = machine_config_set[np.random.choice(num_configs)]
            else:
                cfg = get_machine_config(host_num, alpha_H, prev_configs, alpha_V, prev_vuls,
                                         lambda_V, num_services)
                host_num += 1
            address = (subnet, m)
            value = get_machine_value(sensitive_machines, address)
            machine = Machine(address, cfg, value)
            machines[address] = machine
    return machines


def possible_machine_configs(ns):
    """
    Generate set of all possible machine service configurations based
    on number of exploits/services in environment.

    Note: Each machine is vulnerable to at least one exploit, so there is
    no configuration where all services are absent.

    Argument:
        int ns : number of possible services on machines

    Returns:
        ndarray configs : numpy array of all possible configurations, where
            each configuration is a list of bools corresponding to the
            presence or absence of a service
    """
    # remove last permutation which is all False
    configs = permutations(ns)[:-1]
    return configs


def permutations(n):
    """
    Generate list of all possible permutations of n bools

    N.B First permutation in list is always the all True permutation and final
    permutation in list is always the all False permutationself.

    perms[1] = [True, ..., True]
    perms[-1] = [False, ..., False]

    Arguments:
    int n : bool list length

    Returns:
    list[list] perms : list of all possible permutations of n bools
    """
    # base cases
    if n <= 0:
        return []
    if n == 1:
        return [[True], [False]]

    perms = []
    for p in permutations(n - 1):
        perms.append([True] + p)
        perms.append([False] + p)
    return perms


def get_machine_config(host_num, alpha_H, prev_configs, alpha_V, prev_vuls, lambda_V,
                       num_services):
    """
    Select a machine configuration from all possible configurations based using a Nested
    Dirichlet Process
    """
    if host_num == 0 or np.random.rand() < (alpha_H / (alpha_H + host_num - 1)):
        # if first host or with prob proportional to alpha_H choose new config
        new_config = sample_config(alpha_V, prev_vuls, lambda_V, num_services)
    else:
        # sample uniformly from previous sampled configs
        new_config = prev_configs[np.random.choice(len(prev_configs))]
    prev_configs.append(new_config)
    return new_config


def sample_config(alpha_V, prev_vuls, lambda_V, num_services):
    """
    Sample a machine configuration from all possible configurations based using a Dirichlet
    Process
    """
    # no services present by default
    new_config = [False for i in range(num_services)]
    # randomly get number of times to sample using poission dist in range
    # (0, num_services) minimum 1 service running
    n = max(np.random.poisson(lambda_V), 1)
    # draw n samples from Dirichlet Process (alpha_V, uniform dist of services)
    for i in range(n):
        if i == 0 or np.random.rand() < (alpha_V / (alpha_V + i - 1)):
            # draw randomly from uniform dist over services
            x = np.random.randint(0, num_services)
        else:
            # draw uniformly at random from previous choices
            x = np.random.choice(prev_vuls)
        new_config[x] = True
        prev_vuls.append(x)
    return new_config


def get_machine_value(sensitive_machines, address):
    """
    Get the value of machine at given address
    """
    for m in sensitive_machines:
        if m[0] == address[0] and m[1] == address[1]:
            return float(m[2])
    return 0.0


def generate_firewall_map(subnets, topology, num_services, machines, restrictiveness):
    """
    Generate the firewall rules as a mapping from (src, dest) connection to set of allowed services,
    which defines for each service whether traffic using that service is allowed between pairs of
    subnets.

    Restrictiveness parameter controls how many services are blocked by firewall between zones
    (i.e. between internet, DMZ, sensitive and user zones). Traffic from at least one service
    running on each subnet will be allowed between each zone. This may mean more services will be
    allowed than restrictiveness parameter

    Arguments:
        list subnets : list of subnet sizes
        2Dmatrix topology : network topology matrix
        int num_services : number if services running in network
        dict machine : ordered dictionary of machines in network, with address as keys and
                       machine objects as values
        int restrictiveness : max number of services allowed to pass through a firewall

    Returns:
        dict firewall_dict : firewall map
    """
    num_subnets = len(subnets)
    firewall_map = {}

    # find services running on each subnet, and set to true
    subnet_services = {}
    for m in machines.values():
        subnet = m.address[0]
        if subnet not in subnet_services:
            subnet_services[subnet] = set()
        for name, present in m._services.items():
            if present:
                subnet_services[subnet].add(name)
    subnet_services[INTERNET] = set()

    service_list = list(range(num_services))

    for src in range(num_subnets):
        for dest in range(len(subnets)):
            if src == dest or not topology[src][dest]:
                # no inter subnet connection so no firewall
                continue
            elif src > SENSITIVE and dest > SENSITIVE:
                # all services allowed between user subnets
                allowed = set(service_list)
                firewall_map[(src, dest)] = allowed
                continue
            # else src and dest in different zones => block services based on restrictiveness
            dest_avail = subnet_services[dest].copy()
            if len(dest_avail) < restrictiveness:
                # restrictiveness not limiting allowed traffic, all services allowed
                firewall_map[(src, dest)] = dest_avail.copy()
                continue
            # add at least one service to allowed service
            dest_allowed = np.random.choice(list(dest_avail))
            # for dest subnet choose available services upto restrictiveness limit or all services
            dest_avail.remove(dest_allowed)
            allowed = set()
            allowed.add(dest_allowed)
            while len(allowed) < restrictiveness:
                dest_allowed = np.random.choice(list(dest_avail))
                if dest_allowed not in allowed:
                    allowed.add(dest_allowed)
                    dest_avail.remove(dest_allowed)
            firewall_map[(src, dest)] = allowed
    return firewall_map


def generate_service_exploits(num_services, exploit_cost, exploit_probs):
    """
    Generate service exploit dictionary which maps services to (probability, cost) tuples

    Arguments:
        int num_services : number of services that agent has exploit for (minimum is 1)
        int exploit_cost : cost for an exploit
        mixed exploit_probs : success probability of exploits (see contstructor comment for info)

    Returns:
        dict service_exploits : service exploits map
    """
    service_exploits = {}
    if exploit_probs is None:
        exploit_probs = np.random.random_sample(num_services)
    elif exploit_probs == 'mixed':
        # success probability of low, med, high attack complexity
        if num_services == 1:
            # for case where only 1 service ignore low probability actions
            # since could lead to unnecessarily long attack paths
            levels = [0.5, 0.8]
            probs = [0.5, 0.5]
        else:
            levels = [0.2, 0.5, 0.8]
            probs = [0.2, 0.4, 0.4]
        exploit_probs = np.random.choice(levels, num_services, p=probs)
    elif type(exploit_probs) is list:
        if len(exploit_probs) == num_services:
            raise ValueError("Lengh of exploit probability list must equal number of services")
        for e in exploit_probs:
            if e <= 0.0 or e > 1.0:
                raise ValueError("Exploit probabilities must be > 0.0 and <=1.0")
    else:
        if exploit_probs <= 0.0 or exploit_probs > 1.0:
            raise ValueError("Exploit probabilities must be > 0.0 and <=1.0")
        exploit_probs = [exploit_probs] * num_services

    for srv in range(num_services):
        service_exploits[srv] = (exploit_probs[srv], exploit_cost)

    return service_exploits
