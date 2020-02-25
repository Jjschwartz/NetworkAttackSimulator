"""This module contains functionality for generating network configurations based on number of
hosts and services in network using standard formula.
"""
import numpy as np

from nasim.env.host import Host
import nasim.scenarios.utils as u
from nasim.scenarios import Scenario

# Constants for generating network
USER_SUBNET_SIZE = 5
DMZ = 1
SENSITIVE = 2
USER = 3


class ScenarioGenerator:
    """Generates a scenario based on standard formula.

    Host Configuration distribution:
        1. if uniform=True
            => host configurations are chosen uniformly at random from set of all valid
               possible configurations
        2. if uniform=False
            => host configurations are chosen to be corelated (see below)

    CORRELATED CONFIGURATIONS:
    The distribution of configurations of each host in the network are generated using a
    Nested Dirichlet Process, so that across the network hosts will have corelated
    configurations (i.e. certain services/configurations will be more common across hosts on
    the network), the degree of corelation is controlled by alpha_H and alpha_V, with lower
    values leading to greater corelation.

    lambda_V controls the average number of services running per host. Higher values will
    mean more services (so more vulnerable) hosts on average.

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
    """

    def generate(self,
                 num_hosts,
                 num_services,
                 num_os=2,
                 num_exploits=None,
                 r_sensitive=10,
                 r_user=10,
                 exploit_cost=1,
                 exploit_probs=1.0,
                 service_scan_cost=1,
                 os_scan_cost=1,
                 uniform=False,
                 alpha_H=2.0,
                 alpha_V=2.0,
                 lambda_V=1.0,
                 restrictiveness=5,
                 random_goal=False,
                 seed=None):
        """Generate the network configuration based on standard formula.

        Arguments
        ---------
        num_hosts : int
            number of hosts to include in network (minimum is 3)
        num_services : int
            number of services running on network (minimum is 1)
        num_os : int, optional
            number of OS running on network (minimum is 1) (default=2)
        num_exploits : int, optional
            number of exploits to use. minimum is 1. If None will use num_services.
            (default=None)
        r_sensitive : float, optional
            reward for sensitive subnet documents (default=10)
        r_user : float, optional
            reward for user subnet documents (default=10)
        exploit_cost : (int, float), optional
            cost for an exploit (default=1)
        exploit_probs : mixed, optional
            success probability of exploits (default=1.0)
        service_scan_cost : int or float, optional
            cost for a service scan (default=1)
        os_scan_cost : int or float, optional
            cost for an os scan (default=1)
        uniform : bool, optional
            whether to use uniform distribution or correlatted of host configs (default=False)
        alpha_H : float, optional
            (only used when uniform=False) Scaling/concentration parameter for
            controlling corelation between host configurations (must be > 0)
            (default=2.0)
        alpha_V : float, optional
            (only used when uniform=False) scaling/concentration parameter for
            controlling corelation between services across host configurations
            (must be > 0) (default=2.0)
        lambda_V : float, optional
            (only used when uniform=False) parameter for controlling average
            number of services running per host configuration (must be > 0)
            (default=1.0)
        restrictiveness : int, optional
            max number of services allowed to pass through firewalls between zones
            (default=5)
        random_goal : bool, optional
            whether to randomly assign the goal user host or not (default=False)
        seed : int, optional
            random number generator seed (default=None)

        Returns
        -------
        scenario_dict : dict
            dictionary with scenario definition
        """
        assert 0 < num_services
        assert 2 < num_hosts
        assert num_exploits is None or 0 < num_exploits
        assert 0 < num_os
        assert 0 < r_sensitive and 0 < r_user
        assert 0 < alpha_H and 0 < alpha_V and 0 < lambda_V
        assert 0 < restrictiveness

        if seed is not None:
            np.random.seed(seed)

        if num_exploits is None:
            num_exploits = num_services

        self._generate_subnets(num_hosts)
        self._generate_topology()
        self._generate_services(num_services)
        self._generate_os(num_os)
        self._generate_exploits(num_exploits, exploit_cost, exploit_probs)
        self._generate_sensitive_hosts(r_sensitive, r_user, random_goal)
        if uniform:
            self._generate_uniform_hosts()
        else:
            self._generate_correlated_hosts(alpha_H, alpha_V, lambda_V)
        self._ensure_host_vulnerability()
        self._generate_firewall(restrictiveness)
        self.service_scan_cost = service_scan_cost
        self.os_scan_cost = os_scan_cost
        return self._construct_scenario()

    def _construct_scenario(self):
        scenario_dict = dict()
        scenario_dict[u.SUBNETS] = self.subnets
        scenario_dict[u.TOPOLOGY] = self.topology
        scenario_dict[u.SERVICES] = self.services
        scenario_dict[u.OS] = self.os
        scenario_dict[u.SENSITIVE_HOSTS] = self.sensitive_hosts
        scenario_dict[u.EXPLOITS] = self.exploits
        scenario_dict[u.SERVICE_SCAN_COST] = self.service_scan_cost
        scenario_dict[u.OS_SCAN_COST] = self.os_scan_cost
        scenario_dict[u.FIREWALL] = self.firewall
        scenario_dict[u.HOSTS] = self.hosts
        scenario = Scenario(scenario_dict)
        return scenario

    def _generate_subnets(self, num_hosts):
        # Internet (0), DMZ (1) and sensitive (2) subnets both contain 1 host
        subnets = [1, 1, 1]
        # remainder of hosts go into user subnet tree
        num_full_user_subnets = ((num_hosts - 2) // USER_SUBNET_SIZE)
        subnets += [USER_SUBNET_SIZE] * num_full_user_subnets
        if ((num_hosts - 2) % USER_SUBNET_SIZE) != 0:
            subnets.append((num_hosts - 2) % USER_SUBNET_SIZE)
        self.subnets = subnets

    def _generate_topology(self):
        # including internet subnet
        num_subnets = len(self.subnets)
        topology = np.zeros((num_subnets, num_subnets))
        # DMZ subnet is connected to sensitive and first user subnet and also
        # to internet
        for row in range(USER + 1):
            for col in range(USER + 1):
                if row == u.INTERNET and col > DMZ:
                    continue
                if row > DMZ and col == u.INTERNET:
                    continue
                topology[row][col] = 1
        if num_subnets == USER + 1:
            self.topology = topology
            return
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
        self.topology = topology

    def _generate_services(self, num_services):
        self.services = [f"srv_{s}" for s in range(num_services)]

    def _generate_os(self, num_os):
        self.os = [f"os_{i}" for i in range(num_os)]

    def _generate_sensitive_hosts(self, r_sensitive, r_user, random_goal):
        sensitive_hosts = {}
        # first sensitive host is first host in SENSITIVE network
        sensitive_hosts[(SENSITIVE, 0)] = r_sensitive

        # second sensitive host in USER network
        if random_goal and len(self.subnets) > SENSITIVE:
            # randomly choose user host to be goal
            subnet_id = np.random.randint(USER, len(self.subnets))
            host_id = np.random.randint(0, self.subnets[subnet_id])
            sensitive_hosts[(subnet_id, host_id)] = r_user
        else:
            # second last host in USER network is goal
            sensitive_hosts[(len(self.subnets)-1, self.subnets[-1]-1)] = r_user
        self.sensitive_hosts = sensitive_hosts

    def _generate_uniform_hosts(self):
        hosts = dict()
        host_config_set = self._possible_host_configs()
        num_configs = len(host_config_set)

        for subnet, size in enumerate(self.subnets):
            if subnet == u.INTERNET:
                continue
            for h in range(size):
                service_cfg = host_config_set[np.random.choice(num_configs)]
                service_cfg = self._convert_to_service_map(service_cfg)
                os = np.random.choice(self.os)
                os_cfg = self._convert_to_os_map(os)
                address = (subnet, h)
                value = self._get_host_value(address)
                host = Host(address, os_cfg.copy(), service_cfg.copy(), value)
                hosts[address] = host
        self.hosts = hosts

    def _possible_host_configs(self):
        """Generate set of all possible host service configurations based
        on number of exploits/services in environment.

        Note: Each host is vulnerable to at least one exploit, so there is
        no configuration where all services are absent.

        Returns
        -------
        configs : ndarray
            all possible configurations, where each configuration is a list of
            bools corresponding to the presence or absence of a service
        """
        # remove last permutation which is all False
        configs = self._permutations(len(self.services))[:-1]
        return configs

    def _permutations(self, n):
        """Generate list of all possible permutations of n bools

        N.B First permutation in list is always the all True permutation and final
        permutation in list is always the all False permutationself.

        perms[1] = [True, ..., True]
        perms[-1] = [False, ..., False]

        Arguments
        ---------
        n : int
            bool list length

        Returns
        -------
        perms : list[list]
            all possible permutations of n bools
        """
        # base cases
        if n <= 0:
            return []
        if n == 1:
            return [[True], [False]]

        perms = []
        for p in self._permutations(n - 1):
            perms.append([True] + p)
            perms.append([False] + p)
        return perms

    def _generate_correlated_hosts(self, alpha_H, alpha_V, lambda_V):
        hosts = dict()
        prev_configs = []
        prev_vuls = []
        prev_os = []
        host_num = 0
        for subnet, size in enumerate(self.subnets):
            if subnet == u.INTERNET:
                continue
            for m in range(size):
                services, os = self._get_host_config(host_num, alpha_H, prev_configs,
                                                     alpha_V, prev_vuls, lambda_V, prev_os)
                service_cfg = self._convert_to_service_map(services)
                os_cfg = self._convert_to_os_map(os)
                host_num += 1
                address = (subnet, m)
                value = self._get_host_value(address)
                host = Host(address, os_cfg.copy(), service_cfg.copy(), value)
                hosts[address] = host
        self.hosts = hosts

    def _get_host_config(self, host_num, alpha_H, prev_configs, alpha_V, prev_vuls, lambda_V, prev_os):
        """Select a host configuration from all possible configurations based using a Nested
        Dirichlet Process
        """
        if host_num == 0 or np.random.rand() < (alpha_H / (alpha_H + host_num - 1)):
            # if first host or with prob proportional to alpha_H choose new config
            new_config = self._sample_config(alpha_V, prev_vuls, lambda_V, prev_os)
        else:
            # sample uniformly from previous sampled configs
            new_config = prev_configs[np.random.choice(len(prev_configs))]
        prev_configs.append(new_config)
        return new_config

    def _sample_config(self, alpha_V, prev_vuls, lambda_V, prev_os):
        """Sample a host configuration from all possible configurations based using a Dirichlet
        Process
        """
        num_services = len(self.services)
        # no services present by default
        new_services_cfg = [False for i in range(num_services)]
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
            new_services_cfg[x] = True
            prev_vuls.append(x)
        # sample an os from Dirichlet Process (alpha_V, uniform dist of OSs)
        if len(prev_os) == 0 or np.random.rand() < (alpha_V / (alpha_V + i - 1)):
            # draw randomly from uniform dist over services
            os = np.random.choice(self.os)
        else:
            # draw uniformly at random from previous choices
            os = np.random.choice(prev_os)
            prev_os.append(os)
        return (new_services_cfg, os)

    def _is_sensitive_host(self, addr):
        return addr in self.sensitive_hosts

    def _convert_to_service_map(self, config):
        """Converts list of bools to a map from service name -> bool """
        service_map = {}
        for srv, val in zip(self.services, config):
            service_map[srv] = val
        return service_map

    def _convert_to_os_map(self, os):
        """Converts an OS string to a map from os name -> bool

        N.B. also adds an entry for None os, which makes it easier for vectorizing
        and checking if an exploit will work (since exploits can have os=None)
        """
        os_map = {None: False}
        for os_name in self.os:
            os_map[os_name] = os_name == os
        return os_map

    def _ensure_host_vulnerability(self):
        """Ensures each subnet has atleast one vulnerable host and all sensitive hosts
        are vulnerable
        """
        vulnerable_subnets = set()
        for host_addr, host in self.hosts.items():
            if not self._is_sensitive_host(host_addr) and host_addr[0] in vulnerable_subnets:
                continue
            if self._host_is_vulnerable(host):
                vulnerable_subnets.add(host_addr[0])
            elif self._is_sensitive_host(host_addr):
                self._update_host_to_vulnerable(host)
                vulnerable_subnets.add(host_addr[0])

        for subnet, size in enumerate(self.subnets):
            if subnet in vulnerable_subnets or subnet == u.INTERNET:
                continue
            host_num = np.random.randint(size)
            host = self.hosts[(subnet, host_num)]
            self._update_host_to_vulnerable(host)
            vulnerable_subnets.add(subnet)

    def _host_is_vulnerable(self, host):
        for e_def in self.exploits.values():
            if self._host_is_vulnerable_to_exploit(host, e_def):
                return True
        return False

    def _host_is_vulnerable_to_exploit(self, host, exploit_def):
        e_srv = exploit_def[u.EXPLOIT_SERVICE]
        e_os = exploit_def[u.EXPLOIT_OS]
        if not host.services[e_srv]:
            return False
        return e_os is None or host.os[e_os]

    def _update_host_to_vulnerable(self, host):
        """Update the host config so it's vulnerable to at least one exploit """
        # choose an exploit randomly and make host vulnerable to it
        e_def = np.random.choice(list(self.exploits.values()))
        host.services[e_def[u.EXPLOIT_SERVICE]] = True
        if e_def[u.EXPLOIT_OS] is not None:
            # must set all to false first, so only one host OS is true
            for os_name in host.os.keys():
                host.os[os_name] = False
            host.os[e_def[u.EXPLOIT_OS]] = True
        host.update_vector()

    def _get_host_value(self, address):
        return float(self.sensitive_hosts.get(address, 0.0))

    def _generate_firewall(self, restrictiveness):
        """Generate the firewall rules as a mapping from (src, dest) connection to set
        of allowed services, which defines for each service whether traffic using that
        service is allowed between pairs of subnets.

        Restrictiveness parameter controls how many services are blocked by firewall
        between zones (i.e. between internet, DMZ, sensitive and user zones). Traffic
        from at least one service running on each subnet will be allowed between each
        zone. This may mean more services will be allowed than restrictiveness parameter.

        Arguments
        ---------
        restrictiveness : int
            max number of services allowed to pass through a firewall

        Returns
        -------
        firewall_dict : dict
            firewall map
        """
        num_subnets = len(self.subnets)
        firewall = {}

        # find services running on each subnet that are vulnerable
        subnet_services = {}
        subnet_services[u.INTERNET] = set()
        for host_addr, host in self.hosts.items():
            subnet = host_addr[0]
            if subnet not in subnet_services:
                subnet_services[subnet] = set()
            for e_def in self.exploits.values():
                if self._host_is_vulnerable_to_exploit(host, e_def):
                    subnet_services[subnet].add(e_def[u.EXPLOIT_SERVICE])

        for src in range(num_subnets):
            for dest in range(num_subnets):
                if src == dest or not self.topology[src][dest]:
                    # no inter subnet connection so no firewall
                    continue
                elif src > SENSITIVE and dest > SENSITIVE:
                    # all services allowed between user subnets
                    allowed = set(self.services)
                    firewall[(src, dest)] = allowed
                    continue
                # else src and dest in different zones => block services based on restrictiveness
                dest_avail = subnet_services[dest].copy()
                if len(dest_avail) < restrictiveness:
                    # restrictiveness not limiting allowed traffic, all services allowed
                    firewall[(src, dest)] = dest_avail.copy()
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
                firewall[(src, dest)] = allowed
        self.firewall = firewall

    def _generate_exploits(self, num_exploits, exploit_cost, exploit_probs):
        exploits = {}
        exploit_probs = self._get_exploit_probs(num_exploits, exploit_probs)
        # add None since some exploits might work for all OS
        possible_os = self.os + [None]
        # we create one exploit per service
        exploits_added = 0
        while exploits_added < num_exploits:
            srv = np.random.choice(self.services)
            os = np.random.choice(possible_os)
            e_name = f"e_{srv}"
            if os is not None:
                e_name += f"_{os}"
            if e_name not in exploits:
                exploits[e_name] = {
                    u.EXPLOIT_SERVICE: srv,
                    u.EXPLOIT_OS: os,
                    u.EXPLOIT_PROB: exploit_probs[exploits_added],
                    u.EXPLOIT_COST: exploit_cost}
                exploits_added += 1
        self.exploits = exploits

    def _get_exploit_probs(self, num_exploits, exploit_probs):
        if exploit_probs is None:
            exploit_probs = np.random.random_sample(num_exploits)

        elif exploit_probs == 'mixed':
            # success probability of low, med, high attack complexity
            if num_exploits == 1:
                # for case where only 1 service ignore low probability actions
                # since could lead to unnecessarily long attack paths
                levels = [0.6, 0.9]
                probs = [0.5, 0.5]
            else:
                levels = [0.3, 0.6, 0.9]
                probs = [0.2, 0.4, 0.4]
            exploit_probs = np.random.choice(levels, num_exploits, p=probs)

        elif type(exploit_probs) is list:
            if len(exploit_probs) == num_exploits:
                raise ValueError("Lengh of exploit probability list must equal number of exploits")
            for e in exploit_probs:
                if e <= 0.0 or e > 1.0:
                    raise ValueError("Exploit probabilities must be > 0.0 and <=1.0")

        else:
            if exploit_probs <= 0.0 or exploit_probs > 1.0:
                raise ValueError("Exploit probabilities must be > 0.0 and <=1.0")
            exploit_probs = [exploit_probs] * num_exploits

        return exploit_probs
