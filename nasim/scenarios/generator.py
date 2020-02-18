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
                 r_sensitive=10,
                 r_user=10,
                 exploit_cost=1,
                 exploit_probs=1.0,
                 scan_cost=1,
                 uniform=False,
                 alpha_H=2.0,
                 alpha_V=2.0,
                 lambda_V=1.0,
                 restrictiveness=5,
                 seed=None):
        """Generate the network configuration based on standard formula.

        Arguments
        ---------
        num_hosts : int
            number of hosts to include in network (minimum is 3)
        num_exploits : int
            number of exploits (and hence services) to use (minimum is 1)
        r_sensitive : float, optional
            reward for sensitive subnet documents (default=10)
        r_user : float, optional
            reward for user subnet documents (default=10)
        exploit_cost : (int, float), optional
            cost for an exploit (default=1)
        exploit_probs : mixed, optional
            success probability of exploits (default=1.0)
        scan_cost : (int, float), optional
            cost for a scan (default=1)
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
        seed : int, optional
            random number generator seed (default=None)

        Returns
        -------
        scenario_dict : dict
            dictionary with scenario definition
        """
        assert 0 < num_services
        assert 2 < num_hosts
        assert 0 < r_sensitive and 0 < r_user
        assert 0 < alpha_H and 0 < alpha_V and 0 < lambda_V
        assert 0 < restrictiveness

        if seed is not None:
            np.random.seed(seed)

        self._generate_subnets(num_hosts)
        self._generate_topology()
        self._generate_services(num_services)
        self._generate_sensitive_hosts(r_sensitive, r_user)
        if uniform:
            self._generate_uniform_hosts()
        else:
            self._generate_correlated_hosts(alpha_H, alpha_V, lambda_V)
        self._generate_firewall(restrictiveness)
        self._generate_exploits(exploit_cost, exploit_probs)
        self.scan_cost = scan_cost
        return self._construct_scenario()

    def _construct_scenario(self):
        scenario_dict = dict()
        scenario_dict[u.SUBNETS] = self.subnets
        scenario_dict[u.TOPOLOGY] = self.topology
        scenario_dict[u.SERVICES] = self.services
        scenario_dict[u.SENSITIVE_HOSTS] = self.sensitive_hosts
        scenario_dict[u.EXPLOITS] = self.exploits
        scenario_dict[u.SCAN_COST] = self.scan_cost
        scenario_dict[u.FIREWALL] = self.firewall
        scenario_dict[u.HOSTS] = self.hosts
        return Scenario(scenario_dict)

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
        services = []
        for s in range(num_services):
            services.append(f"srv_{s}")
        self.services = services

    def _generate_sensitive_hosts(self, r_sensitive, r_user):
        sensitive_hosts = {}
        # first sensitive host is first host in SENSITIVE network
        sensitive_hosts[(SENSITIVE, 0)] = r_sensitive
        # second sensitive host is last host on last USER network
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
                cfg = host_config_set[np.random.choice(num_configs)]
                cfg = self._convert_to_service_map(cfg)
                address = (subnet, h)
                value = self._get_host_value(address)
                host = Host(address, cfg, value)
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
        host_num = 0
        for subnet, size in enumerate(self.subnets):
            if subnet == u.INTERNET:
                continue
            for m in range(size):
                cfg = self._get_host_config(host_num, alpha_H, prev_configs,
                                            alpha_V, prev_vuls, lambda_V)
                cfg = self._convert_to_service_map(cfg)
                host_num += 1
                address = (subnet, m)
                value = self._get_host_value(address)
                host = Host(address, cfg, value)
                hosts[address] = host
        self.hosts = hosts

    def _get_host_config(self, host_num, alpha_H, prev_configs, alpha_V, prev_vuls, lambda_V):
        """Select a host configuration from all possible configurations based using a Nested
        Dirichlet Process
        """
        if host_num == 0 or np.random.rand() < (alpha_H / (alpha_H + host_num - 1)):
            # if first host or with prob proportional to alpha_H choose new config
            new_config = self._sample_config(alpha_V, prev_vuls, lambda_V)
        else:
            # sample uniformly from previous sampled configs
            new_config = prev_configs[np.random.choice(len(prev_configs))]
        prev_configs.append(new_config)
        return new_config

    def _sample_config(self, alpha_V, prev_vuls, lambda_V):
        """Sample a host configuration from all possible configurations based using a Dirichlet
        Process
        """
        num_services = len(self.services)
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

    def _convert_to_service_map(self, config):
        """Converts list of bools to a map from service name -> bool """
        service_map = {}
        for srv, val in zip(self.services, config):
            service_map[srv] = val
        return service_map

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

        # find services running on each subnet, and set to true
        subnet_services = {}
        for h in self.hosts.values():
            subnet = h.address[0]
            if subnet not in subnet_services:
                subnet_services[subnet] = set()
            for name, present in h.services.items():
                if present:
                    subnet_services[subnet].add(name)
        subnet_services[u.INTERNET] = set()

        service_list = list(range(len(self.services)))

        for src in range(num_subnets):
            for dest in range(len(self.subnets)):
                if src == dest or not self.topology[src][dest]:
                    # no inter subnet connection so no firewall
                    continue
                elif src > SENSITIVE and dest > SENSITIVE:
                    # all services allowed between user subnets
                    allowed = set(service_list)
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

    def _generate_exploits(self, exploit_cost, exploit_probs):
        exploits = {}
        exploit_probs = self._get_exploit_probs(exploit_probs)
        # we create one exploit per service
        for srv in range(len(self.services)):
            e_name = f"e_{srv}"
            exploits[e_name] = {
                u.EXPLOIT_SERVICE: srv,
                u.EXPLOIT_PROB: exploit_probs[srv],
                u.EXPLOIT_COST: exploit_cost}
        self.exploits = exploits

    def _get_exploit_probs(self, exploit_probs):
        num_services = len(self.services)
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

        return exploit_probs
