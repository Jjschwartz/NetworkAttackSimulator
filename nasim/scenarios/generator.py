"""This module contains functionality for generating scenarios.

Specifically, it generates network configurations and action space
configurations based on number of hosts and services in network using standard
formula.
"""
import math
import numpy as np

import nasim.scenarios.utils as u
from nasim.scenarios import Scenario
from nasim.scenarios.host import Host

# Constants for generating network
USER_SUBNET_SIZE = 5
HOST_ASSIGNMENT_PERIOD = 40
DMZ = 1
SENSITIVE = 2
USER = 3

# Number of time to attempt to find valid vulnerable config
VUL_RETRIES = 5


class ScenarioGenerator:
    """Generates a scenario based on standard formula

    For explanation of the details of how scenarios are generated see
    :ref:`scenario_generation_explanation`.

    Notes
    -----

    **Exploit Probabilities**:

    Success probabilities of each exploit are determined based on the value of
    the ``exploit_probs`` argument, as follows:

    - ``exploit_probs=None`` - probabilities generated randomly from uniform
      distribution
    - ``exploit_probs="mixed"`` - probabilities are chosen from [0.3, 0.6, 0.9]
      with probability [0.2, 0.4, 0.4] (see :ref:`generated_exploit_probs` for
      explanation).
    - ``exploit_probs=float`` - probability of each exploit is set to value
    - ``exploit_probs=list[float]`` - probability of each exploit is set to
      corresponding value in list

    For deterministic exploits set ``exploit_probs=1.0``.

    **Privilege Escalation Probabilities**:

    Success probabilities of each privilege escalation are determined based
    on the value of the ``privesc_probs`` argument, and are determined the same
    as for exploits with the exclusion of the "mixed" option.

    **Host Configuration distribution**:

    1. if ``uniform=True`` then host configurations are chosen uniformly at
       random from set of all valid possible configurations
    2. if ``uniform=False`` host configurations are chosen to be correlated
       (see :ref:`correlated_configurations` for explanation)


    """

    def generate(self,
                 num_hosts,
                 num_services,
                 num_os=2,
                 num_processes=2,
                 num_exploits=None,
                 num_privescs=None,
                 r_sensitive=10,
                 r_user=10,
                 exploit_cost=1,
                 exploit_probs=1.0,
                 privesc_cost=1,
                 privesc_probs=1.0,
                 service_scan_cost=1,
                 os_scan_cost=1,
                 subnet_scan_cost=1,
                 process_scan_cost=1,
                 uniform=False,
                 alpha_H=2.0,
                 alpha_V=2.0,
                 lambda_V=1.0,
                 restrictiveness=5,
                 random_goal=False,
                 base_host_value=1,
                 host_discovery_value=1,
                 seed=None,
                 name=None,
                 step_limit=None,
                 address_space_bounds=None,
                 **kwargs):
        """Generate the network configuration based on standard formula.

        Parameters
        ----------
        num_hosts : int
            number of hosts to include in network (minimum is 3)
        num_services : int
            number of services running on network (minimum is 1)
        num_os : int, optional
            number of OS running on network (minimum is 1) (default=2)
        num_processes : int, optional
            number of processes running on hosts on network (minimum is 1)
            (default=2)
        num_exploits : int, optional
            number of exploits to use. minimum is 1. If None will use
            num_services (default=None)
        num_privescs : int, optional
            number of privilege escalation actions to use. minimum is 1.
            If None will use num_processes (default=None)
        r_sensitive : float, optional
            reward for sensitive subnet documents (default=10)
        r_user : float, optional
            reward for user subnet documents (default=10)
        exploit_cost : int or float, optional
            cost for an exploit (default=1)
        exploit_probs : None, float, list of floats or "mixed", optional
            success probability of exploits (default=1.0)
        privesc_cost : int or float, optional
            cost for an privilege escalation action (default=1)
        privesc_probs : None, float, list of floats, optional
            success probability of privilege escalation actions (default=1.0)
        service_scan_cost : int or float, optional
            cost for a service scan (default=1)
        os_scan_cost : int or float, optional
            cost for an os scan (default=1)
        subnet_scan_cost : int or float, optional
            cost for a subnet scan (default=1)
        process_scan_cost : int or float, optional
            cost for a process scan (default=1)
        uniform : bool, optional
            whether to use uniform distribution or correlated host configs
            (default=False)
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
            max number of services allowed to pass through firewalls between
            zones (default=5)
        random_goal : bool, optional
            whether to randomly assign the goal user host or not
            (default=False)
        base_host_value : int, optional,
            value of non sensitive hosts (default=1)
        host_discovery_value : int, optional
            value of discovering a host for the first time (default=1)
        seed : int, optional
            random number generator seed (default=None)
        name : str, optional
            name of the scenario, if None one will be generated (default=None)
        step_limit : int, optional
            max number of steps permitted in a single episode, if None there is
            no limit (default=None)
        address_space_bounds : (int, int), optional
            bounds for the (subnet#, host#) address space. If None bounds will
            be determined by the number of subnets in the scenario and the max
            number of hosts in any subnet.

        Returns
        -------
        Scenario
            scenario description
        """
        assert 0 < num_services
        assert 2 < num_hosts
        assert 0 < num_processes
        assert num_exploits is None or 0 < num_exploits
        assert num_privescs is None or 0 < num_privescs
        assert 0 < num_os
        assert 0 < r_sensitive and 0 < r_user
        assert 0 < alpha_H and 0 < alpha_V and 0 < lambda_V
        assert 0 < restrictiveness

        if seed is not None:
            np.random.seed(seed)

        if num_exploits is None:
            num_exploits = num_services

        if num_privescs is None:
            num_privescs = num_processes

        self._generate_subnets(num_hosts)
        self._generate_topology()
        self._generate_address_space_bounds(address_space_bounds)
        self._generate_os(num_os)
        self._generate_services(num_services)
        self._generate_processes(num_processes)
        self._generate_exploits(num_exploits, exploit_cost, exploit_probs)
        self._generate_privescs(num_privescs, privesc_cost, privesc_probs)
        self._generate_sensitive_hosts(r_sensitive, r_user, random_goal)
        self.base_host_value = base_host_value
        self.host_discovery_value = host_discovery_value
        if uniform:
            self._generate_uniform_hosts()
        else:
            self._generate_correlated_hosts(alpha_H, alpha_V, lambda_V)
        self._ensure_host_vulnerability()
        self._generate_firewall(restrictiveness)
        self.service_scan_cost = service_scan_cost
        self.os_scan_cost = os_scan_cost
        self.subnet_scan_cost = subnet_scan_cost
        self.process_scan_cost = process_scan_cost

        if name is None:
            name = f"gen_H{num_hosts}_E{num_exploits}_S{num_services}"
        self.name = name

        self.step_limit = step_limit

        return self._construct_scenario()

    def _construct_scenario(self):
        scenario_dict = dict()
        scenario_dict[u.SUBNETS] = self.subnets
        scenario_dict[u.ADDRESS_SPACE_BOUNDS] = self.address_space_bounds
        scenario_dict[u.TOPOLOGY] = self.topology
        scenario_dict[u.SERVICES] = self.services
        scenario_dict[u.PROCESSES] = self.processes
        scenario_dict[u.OS] = self.os
        scenario_dict[u.SENSITIVE_HOSTS] = self.sensitive_hosts
        scenario_dict[u.EXPLOITS] = self.exploits
        scenario_dict[u.PRIVESCS] = self.privescs
        scenario_dict[u.SERVICE_SCAN_COST] = self.service_scan_cost
        scenario_dict[u.OS_SCAN_COST] = self.os_scan_cost
        scenario_dict[u.SUBNET_SCAN_COST] = self.subnet_scan_cost
        scenario_dict[u.PROCESS_SCAN_COST] = self.process_scan_cost
        scenario_dict[u.FIREWALL] = self.firewall
        scenario_dict[u.HOSTS] = self.hosts
        scenario_dict[u.STEP_LIMIT] = self.step_limit
        scenario = Scenario(
            scenario_dict, name=self.name, generated=True
        )
        return scenario

    def _generate_subnets(self, num_hosts):
        # Internet (0) and sensitive (2) subnets both start with 1 host
        subnets = [1]
        # For every HOST_ASSIGNMENT_PERIOD hosts we have:
        # first host assigned to DMZ (1),
        dmz_hosts = math.ceil(num_hosts / HOST_ASSIGNMENT_PERIOD)
        subnets.append(dmz_hosts)

        # second host assigned sensitive (2)
        sensitive_hosts = math.ceil(num_hosts / (HOST_ASSIGNMENT_PERIOD+1))
        subnets.append(sensitive_hosts)

        # remainder of hosts go into user subnet tree
        num_user_hosts = num_hosts - dmz_hosts - sensitive_hosts
        num_full_user_subnets = num_user_hosts // USER_SUBNET_SIZE
        subnets += [USER_SUBNET_SIZE] * num_full_user_subnets
        if (num_user_hosts % USER_SUBNET_SIZE) != 0:
            subnets.append(num_user_hosts % USER_SUBNET_SIZE)
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

    def _generate_address_space_bounds(self, address_space_bounds):
        if address_space_bounds is None:
            address_space_bounds = (len(self.subnets), max(self.subnets))

        err_msg = (
            "address_space_bounds must be None or a tuple/list of length 2"
            f"containing positive ints. '{address_space_bounds}' is invalid"
        )
        assert isinstance(address_space_bounds, (tuple, list)), err_msg
        address_space_bounds = tuple(address_space_bounds)

        assert len(address_space_bounds) == 2, err_msg
        for val in address_space_bounds:
            assert isinstance(val, int) and 0 < val, err_msg
        assert address_space_bounds[0] >= len(self.subnets), \
            ("Number of subnets in address bound must be >= number of subnets"
             f" in the scenario. '{address_space_bounds[0]}' is invalid")
        assert address_space_bounds[1] >= max(self.subnets), \
            ("Number of hosts in address bound must be >= number of hosts "
             " in the largest subnet in the scenario. "
             f"'{address_space_bounds[1]}' is invalid")
        self.address_space_bounds = address_space_bounds

    def _generate_os(self, num_os):
        self.os = [f"os_{i}" for i in range(num_os)]

    def _generate_services(self, num_services):
        self.services = [f"srv_{s}" for s in range(num_services)]

    def _generate_processes(self, num_processes):
        self.processes = [f"proc_{s}" for s in range(num_processes)]

    def _generate_exploits(self, num_exploits, exploit_cost, exploit_probs):
        exploits = {}
        exploit_probs = self._get_action_probs(num_exploits, exploit_probs)
        # add None since some exploits might work for all OS
        possible_os = self.os + [None]
        # we create one exploit per service
        exploits_added = 0
        while exploits_added < num_exploits:
            srv = np.random.choice(self.services)
            os = np.random.choice(possible_os)
            al = np.random.randint(u.USER_ACCESS, u.ROOT_ACCESS+1)
            e_name = f"e_{srv}"
            if os is not None:
                e_name += f"_{os}"
            if e_name not in exploits:
                exploits[e_name] = {
                    u.EXPLOIT_SERVICE: srv,
                    u.EXPLOIT_OS: os,
                    u.EXPLOIT_PROB: exploit_probs[exploits_added],
                    u.EXPLOIT_COST: exploit_cost,
                    u.EXPLOIT_ACCESS: al
                }
                exploits_added += 1
        self.exploits = exploits

    def _generate_privescs(self, num_privesc, privesc_cost, privesc_probs):
        privescs = {}
        privesc_probs = self._get_action_probs(num_privesc, privesc_probs)
        # add None since some privesc might work for all OS
        possible_os = self.os + [None]

        # need to ensure there is a privesc for each OS,
        # or >= 1 OS agnostic privesc
        # This ensures we can make it possible to get ROOT access on a
        # host, independendent of the exploit the host is vulnerable too
        if num_privesc < len(self.os):
            os_choices = [None]
            os_choices.extend(
                list(np.random.choice(possible_os, num_privesc-1))
            )
        else:
            while True:
                os_choices = list(
                    np.random.choice(possible_os, num_privesc)
                )
                if None in os_choices \
                   or all([os in os_choices for os in self.os]):
                    break

        # we create one exploit per service
        privescs_added = 0
        while privescs_added < num_privesc:
            proc = np.random.choice(self.processes)
            os = os_choices[privescs_added]
            pe_name = f"pe_{proc}"
            if os is not None:
                pe_name += f"_{os}"
            if pe_name not in privescs:
                privescs[pe_name] = {
                    u.PRIVESC_PROCESS: proc,
                    u.PRIVESC_OS: os,
                    u.PRIVESC_PROB: privesc_probs[privescs_added],
                    u.PRIVESC_COST: privesc_cost,
                    u.PRIVESC_ACCESS: u.ROOT_ACCESS
                }
                privescs_added += 1
        self.privescs = privescs

    def _get_action_probs(self, num_actions, action_probs):
        if action_probs is None:
            action_probs = np.random.random_sample(num_actions)
        elif action_probs == 'mixed':
            # success probability of low, med, high attack complexity
            if num_actions == 1:
                # for case where only 1 service ignore low probability actions
                # since could lead to unnecessarily long attack paths
                levels = [0.6, 0.9]
                probs = [0.5, 0.5]
            else:
                levels = [0.3, 0.6, 0.9]
                probs = [0.2, 0.4, 0.4]
            action_probs = np.random.choice(levels, num_actions, p=probs)
        elif type(action_probs) is list:
            assert len(action_probs) == num_actions, \
                ("Length of action probability list must equal number of"
                 " exploits")
            for a in action_probs:
                assert 0.0 < a <= 1.0, \
                    "Action probabilities in list must be in (0.0, 1.0]"
        else:
            assert isinstance(action_probs, float), \
                ("Action probabilities must be float, list of floats or "
                 "'mixed' (exploit only)")
            assert 0.0 < action_probs <= 1.0, \
                "Action probability float must be in (0.0, 1.0]"
            action_probs = [action_probs] * num_actions

        return action_probs

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
        srv_config_set, proc_config_set = self._possible_host_configs()
        num_srv_configs = len(srv_config_set)
        num_proc_configs = len(proc_config_set)

        for subnet, size in enumerate(self.subnets):
            if subnet == u.INTERNET:
                continue
            for h in range(size):
                srv_cfg = srv_config_set[np.random.choice(num_srv_configs)]
                srv_cfg = self._convert_to_service_map(srv_cfg)

                proc_cfg = proc_config_set[np.random.choice(num_proc_configs)]
                proc_cfg = self._convert_to_process_map(proc_cfg)

                os = np.random.choice(self.os)
                os_cfg = self._convert_to_os_map(os)

                address = (subnet, h)
                value = self._get_host_value(address)
                host = Host(
                    address=address,
                    os=os_cfg.copy(),
                    services=srv_cfg.copy(),
                    processes=proc_cfg.copy(),
                    firewall={},
                    value=value,
                    discovery_value=self.host_discovery_value
                )
                hosts[address] = host
        self.hosts = hosts

    def _possible_host_configs(self):
        """Generate set of all possible host service and process configurations
        based on number of services and processes in environment.

        Note: Each host is vulnerable to at least one exploit and one privesc,
        so there is no configuration where all services and processes are
        absent.

        Returns
        -------
        list[list]
            all possible service configurations, where each configuration is
            a list of bools corresponding to the presence or absence of a
            service
        list[list]
            all possible process configurations, same as above except for
            processes
        """
        # remove last permutation which is all False
        srv_configs = self._permutations(len(self.services))[:-1]
        proc_configs = self._permutations(len(self.processes))[:-1]
        return srv_configs, proc_configs

    def _permutations(self, n):
        """Generate list of all possible permutations of n bools

        N.B First permutation in list is always the all True permutation
        and final permutation in list is always the all False permutationself.

        perms[1] = [True, ..., True]
        perms[-1] = [False, ..., False]

        Parameters
        ----------
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
        prev_os = []
        prev_srvs = []
        prev_procs = []
        host_num = 0
        for subnet, size in enumerate(self.subnets):
            if subnet == u.INTERNET:
                continue
            for m in range(size):
                os, services, processes = self._get_host_config(
                    host_num,
                    alpha_H,
                    prev_configs,
                    alpha_V,
                    lambda_V,
                    prev_os,
                    prev_srvs,
                    prev_procs
                )
                os_cfg = self._convert_to_os_map(os)
                service_cfg = self._convert_to_service_map(services)
                process_cfg = self._convert_to_process_map(processes)
                host_num += 1
                address = (subnet, m)
                value = self._get_host_value(address)
                host = Host(
                    address=address,
                    os=os_cfg.copy(),
                    services=service_cfg.copy(),
                    processes=process_cfg.copy(),
                    firewall={},
                    value=value,
                    discovery_value=self.host_discovery_value
                )
                hosts[address] = host
        self.hosts = hosts

    def _get_host_config(self,
                         host_num,
                         alpha_H,
                         prev_configs,
                         alpha_V,
                         lambda_V,
                         prev_os,
                         prev_srvs,
                         prev_procs):
        """Select a host configuration from all possible configurations based
        using a Nested Dirichlet Process
        """
        if host_num == 0 \
           or np.random.rand() < (alpha_H / (alpha_H + host_num - 1)):
            # if first host or with prob proportional to alpha_H
            # choose new config
            new_config = self._sample_config(
                alpha_V, prev_srvs, lambda_V, prev_os, prev_procs
            )
        else:
            # sample uniformly from previous sampled configs
            new_config = prev_configs[np.random.choice(len(prev_configs))]
        prev_configs.append(new_config)
        return new_config

    def _sample_config(self,
                       alpha_V,
                       prev_srvs,
                       lambda_V,
                       prev_os,
                       prev_procs):
        """Sample a host configuration from all possible configurations based
        using a Dirichlet Process
        """
        os = self._dirichlet_sample(
            alpha_V, self.os, prev_os
        )

        new_services_cfg = self._dirichlet_process(
            alpha_V, lambda_V, len(self.services), prev_srvs
        )

        new_process_cfg = self._dirichlet_process(
            alpha_V, lambda_V, len(self.processes), prev_procs
        )

        return os, new_services_cfg, new_process_cfg

    def _dirichlet_process(self,
                           alpha_V,
                           lambda_V,
                           num_options,
                           prev_vals):
        """Sample from all possible configurations using Dirichlet Process """
        # no options present by default
        new_cfg = [False for i in range(num_options)]

        # randomly get number of times to sample using poission dist with
        # minimum 1 option choice
        n = max(np.random.poisson(lambda_V), 1)

        # draw n samples from Dirichlet Process
        # (alpha_V, uniform dist of services)
        for i in range(n):
            if i == 0 or np.random.rand() < (alpha_V / (alpha_V + i - 1)):
                # draw randomly from uniform dist over services
                x = np.random.randint(0, num_options)
            else:
                # draw uniformly at random from previous choices
                x = np.random.choice(prev_vals)
            new_cfg[x] = True
            prev_vals.append(x)
        return new_cfg

    def _dirichlet_sample(self, alpha_V, choices, prev_vals):
        """Sample single choice using dirichlet process """
        # sample an os from Dirichlet Process (alpha_V, uniform dist of OSs)
        if len(prev_vals) == 0 \
           or np.random.rand() < (alpha_V / (alpha_V - 1)):
            # draw randomly from uniform dist over services
            choice = np.random.choice(choices)
        else:
            # draw uniformly at random from previous choices
            choice = np.random.choice(prev_vals)
        prev_vals.append(choice)
        return choice

    def _is_sensitive_host(self, addr):
        return addr in self.sensitive_hosts

    def _convert_to_service_map(self, config):
        """Converts list of bools to a map from service name -> bool """
        service_map = {}
        for srv, val in zip(self.services, config):
            service_map[srv] = val
        return service_map

    def _convert_to_process_map(self, config):
        """Converts list of bools to a map from process name -> bool """
        process_map = {}
        for proc, val in zip(self.processes, config):
            process_map[proc] = val
        return process_map

    def _convert_to_os_map(self, os):
        """Converts an OS string to a map from os name -> bool

        N.B. also adds an entry for None os, which makes it easier for
        vectorizing and checking if an exploit will work (since exploits can
        have os=None)
        """
        os_map = {}
        for os_name in self.os:
            os_map[os_name] = os_name == os
        return os_map

    def _ensure_host_vulnerability(self):
        """Ensures each subnet has at least one vulnerable host and all sensitive hosts
        are vulnerable
        """
        vulnerable_subnets = set()
        for host_addr, host in self.hosts.items():
            if not self._is_sensitive_host(host_addr) \
               and host_addr[0] in vulnerable_subnets:
                continue

            if self._is_sensitive_host(host_addr):
                if not self._host_is_vulnerable(host, u.ROOT_ACCESS):
                    self._update_host_to_vulnerable(host, u.ROOT_ACCESS)
                vulnerable_subnets.add(host_addr[0])
            elif self._host_is_vulnerable(host):
                vulnerable_subnets.add(host_addr[0])

        for subnet, size in enumerate(self.subnets):
            if subnet in vulnerable_subnets or subnet == u.INTERNET:
                continue
            host_num = np.random.randint(size)
            host = self.hosts[(subnet, host_num)]
            self._update_host_to_vulnerable(host)
            vulnerable_subnets.add(subnet)

    def _host_is_vulnerable(self, host, access_level=u.USER_ACCESS):
        for e_def in self.exploits.values():
            if self._host_is_vulnerable_to_exploit(host, e_def):
                if e_def[u.EXPLOIT_ACCESS] >= access_level:
                    return True
                for pe_def in self.privescs.values():
                    if self._host_is_vulnerable_to_privesc(host, pe_def):
                        return True
        return False

    def _host_is_vulnerable_to_exploit(self, host, exploit_def):
        e_srv = exploit_def[u.EXPLOIT_SERVICE]
        e_os = exploit_def[u.EXPLOIT_OS]
        if not host.services[e_srv]:
            return False
        return e_os is None or host.os[e_os]

    def _host_is_vulnerable_to_privesc(self, host, privesc_def):
        pe_proc = privesc_def[u.PRIVESC_PROCESS]
        pe_os = privesc_def[u.PRIVESC_OS]
        if not host.processes[pe_proc]:
            return False
        return pe_os is None or host.os[pe_os]

    def _update_host_to_vulnerable(self, host, access_level=u.USER_ACCESS):
        """Update host config so it's vulnerable to at least one exploit """
        # choose an exploit randomly and make host vulnerable to it
        # will retry X times before giving up
        # If vulnerable config is not found in X tries then the scenario
        # probably needs more options (processes, privesc actions)
        for i in range(VUL_RETRIES):
            success, e_def = self._update_host_exploit_vulnerability(
                host, False
            )
            # don't need to check success since should always succeed
            # in finding exploit, when there is no contraint on OS
            if e_def[u.EXPLOIT_ACCESS] >= access_level:
                return
            # Need to ensure host is now vulnerable to >= 1 privesc action
            success, pe_def = self._update_host_privesc_vulnerability(
                host, True
            )
            if success:
                return

        raise AssertionError(
            "After {VUL_RETRIES}, unable to find privilege escalation action"
            " for target OS, when looking for vulnerable host configuration,"
            " try again using more privilege escalation actions or processes"
        )

    def _update_host_exploit_vulnerability(self, host, os_constraint):
        # choose an exploit randomly and make host vulnerable to it
        if not os_constraint:
            # can change host OS, so all exploits valid
            valid_e = list(self.exploits.values())
        else:
            # exploits must match OS of host, or be OS agnostic
            # since cannot change host OS
            valid_e = []
            for e_def in self.exploits.values():
                e_os = e_def[u.EXPLOIT_OS]
                if e_os is None or host.os[e_os]:
                    valid_e.append(e_def)

            if len(valid_e) == 0:
                return False, None

        e_def = np.random.choice(valid_e)
        host.services[e_def[u.EXPLOIT_SERVICE]] = True
        if e_def[u.EXPLOIT_OS] is not None and not os_constraint:
            self._update_host_os(host, e_def[u.EXPLOIT_OS])

        return True, e_def

    def _update_host_privesc_vulnerability(self, host, os_constraint):
        # choose an exploit randomly and make host vulnerable to it
        if not os_constraint:
            # no OS constraint
            valid_pe = list(self.privescs.values())
        else:
            valid_pe = []
            for pe_def in self.privescs.values():
                pe_os = pe_def[u.PRIVESC_OS]
                if pe_os is None or host.os[pe_os]:
                    valid_pe.append(pe_def)

            if len(valid_pe) == 0:
                return False, None

        pe_def = np.random.choice(valid_pe)
        host.processes[pe_def[u.PRIVESC_PROCESS]] = True
        if pe_def[u.PRIVESC_OS] is not None and not os_constraint:
            self._update_host_os(host, pe_def[u.PRIVESC_OS])

        return True, pe_def

    def _update_host_os(self, host, os):
        # must set all to false first, so only one host OS is true
        for os_name in host.os.keys():
            host.os[os_name] = False
        host.os[os] = True

    def _get_host_value(self, address):
        return float(self.sensitive_hosts.get(address, self.base_host_value))

    def _generate_firewall(self, restrictiveness):
        """Generate the firewall rules.

        Parameters
        ----------
        restrictiveness : int
            parameter that controls how many services are blocked by
            firewall between zones (i.e. between internet, DMZ, sensitive
            and user zones).

        Returns
        -------
        dict
            firewall rules that are a mapping from (src, dest) connection to
            set of allowed services, which defines for each service whether
            traffic using that service is allowed between pairs of subnets.

        Notes
        -----
        Traffic from at least one service running on each subnet will be
        allowed between each zone. This may mean more services will be allowed
        than restrictiveness parameter.
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
                # else src and dest in different zones => block services based
                # on restrictiveness
                dest_avail = subnet_services[dest].copy()
                if len(dest_avail) < restrictiveness:
                    # restrictiveness not limiting allowed traffic, all
                    # services allowed
                    firewall[(src, dest)] = dest_avail.copy()
                    continue
                # add at least one service to allowed service
                dest_allowed = np.random.choice(list(dest_avail))
                # for dest subnet choose available services upto
                # restrictiveness limit or all services
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
