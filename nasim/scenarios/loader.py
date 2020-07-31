"""This module contains functionality for loading network scenarios from yaml
files.
"""
import nasim.scenarios.utils as u
from nasim.scenarios import Scenario
from nasim.scenarios.host import Host


# dictionary of valid key names and value types for config file
VALID_CONFIG_KEYS = {u.SUBNETS: list,
                     u.TOPOLOGY: list,
                     u.SENSITIVE_HOSTS: dict,
                     u.SERVICES: list,
                     u.OS: list,
                     u.EXPLOITS: dict,
                     u.SERVICE_SCAN_COST: int,
                     u.SUBNET_SCAN_COST: int,
                     u.OS_SCAN_COST: int,
                     u.HOST_CONFIGS: dict,
                     u.FIREWALL: dict}

OPTIONAL_CONFIG_KEYS = {u.STEP_LIMIT: int}


# required keys for exploits
EXPLOIT_KEYS = {u.EXPLOIT_SERVICE: str,
                u.EXPLOIT_OS: str,
                u.EXPLOIT_PROB: float,
                u.EXPLOIT_COST: (int, float)}

# required keys for host configs
HOST_CONFIG_KEYS = {u.HOST_SERVICES: list,
                    u.HOST_OS: (str, None)}


class ScenarioLoader:

    def load(self, file_path, name=None):
        """Load the scenario from file

        Arguments
        ---------
        file_path : str
            path to scenario file
        name : str, optional
            the scenarios name, if None name will be generated from file path
            (default=None)

        Returns
        -------
        scenario_dict : dict
            dictionary with scenario definition

        Raises
        ------
        Exception
            If file unable to load or scenario file is invalid.
        """
        self.yaml_dict = u.load_yaml(file_path)
        if name is None:
            name = u.get_file_name(file_path)
        self.name = name
        self._check_scenario_sections_valid()

        self._parse_subnets()
        self._parse_topology()
        self._parse_services()
        self._parse_os()
        self._parse_sensitive_hosts()
        self._parse_exploits()
        self._parse_scan_costs()
        self._parse_host_configs()
        self._parse_firewall()
        self._parse_hosts()
        self._parse_step_limit()
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
        scenario_dict[u.SUBNET_SCAN_COST] = self.subnet_scan_cost
        scenario_dict[u.FIREWALL] = self.firewall
        scenario_dict[u.HOSTS] = self.hosts
        scenario_dict[u.STEP_LIMIT] = self.step_limit
        return Scenario(scenario_dict, name=self.name)

    def _check_scenario_sections_valid(self):
        """Checks if scenario dictionary contains all required sections and
        they are valid type.
        """
        # 0. check correct number of keys
        if len(self.yaml_dict) < len(VALID_CONFIG_KEYS):
            raise KeyError(f"Too few config file keys: "
                           f"{len(self.yaml_dict)} < {len(VALID_CONFIG_KEYS)}")

        # 1. check keys are valid and values are correct type
        for k, v in self.yaml_dict.items():
            if k not in VALID_CONFIG_KEYS and k not in OPTIONAL_CONFIG_KEYS:
                raise KeyError(f"{k} not a valid config file key")
            if k in VALID_CONFIG_KEYS:
                expected_type = VALID_CONFIG_KEYS[k]
            else:
                expected_type = OPTIONAL_CONFIG_KEYS[k]
            if type(v) is not expected_type:
                raise TypeError(f"{v} invalid type for config file key '{k}': "
                                f"{type(v)} != {expected_type}")

    def _parse_subnets(self):
        subnets = self.yaml_dict[u.SUBNETS]
        self._validate_subnets(subnets)
        # insert internet subnet
        subnets.insert(0, 1)
        self.subnets = subnets
        self.num_hosts = sum(subnets)-1

    def _validate_subnets(self, subnets):
        # check subnets is valid list of positive ints
        if len(subnets) < 1:
            raise ValueError("Subnets connot be empty list")
        for subnet_size in subnets:
            if type(subnet_size) is not int or subnet_size <= 0:
                raise ValueError(
                    f"{subnet_size} invalid subnet size, must be positive int"
                )

    def _parse_topology(self):
        topology = self.yaml_dict[u.TOPOLOGY]
        self._validate_topology(topology)
        self.topology = topology

    def _validate_topology(self, topology):
        # check topology is valid adjacency matrix
        if len(topology) != len(self.subnets):
            raise ValueError(
                "Number of rows in topology adjacency matrix must equal "
                f"number of subnets: {len(topology)} != {len(self.subnets)}")
        for row in topology:
            if type(row) is not list:
                raise ValueError(
                    "topology must be 2D adjacency matrix (i.e. list of lists)"
                )
            if len(row) != len(self.subnets):
                raise ValueError(
                    "Number of columns in topology matrix must equal number of"
                    f" subnets: {len(topology)} != {len(self.subnets)}"
                )
            for col in row:
                if type(col) is not int or (col != 1 and col != 0):
                    raise ValueError(
                        "Subnet_connections adjaceny matrix must contain only"
                        f" 1 (connected) or 0 (not connected): {col} invalid"
                    )

    def _parse_services(self):
        services = self.yaml_dict[u.SERVICES]
        self._validate_services(services)
        self.services = services

    def _validate_services(self, services):
        # check services is postive int
        if len(services) < 1:
            raise ValueError(
                f"{len(services)}. Invalid number of services, must be >= 1"
            )
        if len(services) < len(set(services)):
            raise ValueError(
                f"{services}. Services must not contain duplicates"
            )

    def _parse_os(self):
        os = self.yaml_dict[u.OS]
        self._validate_os(os)
        self.os = os

    def _validate_os(self, os):
        # check services is postive int
        if len(os) < 1:
            raise ValueError(f"{len(os)}. Invalid number of OSs, must be >= 1")
        if len(os) < len(set(os)):
            raise ValueError(f"{os}. OSs must not contain duplicates")

    def _parse_sensitive_hosts(self):
        sensitive_hosts = self.yaml_dict[u.SENSITIVE_HOSTS]
        self._validate_sensitive_hosts(sensitive_hosts)

        self.sensitive_hosts = dict()
        for address, value in sensitive_hosts.items():
            self.sensitive_hosts[eval(address)] = value

    def _validate_sensitive_hosts(self, sensitive_hosts):
        # check sensitive_hosts is valid dict of (subnet, id) : value
        if len(sensitive_hosts) < 1:
            raise ValueError(
                "Number of sensitive hosts must be >= 1: "
                f"{len(sensitive_hosts)} not >= 1"
            )
        if len(sensitive_hosts) > self.num_hosts:
            raise ValueError(
                "Number of sensitive hosts must be <= total number of "
                f"hosts: {len(sensitive_hosts)} not <= {self.num_hosts}"
            )

        # sensitive hosts must be valid address
        for address, value in sensitive_hosts.items():
            subnet_id, host_id = eval(address)
            if not self._is_valid_subnet_ID(subnet_id):
                raise ValueError(
                    "Invalid sensitive host tuple: subnet_id must be a valid"
                    f" subnet: {subnet_id} != non-negative int less than "
                    f"{len(self.subnets) + 1}"
                )
            if not self._is_valid_host_address(subnet_id, host_id):
                raise ValueError(
                    "Invalid sensitive host tuple: host_id must be a valid"
                    f" int: {host_id} != non-negative int less than"
                    f" {self.subnets[subnet_id]}"
                )
            if not isinstance(value, (float, int)) or value <= 0:
                raise ValueError(
                    f"Invalid sensitive host tuple: invalid value: {value}"
                    f" != a positive int or float"
                )

        # 5.c sensitive hosts must not contain duplicate addresses
        for i, m in enumerate(sensitive_hosts.keys()):
            h1_addr = eval(m)
            for j, n in enumerate(sensitive_hosts.keys()):
                h2_addr = eval(n)
                if i != j and h1_addr == h2_addr:
                    raise ValueError(
                        "Sensitive hosts list must not contain duplicate host "
                        f"addresses: {m} == {n}"
                    )

    def _is_valid_subnet_ID(self, subnet_ID):
        if type(subnet_ID) is not int \
           or subnet_ID < 1 \
           or subnet_ID > len(self.subnets):
            return False
        return True

    def _is_valid_host_address(self, subnet_ID, host_ID):
        if not self._is_valid_subnet_ID(subnet_ID):
            return False
        if type(host_ID) is not int \
           or host_ID < 0 \
           or host_ID >= self.subnets[subnet_ID]:
            return False
        return True

    def _parse_exploits(self):
        exploits = self.yaml_dict[u.EXPLOITS]
        self._validate_exploits(exploits)
        self.exploits = exploits

    def _validate_exploits(self, exploits):
        for e_name, e in exploits.items():
            self._validate_single_exploit(e_name, e)

    def _validate_single_exploit(self, e_name, e):
        if not isinstance(e, dict):
            raise ValueError(f"{e_name}. Exploit must be a dict.")
        for k, t in EXPLOIT_KEYS.items():
            if k not in e:
                raise ValueError(f"{e_name}. Exploit missing key: '{k}'")
            if not isinstance(e[k], t):
                raise ValueError(
                    f"{e_name}. Exploit '{k}' incorrect type. Expected {t}"
                )
        if e[u.EXPLOIT_SERVICE] not in self.services:
            raise ValueError(
                f"{e_name}. Exploit target service invalid: "
                f"'{e[u.EXPLOIT_SERVICE]}'"
            )

        if str(e[u.EXPLOIT_OS]).lower() == "none":
            e[u.EXPLOIT_OS] = None
        if e[u.EXPLOIT_OS] is not None and e[u.EXPLOIT_OS] not in self.os:
            raise ValueError(
                f"{e_name}. Exploit target OS is invalid. "
                f"'{e[u.EXPLOIT_OS]}'. Should be None or one of the OS in"
                " the os list."
            )
        if e[u.EXPLOIT_PROB] < 0 or 1 < e[u.EXPLOIT_PROB]:
            raise ValueError(
                f"{e_name}. Exploit probability, '{e[u.EXPLOIT_PROB]}' not "
                "a valid probability"
            )
        if e[u.EXPLOIT_COST] < 0:
            raise ValueError(f"{e_name}. Exploit cost must be > 0.")

    def _parse_scan_costs(self):
        service_scan_cost = self.yaml_dict[u.SERVICE_SCAN_COST]
        os_scan_cost = self.yaml_dict[u.OS_SCAN_COST]
        subnet_scan_cost = self.yaml_dict[u.SUBNET_SCAN_COST]
        self._validate_scan_cost(
            service_scan_cost, os_scan_cost, subnet_scan_cost
        )
        self.service_scan_cost = service_scan_cost
        self.os_scan_cost = os_scan_cost
        self.subnet_scan_cost = subnet_scan_cost

    def _validate_scan_cost(self,
                            service_scan_cost,
                            os_scan_cost,
                            subnet_scan_cost):
        if service_scan_cost < 0:
            raise ValueError("Service Scan Cost must be >= 0.")
        if os_scan_cost < 0:
            raise ValueError("OS Scan Cost must be >= 0.")
        if subnet_scan_cost < 0:
            raise ValueError("Subnet Scan Cost must be >= 0.")

    def _parse_host_configs(self):
        host_configs = self.yaml_dict[u.HOST_CONFIGS]
        self._validate_host_configs(host_configs)
        self.host_configs = host_configs

    def _validate_host_configs(self, host_configs):
        if len(host_configs) != self.num_hosts:
            raise ValueError(
                "Number of host configurations must match the number of hosts "
                f"in network: {len(host_configs)} != {self.num_hosts}"
            )
        if not self._has_all_host_addresses(host_configs.keys()):
            raise ValueError(
                "Host configurations must have no duplicates and have an"
                " address for each host on network."
            )
        for cfg in host_configs.values():
            self._validate_host_config(cfg)

    def _has_all_host_addresses(self, addresses):
        """Check that list of (subnet_ID, host_ID) tuples contains all
        addresses on network based on subnets list
        """
        for s_id, s_size in enumerate(self.subnets[1:]):
            for m in range(s_size):
                # +1 to s_id since first subnet is 1
                if str((s_id + 1, m)) not in addresses:
                    return False
        return True

    def _validate_host_config(self, cfg):
        """Check if a host config is valid or not given the list of exploits available
        N.B. each host config must contain at least one service
        """
        if not isinstance(cfg, dict) or len(cfg) != 2:
            raise ValueError(
                f"Host configurations must be at dict of length 2 {cfg} is"
                " invalid"
            )

        for k in HOST_CONFIG_KEYS:
            if k not in cfg:
                raise ValueError(f"Host configuation missing key: {k}")

        host_services = cfg[u.HOST_SERVICES]
        for service in host_services:
            if service not in self.services:
                raise ValueError(
                    "Invalid service in host configuration services list:"
                    f" {service}"
                )
        if len(host_services) < len(set(host_services)):
            raise ValueError(
                f"Host configuation services list cannot contain duplicates"
            )

        host_os = cfg[u.HOST_OS]
        if host_os not in self.os:
            raise ValueError(f"Invalid os in host configuration: {host_os}")

    def _parse_firewall(self):
        firewall = self.yaml_dict[u.FIREWALL]
        self._validate_firewall(firewall)
        # convert (subnet_id, subnet_id) string to tuple
        self.firewall = {}
        for connect, v in firewall.items():
            self.firewall[eval(connect)] = v

    def _validate_firewall(self, firewall):
        if not self._contains_all_required_firewalls(firewall):
            raise ValueError(
                "Firewall dictionary must contain two entries for each subnet "
                "connection in network (including from outside) as defined by "
                "network topology matrix"
            )
        for f in firewall.values():
            if not self._is_valid_firewall_setting(f):
                raise ValueError(
                    "Firewall setting must be a list, contain only valid "
                    "services and contain no duplicates: {f} is not valid"
                )

    def _contains_all_required_firewalls(self, firewall):
        for src, row in enumerate(self.topology):
            for dest, col in enumerate(row):
                if src == dest:
                    continue
                if col == 1 and (str((src, dest)) not in firewall
                                 or str((dest, src)) not in firewall):
                    return False
        return True

    def _is_valid_firewall_setting(self, f):
        if type(f) != list:
            return False
        for service in f:
            if service not in self.services:
                return False
        for i, x in enumerate(f):
            for j, y in enumerate(f):
                if i != j and x == y:
                    return False
        return True

    def _parse_hosts(self):
        """Returns ordered dictionary of hosts in network, with address as
        keys and host objects as values
        """
        hosts = dict()
        for address, h_cfg in self.host_configs.items():
            formatted_address = eval(address)
            services_cfg, os_cfg = self._construct_host_config(h_cfg)
            value = self._get_host_value(formatted_address)
            hosts[formatted_address] = Host(formatted_address,
                                            os_cfg,
                                            services_cfg,
                                            value)
        self.hosts = hosts

    def _construct_host_config(self, host_cfg):
        services_cfg = {}
        for service in self.services:
            services_cfg[service] = service in host_cfg[u.HOST_SERVICES]
        os_cfg = {}
        for os_name in self.os:
            os_cfg[os_name] = os_name == host_cfg[u.HOST_OS]
        return services_cfg, os_cfg

    def _get_host_value(self, address):
        return float(self.sensitive_hosts.get(address, 0.0))

    def _parse_step_limit(self):
        if u.STEP_LIMIT not in self.yaml_dict:
            step_limit = None
        else:
            step_limit = self.yaml_dict[u.STEP_LIMIT]
            if step_limit <= 0:
                raise ValueError(
                    f"Step limit must be positive int: {step_limit} is invalid"
                )
        self.step_limit = step_limit
