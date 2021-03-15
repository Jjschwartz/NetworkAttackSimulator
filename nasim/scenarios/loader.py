"""This module contains functionality for loading network scenarios from yaml
files.
"""
import math

import nasim.scenarios.utils as u
from nasim.scenarios import Scenario
from nasim.scenarios.host import Host


# dictionary of valid key names and value types for config file
VALID_CONFIG_KEYS = {
    u.SUBNETS: list,
    u.TOPOLOGY: list,
    u.SENSITIVE_HOSTS: dict,
    u.OS: list,
    u.SERVICES: list,
    u.PROCESSES: list,
    u.EXPLOITS: dict,
    u.PRIVESCS: dict,
    u.SERVICE_SCAN_COST: (int, float),
    u.SUBNET_SCAN_COST: (int, float),
    u.OS_SCAN_COST: (int, float),
    u.PROCESS_SCAN_COST: (int, float),
    u.HOST_CONFIGS: dict,
    u.FIREWALL: dict
}

OPTIONAL_CONFIG_KEYS = {u.STEP_LIMIT: int}

VALID_ACCESS_VALUES = ["user", "root", u.USER_ACCESS, u.ROOT_ACCESS]
ACCESS_LEVEL_MAP = {
    "user": u.USER_ACCESS,
    "root": u.ROOT_ACCESS
}


# required keys for exploits
EXPLOIT_KEYS = {
    u.EXPLOIT_SERVICE: str,
    u.EXPLOIT_OS: str,
    u.EXPLOIT_PROB: (int, float),
    u.EXPLOIT_COST: (int, float),
    u.EXPLOIT_ACCESS: (str, int)
}

# required keys for privesc actions
PRIVESC_KEYS = {
    u.PRIVESC_OS: str,
    u.PRIVESC_PROCESS: str,
    u.PRIVESC_PROB: (int, float),
    u.PRIVESC_COST: (int, float),
    u.PRIVESC_ACCESS: (str, int)
}

# required keys for host configs
HOST_CONFIG_KEYS = {
    u.HOST_OS: (str, None),
    u.HOST_SERVICES: list,
    u.HOST_PROCESSES: list
}


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
        self._parse_os()
        self._parse_services()
        self._parse_processes()
        self._parse_sensitive_hosts()
        self._parse_exploits()
        self._parse_privescs()
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
        scenario_dict[u.OS] = self.os
        scenario_dict[u.SERVICES] = self.services
        scenario_dict[u.PROCESSES] = self.processes
        scenario_dict[u.SENSITIVE_HOSTS] = self.sensitive_hosts
        scenario_dict[u.EXPLOITS] = self.exploits
        scenario_dict[u.PRIVESCS] = self.privescs
        scenario_dict[u.OS_SCAN_COST] = self.os_scan_cost
        scenario_dict[u.SERVICE_SCAN_COST] = self.service_scan_cost
        scenario_dict[u.SUBNET_SCAN_COST] = self.subnet_scan_cost
        scenario_dict[u.PROCESS_SCAN_COST] = self.process_scan_cost
        scenario_dict[u.FIREWALL] = self.firewall
        scenario_dict[u.HOSTS] = self.hosts
        scenario_dict[u.STEP_LIMIT] = self.step_limit
        return Scenario(
            scenario_dict, name=self.name, generated=False
        )

    def _check_scenario_sections_valid(self):
        """Checks if scenario dictionary contains all required sections and
        they are valid type.
        """
        # 0. check correct number of keys
        assert len(self.yaml_dict) >= len(VALID_CONFIG_KEYS), \
            (f"Too few config file keys: {len(self.yaml_dict)} "
             f"< {len(VALID_CONFIG_KEYS)}")

        # 1. check keys are valid and values are correct type
        for k, v in self.yaml_dict.items():
            assert k in VALID_CONFIG_KEYS or k in OPTIONAL_CONFIG_KEYS, \
                f"{k} not a valid config file key"

            if k in VALID_CONFIG_KEYS:
                expected_type = VALID_CONFIG_KEYS[k]
            else:
                expected_type = OPTIONAL_CONFIG_KEYS[k]

            assert isinstance(v, expected_type), \
                (f"{v} invalid type for config file key '{k}': {type(v)}"
                 f" != {expected_type}")

    def _parse_subnets(self):
        subnets = self.yaml_dict[u.SUBNETS]
        self._validate_subnets(subnets)
        # insert internet subnet
        subnets.insert(0, 1)
        self.subnets = subnets
        self.num_hosts = sum(subnets)-1

    def _validate_subnets(self, subnets):
        # check subnets is valid list of positive ints
        assert len(subnets) > 0, "Subnets cannot be empty list"
        for subnet_size in subnets:
            assert type(subnet_size) is int and subnet_size > 0, \
                f"{subnet_size} invalid subnet size, must be positive int"

    def _parse_topology(self):
        topology = self.yaml_dict[u.TOPOLOGY]
        self._validate_topology(topology)
        self.topology = topology

    def _validate_topology(self, topology):
        # check topology is valid adjacency matrix
        assert len(topology) == len(self.subnets), \
            ("Number of rows in topology adjacency matrix must equal "
             f"number of subnets: {len(topology)} != {len(self.subnets)}")

        for row in topology:
            assert isinstance(row, list), \
                "topology must be 2D adjacency matrix (i.e. list of lists)"
            assert len(row) == len(self.subnets), \
                ("Number of columns in topology matrix must equal number of"
                 f" subnets: {len(topology)} != {len(self.subnets)}")
            for col in row:
                assert isinstance(col, int) and (col == 1 or col == 0), \
                    ("Subnet_connections adjaceny matrix must contain only"
                     f" 1 (connected) or 0 (not connected): {col} invalid")

    def _parse_os(self):
        os = self.yaml_dict[u.OS]
        self._validate_os(os)
        self.os = os

    def _validate_os(self, os):
        assert len(os) > 0, \
            f"{len(os)}. Invalid number of OSs, must be >= 1"
        assert len(os) == len(set(os)), \
            f"{os}. OSs must not contain duplicates"

    def _parse_services(self):
        services = self.yaml_dict[u.SERVICES]
        self._validate_services(services)
        self.services = services

    def _validate_services(self, services):
        assert len(services) > 0, \
           f"{len(services)}. Invalid number of services, must be > 0"
        assert len(services) == len(set(services)), \
            f"{services}. Services must not contain duplicates"

    def _parse_processes(self):
        processes = self.yaml_dict[u.PROCESSES]
        self._validate_processes(processes)
        self.processes = processes

    def _validate_processes(self, processes):
        assert len(processes) >= 1, \
            f"{len(processes)}. Invalid number of services, must be > 0"
        assert len(processes) == len(set(processes)), \
            f"{processes}. Processes must not contain duplicates"

    def _parse_sensitive_hosts(self):
        sensitive_hosts = self.yaml_dict[u.SENSITIVE_HOSTS]
        self._validate_sensitive_hosts(sensitive_hosts)

        self.sensitive_hosts = dict()
        for address, value in sensitive_hosts.items():
            self.sensitive_hosts[eval(address)] = value

    def _validate_sensitive_hosts(self, sensitive_hosts):
        # check sensitive_hosts is valid dict of (subnet, id) : value
        assert len(sensitive_hosts) > 0, \
            ("Number of sensitive hosts must be >= 1: "
             f"{len(sensitive_hosts)} not >= 1")

        assert len(sensitive_hosts) <= self.num_hosts, \
            ("Number of sensitive hosts must be <= total number of "
             f"hosts: {len(sensitive_hosts)} not <= {self.num_hosts}")

        # sensitive hosts must be valid address
        for address, value in sensitive_hosts.items():
            subnet_id, host_id = eval(address)
            assert self._is_valid_subnet_ID(subnet_id), \
                ("Invalid sensitive host tuple: subnet_id must be a valid"
                 f" subnet: {subnet_id} != non-negative int less than "
                 f"{len(self.subnets) + 1}")

            assert self._is_valid_host_address(subnet_id, host_id), \
                ("Invalid sensitive host tuple: host_id must be a valid"
                 f" int: {host_id} != non-negative int less than"
                 f" {self.subnets[subnet_id]}")

            assert isinstance(value, (float, int)) and value > 0, \
                (f"Invalid sensitive host tuple: invalid value: {value}"
                 f" != a positive int or float")

        # 5.c sensitive hosts must not contain duplicate addresses
        for i, m in enumerate(sensitive_hosts.keys()):
            h1_addr = eval(m)
            for j, n in enumerate(sensitive_hosts.keys()):
                if i == j:
                    continue
                h2_addr = eval(n)
                assert h1_addr != h2_addr, \
                    ("Sensitive hosts list must not contain duplicate host "
                     f"addresses: {m} == {n}")

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
        assert isinstance(e, dict), \
            f"{e_name}. Exploit must be a dict."

        for k, t in EXPLOIT_KEYS.items():
            assert k in e, f"{e_name}. Exploit missing key: '{k}'"
            assert isinstance(e[k], t), \
                f"{e_name}. Exploit '{k}' incorrect type. Expected {t}"

        assert e[u.EXPLOIT_SERVICE] in self.services, \
            (f"{e_name}. Exploit target service invalid: "
             f"'{e[u.EXPLOIT_SERVICE]}'")

        if str(e[u.EXPLOIT_OS]).lower() == "none":
            e[u.EXPLOIT_OS] = None

        assert e[u.EXPLOIT_OS] is None or e[u.EXPLOIT_OS] in self.os, \
            (f"{e_name}. Exploit target OS is invalid. '{e[u.EXPLOIT_OS]}'."
             " Should be None or one of the OS in the os list.")

        assert 0 <= e[u.EXPLOIT_PROB] < 1, \
            (f"{e_name}. Exploit probability, '{e[u.EXPLOIT_PROB]}' not "
             "a valid probability")

        assert e[u.EXPLOIT_COST] > 0, f"{e_name}. Exploit cost must be > 0."

        assert e[u.EXPLOIT_ACCESS] in VALID_ACCESS_VALUES, \
            (f"{e_name}. Exploit access value '{e[u.EXPLOIT_ACCESS]}' "
             f"invalid. Must be one of {VALID_ACCESS_VALUES}")

        if isinstance(e[u.EXPLOIT_ACCESS], str):
            e[u.EXPLOIT_ACCESS] = ACCESS_LEVEL_MAP[e[u.EXPLOIT_ACCESS]]

    def _parse_privescs(self):
        self.privescs = self.yaml_dict[u.PRIVESCS]
        self._validate_privescs(self.privescs)

    def _validate_privescs(self, privescs):
        for pe_name, pe in privescs.items():
            self._validate_single_privesc(pe_name, pe)

    def _validate_single_privesc(self, pe_name, pe):
        s_name = "Priviledge Escalation"

        assert isinstance(pe, dict), f"{pe_name}. {s_name} must be a dict."

        for k, t in PRIVESC_KEYS.items():
            assert k in pe, f"{pe_name}. {s_name} missing key: '{k}'"
            assert isinstance(pe[k], t), \
                (f"{pe_name}. {s_name} '{k}' incorrect type. Expected {t}")

        assert pe[u.PRIVESC_PROCESS] in self.processes, \
            (f"{pe_name}. {s_name} target process invalid: "
             f"'{pe[u.PRIVESC_PROCESS]}'")

        if str(pe[u.PRIVESC_OS]).lower() == "none":
            pe[u.PRIVESC_OS] = None

        assert pe[u.PRIVESC_OS] is None or pe[u.PRIVESC_OS] in self.os, \
            (f"{pe_name}. {s_name} target OS is invalid. '{pe[u.PRIVESC_OS]}'."
             f" Should be None or one of the OS in the os list.")

        assert 0 <= pe[u.PRIVESC_PROB] <= 1.0, \
            (f"{pe_name}. {s_name} probability, '{pe[u.PRIVESC_PROB]}' not "
                "a valid probability")

        assert pe[u.PRIVESC_COST] > 0, \
            f"{pe_name}. {s_name} cost must be > 0."

        assert pe[u.PRIVESC_ACCESS] in VALID_ACCESS_VALUES, \
            (f"{pe_name}. {s_name} access value '{pe[u.PRIVESC_ACCESS]}' "
             f"invalid. Must be one of {VALID_ACCESS_VALUES}")

        if isinstance(pe[u.PRIVESC_ACCESS], str):
            pe[u.PRIVESC_ACCESS] = ACCESS_LEVEL_MAP[pe[u.PRIVESC_ACCESS]]

    def _parse_scan_costs(self):
        self.os_scan_cost = self.yaml_dict[u.OS_SCAN_COST]
        self.service_scan_cost = self.yaml_dict[u.SERVICE_SCAN_COST]
        self.subnet_scan_cost = self.yaml_dict[u.SUBNET_SCAN_COST]
        self.process_scan_cost = self.yaml_dict[u.PROCESS_SCAN_COST]
        for (n, c) in [
                ("OS", self.os_scan_cost),
                ("Service", self.service_scan_cost),
                ("Subnet", self.subnet_scan_cost),
                ("Process", self.process_scan_cost)
        ]:
            self._validate_scan_cost(n, c)

    def _validate_scan_cost(self, scan_name, scan_cost):
        assert scan_cost >= 0, f"{scan_name} Scan Cost must be >= 0."

    def _parse_host_configs(self):
        self.host_configs = self.yaml_dict[u.HOST_CONFIGS]
        self._validate_host_configs(self.host_configs)

    def _validate_host_configs(self, host_configs):
        assert len(host_configs) == self.num_hosts, \
            ("Number of host configurations must match the number of hosts "
             f"in network: {len(host_configs)} != {self.num_hosts}")

        assert self._has_all_host_addresses(host_configs.keys()), \
            ("Host configurations must have no duplicates and have an"
             " address for each host on network.")

        for addr, cfg in host_configs.items():
            self._validate_host_config(addr, cfg)

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

    def _validate_host_config(self, addr, cfg):
        """Check if a host config is valid or not given the list of exploits available
        N.B. each host config must contain at least one service
        """
        err_prefix = f"Host {addr}"
        assert isinstance(cfg, dict) and len(cfg) >= len(HOST_CONFIG_KEYS), \
            (f"{err_prefix} configurations must be a dict of length >= "
             f"{len(HOST_CONFIG_KEYS)}. {cfg} is invalid")

        for k in HOST_CONFIG_KEYS:
            assert k in cfg, f"{err_prefix} configuration missing key: {k}"

        host_services = cfg[u.HOST_SERVICES]
        for service in host_services:
            assert service in self.services, \
                (f"{err_prefix} Invalid service in configuration services "
                 f"list: {service}")

        assert len(host_services) == len(set(host_services)), \
            (f"{err_prefix} configuration services list cannot contain "
             "duplicates")

        host_processes = cfg[u.HOST_PROCESSES]
        for process in host_processes:
            assert process in self.processes, \
                (f"{err_prefix} invalid process in configuration processes"
                 f" list: {process}")

        assert len(host_processes) == len(set(host_processes)), \
            (f"{err_prefix} configuation processes list cannot contain "
             "duplicates")

        host_os = cfg[u.HOST_OS]
        assert host_os in self.os, \
            f"{err_prefix} invalid os in configuration: {host_os}"

        fw_err_prefix = f"{err_prefix} {u.HOST_FIREWALL}"
        if u.HOST_FIREWALL in cfg:
            firewall = cfg[u.HOST_FIREWALL]
            assert isinstance(firewall, dict), \
                (f"{fw_err_prefix} must be a dictionary, with host "
                 "addresses as keys and a list of denied services as values. "
                 f"{firewall} is invalid.")
            for addr, srv_list in firewall.items():
                addr = self._validate_host_address(addr, err_prefix)
                assert self._is_valid_firewall_setting(srv_list), \
                    (f"{fw_err_prefix} setting must be a list, contain only "
                     f"valid services and contain no duplicates: {srv_list}"
                     " is not valid")
        else:
            cfg[u.HOST_FIREWALL] = dict()

        v_err_prefix = f"{err_prefix} {u.HOST_VALUE}"
        if u.HOST_VALUE in cfg:
            host_value = cfg[u.HOST_VALUE]
            assert isinstance(host_value, (int, float)), \
                (f"{v_err_prefix} must be an integer or float value. "
                 f"{host_value} is invalid")

            if addr in self.sensitive_hosts:
                sh_value = self.sensitive_hosts[addr]
                assert math.isclose(host_value, sh_value), \
                    (f"{v_err_prefix} for a sensitive host must either match "
                     f"the value specified in the {u.SENSITIVE_HOSTS} section "
                     f"or be excluded the host config. The value {host_value} "
                     f"is invalid as it does not match value {sh_value}.")

    def _validate_host_address(self, addr, err_prefix=""):
        try:
            addr = eval(addr)
        except Exception:
            raise AssertionError(
                f"{err_prefix} address invalid. Must be (subnet, host) tuple"
                f" of integers. {addr} is invalid."
            )
        assert isinstance(addr, tuple) \
            and len(addr) == 2 \
            and all([isinstance(a, int) for a in addr]), \
            (f"{err_prefix} address invalid. Must be (subnet, host) tuple"
             f" of integers. {addr} is invalid.")
        assert 0 < addr[0] < len(self.subnets), \
            (f"{err_prefix} address invalid. Subnet address must be in range"
             f" 0 < subnet addr < {len(self.subnets)}. {addr[0]} is invalid.")
        assert 0 <= addr[1] < self.subnets[addr[0]], \
            (f"{err_prefix} address invalid. Host address must be in range "
             f"0 < host addr < {self.subnets[addr[0]]}. {addr[1]} is invalid.")
        return True

    def _parse_firewall(self):
        firewall = self.yaml_dict[u.FIREWALL]
        self._validate_firewall(firewall)
        # convert (subnet_id, subnet_id) string to tuple
        self.firewall = {}
        for connect, v in firewall.items():
            self.firewall[eval(connect)] = v

    def _validate_firewall(self, firewall):
        assert self._contains_all_required_firewalls(firewall), \
            ("Firewall dictionary must contain two entries for each subnet "
             "connection in network (including from outside) as defined by "
             "network topology matrix")

        for f in firewall.values():
            assert self._is_valid_firewall_setting(f), \
                ("Firewall setting must be a list, contain only valid "
                 f"services and contain no duplicates: {f} is not valid")

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
            os_cfg, srv_cfg, proc_cfg = self._construct_host_config(h_cfg)
            value = self._get_host_value(formatted_address, h_cfg)
            hosts[formatted_address] = Host(
                address=formatted_address,
                os=os_cfg,
                services=srv_cfg,
                processes=proc_cfg,
                firewall=h_cfg[u.HOST_FIREWALL],
                value=value
            )
        self.hosts = hosts

    def _construct_host_config(self, host_cfg):
        os_cfg = {}
        for os_name in self.os:
            os_cfg[os_name] = os_name == host_cfg[u.HOST_OS]
        services_cfg = {}
        for service in self.services:
            services_cfg[service] = service in host_cfg[u.HOST_SERVICES]
        processes_cfg = {}
        for process in self.processes:
            processes_cfg[process] = process in host_cfg[u.HOST_PROCESSES]
        return os_cfg, services_cfg, processes_cfg

    def _get_host_value(self, address, host_cfg):
        if address in self.sensitive_hosts:
            return float(self.sensitive_hosts[address])
        return float(host_cfg.get(u.HOST_VALUE, u.DEFAULT_HOST_VALUE))

    def _parse_step_limit(self):
        if u.STEP_LIMIT not in self.yaml_dict:
            step_limit = None
        else:
            step_limit = self.yaml_dict[u.STEP_LIMIT]
            assert step_limit > 0, \
                f"Step limit must be positive int: {step_limit} is invalid"

        self.step_limit = step_limit
