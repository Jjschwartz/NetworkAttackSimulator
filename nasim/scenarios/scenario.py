import math
from pprint import pprint

import nasim.scenarios.utils as u


class Scenario:

    def __init__(self, scenario_dict, name=None, generated=False):
        self.scenario_dict = scenario_dict
        self.name = name
        self.generated = generated
        self._e_map = None
        self._pe_map = None
        self._cred = None

        # this is used for consistent positioning of
        # host state and obs in state and obs matrices
        self.host_num_map = {}
        for host_num, host_addr in enumerate(self.hosts):
            self.host_num_map[host_addr] = host_num

    @property
    def step_limit(self):
        return self.scenario_dict.get(u.STEP_LIMIT, None)

    @property
    def services(self):
        return self.scenario_dict[u.SERVICES]

    @property
    def num_services(self):
        return len(self.services)
    
    @property
    def credentials(self):
        if self._cred is None:
            credentials = []
            for _, e_def in self.exploits.items():
                cred_needed = e_def[u.EXPLOIT_CREDENTIALS_NEEDED]
                serv_name = e_def[u.EXPLOIT_SERVICE]
                if cred_needed not in credentials:
                    credentials.append(cred_needed)
            self._cred = credentials
        return self._cred
    
    @property
    def num_cred(self):
        return len(self.credentials)

    @property
    def vul(self):
        return self.scenario_dict[u.VUL]

    @property
    def num_vul(self):
        return len(self.vul)

    @property
    def os(self):
        return self.scenario_dict[u.OS]

    @property
    def num_os(self):
        return len(self.os)

    @property
    def processes(self):
        return self.scenario_dict[u.PROCESSES]

    @property
    def num_processes(self):
        return len(self.processes)

    @property
    def access_levels(self):
        return u.ROOT_ACCESS

    @property
    def exploits(self):
        return self.scenario_dict[u.EXPLOITS]

    @property
    def privescs(self):
        return self.scenario_dict[u.PRIVESCS]

    @property
    def exploit_map(self):
        """A nested dictionary for all exploits in scenario.

            I.e. {service_name: {
             os_name: {
                 vul: {
                    name: e_name,
                    vul: e_vul,
                    cost: e_cost,
                    prob: e_prob,
                    access: e_access
                    cred_needed: e_cred_needed
                 }
             }
        """
        if self._e_map is None:
            e_map = {}
            for e_name, e_def in self.exploits.items():
                srv_name = e_def[u.EXPLOIT_SERVICE]
                if srv_name not in e_map:
                    e_map[srv_name] = {}
                srv_map = e_map[srv_name]

                os = e_def[u.EXPLOIT_OS]
                if os not in srv_map:
                    srv_map[os] = {}
                srv_map2 = srv_map[os]

                vul = e_def[u.EXPLOIT_VUL]
                if vul not in srv_map2:
                    srv_map2[vul] = {}
                srv_map3 = srv_map2[vul]
                
                cred_needed = e_def[u.EXPLOIT_CREDENTIALS_NEEDED]
                if cred_needed not in srv_map3:
                    srv_map3[vul] = {
                        "name": e_name,
                        u.EXPLOIT_SERVICE: srv_name,
                        u.EXPLOIT_OS: os,
                        u.EXPLOIT_VUL: vul,
                        u.EXPLOIT_COST: e_def[u.EXPLOIT_COST],
                        u.EXPLOIT_PROB: e_def[u.EXPLOIT_PROB],
                        u.EXPLOIT_ACCESS: e_def[u.EXPLOIT_ACCESS],
                        u.EXPLOIT_CREDENTIALS_NEEDED: e_def[u.EXPLOIT_CREDENTIALS_NEEDED]
                    }
            self._e_map = e_map
        return self._e_map

    @property
    def privesc_map(self):
        """A nested dictionary for all privilege escalation actions in scenario.

        I.e. {process_name: {
                 os_name: {
                     name: pe_name,
                     cost: pe_cost,
                     prob: pe_prob,
                     access: pe_access
                     credentials_tofind: pe_credentials_tofind
                 }
             }
        """
        if self._pe_map is None:
            pe_map = {}
            for pe_name, pe_def in self.privescs.items():
                proc_name = pe_def[u.PRIVESC_PROCESS]
                if proc_name not in pe_map:
                    pe_map[proc_name] = {}
                proc_map = pe_map[proc_name]

                os = pe_def[u.PRIVESC_OS]
                if os not in proc_map:
                    proc_map[os] = {
                        "name": pe_name,
                        u.PRIVESC_PROCESS: proc_name,
                        u.PRIVESC_OS: os,
                        u.PRIVESC_COST: pe_def[u.PRIVESC_COST],
                        u.PRIVESC_PROB: pe_def[u.PRIVESC_PROB],
                        u.PRIVESC_ACCESS: pe_def[u.PRIVESC_ACCESS],
                        u.PRIVESC_CREDENTIALS_TOFIND: pe_def[u.PRIVESC_CREDENTIALS_TOFIND]
                    }
            self._pe_map = pe_map
        return self._pe_map

    @property
    def subnets(self):
        return self.scenario_dict[u.SUBNETS]

    @property
    def topology(self):
        return self.scenario_dict[u.TOPOLOGY]

    @property
    def sensitive_hosts(self):
        return self.scenario_dict[u.SENSITIVE_HOSTS]

    @property
    def sensitive_addresses(self):
        return list(self.sensitive_hosts.keys())

    @property
    def firewall(self):
        return self.scenario_dict[u.FIREWALL]

    @property
    def hosts(self):
        return self.scenario_dict[u.HOSTS]

    @property
    def address_space(self):
        return list(self.hosts.keys())

    @property
    def service_scan_cost(self):
        return self.scenario_dict[u.SERVICE_SCAN_COST]

    @property
    def vul_scan_cost(self):
        return self.scenario_dict[u.VUL_SCAN_COST]

    @property
    def wiretapping_cost(self):
        return self.scenario_dict[u.WIRETAPPING_COST]

    @property
    def os_scan_cost(self):
        return self.scenario_dict[u.OS_SCAN_COST]

    @property
    def subnet_scan_cost(self):
        return self.scenario_dict[u.SUBNET_SCAN_COST]

    @property
    def process_scan_cost(self):
        return self.scenario_dict[u.PROCESS_SCAN_COST]

    @property
    def address_space_bounds(self):
        return len(self.subnets), max(self.subnets)

    @property
    def host_value_bounds(self):
        """The min and max values of host in scenario

        Returns
        -------
        (float, float)
            (min, max) tuple of host values
        """
        min_value = math.inf
        max_value = -math.inf
        for host in self.hosts.values():
            min_value = min(min_value, host.value)
            max_value = max(max_value, host.value)
        return (min_value, max_value)

    @property
    def host_discovery_value_bounds(self):
        """The min and max discovery values of hosts in scenario

        Returns
        -------
        (float, float)
            (min, max) tuple of host values
        """
        min_value = math.inf
        max_value = -math.inf
        for host in self.hosts.values():
            min_value = min(min_value, host.discovery_value)
            max_value = max(max_value, host.discovery_value)
        return (min_value, max_value)

    def display(self):
        pprint(self.scenario_dict)

    def get_action_space_size(self):
        num_exploits = len(self.exploits)
        num_privescs = len(self.privescs)
        num_vul = len(self.vul)
        # OSScan, ServiceScan, SubnetScan, ProcessScan
        num_scans = 4
        actions_per_host = num_exploits + num_privescs + num_scans + num_vul
        return len(self.hosts) * actions_per_host

    def get_state_space_size(self):
        # compromised, reachable, discovered
        host_aux_bin_features = 3
        num_bin_features = (
            host_aux_bin_features
            + self.num_os
            + self.num_services
            + self.num_vul
            + self.num_processes
        )
        # access
        num_tri_features = 1
        host_states = 2**num_bin_features * 3**num_tri_features
        return len(self.hosts) * host_states

    def get_state_dims(self):
        # compromised, reachable, discovered, value, discovery_value, access
        host_aux_features = 6
        host_state_size = (
            self.address_space_bounds[0]
            + self.address_space_bounds[1]
            + host_aux_features
            + self.num_os
            + self.num_services
            + self.num_vul
            + self.num_processes
        )
        return len(self.hosts), host_state_size

    def get_observation_dims(self):
        state_dims = self.get_state_dims()
        return state_dims[0]+1, state_dims[1]

    def get_description(self):
        description = {
            "Name": self.name,
            "Type": "generated" if self.generated else "static",
            "Subnets": len(self.subnets),
            "Hosts": len(self.hosts),
            "OS": self.num_os,
            "Services": self.num_services,
            "Vul": self.num_vul,
            "Processes": self.num_processes,
            "Exploits": len(self.exploits),
            "PrivEscs": len(self.privescs),
            "Actions": self.get_action_space_size(),
            "Observation Dims": self.get_observation_dims(),
            "States": self.get_state_space_size(),
            "Step Limit": self.step_limit
        }
        return description
