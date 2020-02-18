import nasim.scenarios.utils as u


class Scenario:

    def __init__(self, scenario_dict):
        self.scenario_dict = scenario_dict

    @property
    def services(self):
        return self.scenario_dict[u.SERVICES]

    @property
    def num_services(self):
        return len(self.services)

    @property
    def exploits(self):
        return self.scenario_dict[u.EXPLOITS]

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
    def firewall(self):
        return self.scenario_dict[u.FIREWALL]

    @property
    def hosts(self):
        return self.scenario_dict[u.HOSTS]

    @property
    def address_space(self):
        return list(self.hosts.keys())

    @property
    def scan_cost(self):
        return self.scenario_dict[u.SCAN_COST]
