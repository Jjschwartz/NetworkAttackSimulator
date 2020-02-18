import nasim.scenarios.utils as u
from nasim.scenarios.loader import ScenarioLoader
from nasim.scenarios.generator import ScenarioGenerator


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

    @classmethod
    def load_from_file(cls, path):
        """Load a scenario from scenario file.

        Arguments
        ---------
        path : str
            path to the scenario .yaml file

        Returns
        -------
        Scenario
            object containing scenario definiton
        """
        loader = ScenarioLoader()
        scenario_dict = loader.load(path)
        return cls(scenario_dict)

    @classmethod
    def generate(cls, num_hosts, num_services, **params):
        """Generate a scenario.

        Arguments
        ---------
        num_hosts : int
            number of hosts to include in network (minimum is 3)
        num_services : int
            number of services to use in environment (minimum is 1)
        params : dict
            generator params (see scenarios.generator for full list)

        Returns
        -------
        Scenario
            object containing scenario definiton
        """
        generator = ScenarioGenerator()
        scenario_dict = generator.generate(num_hosts, num_services, **params)
        return cls(scenario_dict)
