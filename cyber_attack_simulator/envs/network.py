import numpy as np
from collections import OrderedDict
from cyber_attack_simulator.envs.util import permutations
from cyber_attack_simulator.envs.machine import Machine


class Network(object):
    """
    A simulated network of machines belonging to different subnetworks.

    Properties:
    - subnets - list of subnet sizes
    - topology - adjacency matrix defining connectivity between subnets and
        public
    - services - number of possible services running on each machine
    - sensitive_machines - list of addresses of machines that contain senstive
        info and also the value (reward) of accessing info
    """

    def __init__(self, config, seed=1):
        """
        Construct a new network based on provided configuration.

        Configuration must be valid and contain items:
            - subnets
            - topology
            - services
            - sensitive_machines

        Arguments:
            dict config : network configuration
            int seed : random number generator seed
        """
        self.config = config

        self.subnets = config["subnets"]
        self.topology = config["topology"]
        self.num_services = config["services"]
        self.sensitive_machines = config["sensitive_machines"]
        self.machines = self.generate_network(seed)
        self.sensitive_addresses = self._get_sensitive_addresses()

    def generate_network(self, seed):
        """
        Generate the network.

        Argument:
            int seed : random number generator seed

        Returns:
            dict machine : ordered dictionary of machines in network, with
                address as keys and machine objects as values
        """
        # set seed for consistency of networks generated
        np.random.seed(seed)
        machines = OrderedDict()

        machine_config_set = self._possible_machine_configs(self.num_services)
        num_configs = machine_config_set.shape[0]

        for subnet, size in enumerate(self.subnets):
            for m in range(size):
                cfg = machine_config_set[np.random.choice(num_configs)]
                address = (subnet, m)
                value = self._get_machine_value(address)
                machine = Machine(address, cfg, value)
                machines[address] = machine
        return machines

    def _possible_machine_configs(self, ns):
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
        return np.asarray(configs)

    def _get_machine_value(self, address):
        """
        Get the value of machine at given address
        """
        for m in self.sensitive_machines:
            if m[0] == address[0] and m[1] == address[1]:
                return float(m[2])
        return 0.0

    def perform_action(self, action):
        """
        Perform the given Action against the network.

        Arguments:
            Action exploit : the exploit Action

        Returns:
            bool success : True if action was successful, False otherwise
                (i.e. False if exploit failed)
            float value : value gained from action (0 if unsuccessful or scan),
                otherwise value of machine
            list services : the list of services identified by action. This is
                the services if exploit was successful or scan, otherwise an
                empty list
        """
        # check if valid target machine
        tgt_subnet, tgt_id = action.target
        assert 0 <= tgt_subnet and tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        # check if valid action type and service
        if not action.is_scan():
            assert 0 <= action.service and action.service < self.num_services

        # action is valid, so perform against machine
        t_machine = self.machines[action.target]
        return t_machine.perform_action(action)

    def get_address_space(self):
        """
        Get a list of all machine addresses in network

        Returns:
            list address_space : a list of all machine addresses
        """
        return list(self.machines.keys())

    def _get_sensitive_addresses(self):
        """
        Get addresses of machines which contain sensitive machines, to store
        for later efficiency
        """
        sensitive_addresses = []
        for m in self.sensitive_machines:
            sensitive_addresses.append((m[0], m[1]))
        return sensitive_addresses

    def get_sensitive_machines(self):
        """
        Get addresses of machines which contain sensitive information (rewards)

        Returns:
            list sensitive_addresses : a list of addresses of sensitive
                machines in network
        """
        return self.sensitive_addresses

    def is_sensitive_machine(self, m):
        """
        Returns whether a given machine is sensitive or not

        Arguments:
            (int, int) m : machine address

        Returns:
            bool is_sensitive : True if machine is sensitive, False otherwise
        """
        return m in self.sensitive_addresses

    def subnets_connected(self, subnet_1, subnet_2):
        """
        Checks whether two subnets are directly connected. A subnet is also
        connected to itself.

        Arguments:
            int subnet_1 : the id of first subnet
            int subnet_2 : the id of second subnet

        Returns:
            bool connected : True if subnets are directly connected
        """
        # plus one for subnet 2 since first column is for public or not
        return self.topology[subnet_1][subnet_2 + 1] == 1

    def subnet_exposed(self, subnet):
        """
        Returns whether a subnet is exposed to the public or not, i.e. is on
        public network and so always reachable by attacker.

        Arguments:
            int subnet : the id of subnet

        Returns:
            bool exposed : True if subnet is publicly exposed
        """
        return self.topology[subnet][0] == 1

    def __str__(self):
        output = "Network:\n"
        output += "Subnets = " + str(self.subnets) + "\n"
        output += "Topology =\n" + str(self.topology) + "\n"
        output += "Services = " + str(self.num_services) + "\n"
        output += "Sensitive machines = " + str(self.sensitive_machines)
        return output
