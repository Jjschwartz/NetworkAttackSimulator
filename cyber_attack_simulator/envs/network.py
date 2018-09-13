import numpy as np
from collections import OrderedDict
from collections import deque
from cyber_attack_simulator.envs.machine import Machine


# column in topology adjacency matrix that represents connection between subnet and public
EXPOSED_COL = 0


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

    def __init__(self, config, uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0, seed=1):
        """
        Construct a new network based on provided configuration.

        Configuration must be valid and contain items:
            - subnets
            - topology
            - services
            - sensitive_machines

        Machine Configuration distribution:
            1. if uniform=True
                => machine configurations are chosen uniformly at random from set of all valid
                   possible configurations
            2. if uniform=False
                => machine configurations are chosen to be corelated (see below)

        CORELATED CONFIGURATIONS:
        The distribution of configurations of each machine in the network are generated using a
        Nested Dirichlet Process, so that across the network machines will have corelated
        configurations (i.e. certain services/configurations will be more common across machines on
        the network), the degree of corelation is controlled by alpha_H and alpha_V, with lower
        values leading to greater corelation.

        lambda_V controls the average number of services running per machine. Higher values will
        mean more services (so more vulnerable) machines on average.

        Arguments:
            dict config : network configuration
            bool uniform : whether to use uniform distribution of machine configs or corelated
                           machine configs
            float alpha_H : (only used when uniform=False), scaling/concentration parameter for
                            controlling corelation between machine configurations (must be > 0)
            float alpha_V : (only used when uniform=False) scaling/concentration parameter for
                            controlling corelation between services across machine configurations
                            (must be > 0)
            float lambda_V : (only used when uniform=False) parameter for controlling average
                             number of services running per machine configuration (must be > 0)
            int seed : random number generator seed
        """
        assert 0 < alpha_H and 0 < alpha_V and 0 < lambda_V
        self.subnets = config["subnets"]
        self.topology = config["topology"]
        self.num_services = config["services"]
        self.sensitive_machines = config["sensitive_machines"]
        self.machines = self._generate_network(uniform, alpha_H, alpha_V, lambda_V, seed)
        self.sensitive_addresses = self._get_sensitive_addresses()

    def _generate_network(self, uniform, alpha_H, alpha_V, lambda_V, seed):
        """
        Generate the network.

        Argument:
            bool uniform : whether to use uniform distribution of machine configs or corelated
                           machine configs
            float alpha_H : scaling/concentration parameter for controlling corelation between
                            machine configurations
            float alpha_V : scaling/concentration parameter for controlling corelation between
                            services across machine configurations
            float lambda_V : parameter for controlling average number of services running per
                             machine configuration
            int seed : random number generator seed

        Returns:
            dict machine : ordered dictionary of machines in network, with
                address as keys and machine objects as values
        """
        # set seed for consistency of networks generated
        np.random.seed(seed)
        machines = OrderedDict()

        if uniform:
            machine_config_set = self._possible_machine_configs(self.num_services)
            num_configs = len(machine_config_set)
        else:
            prev_configs = []
            prev_vuls = []
            host_num = 0

        for subnet, size in enumerate(self.subnets):
            for m in range(size):
                if uniform:
                    cfg = machine_config_set[np.random.choice(num_configs)]
                else:
                    cfg = self._get_machine_config(host_num, alpha_H, prev_configs, alpha_V,
                                                   prev_vuls, lambda_V, self.num_services)
                    host_num += 1
                address = (subnet, m)
                value = self._get_machine_value(address)
                machine = Machine(address, cfg, value)
                machines[address] = machine
        return machines

    def _get_machine_config(self, host_num, alpha_H, prev_configs, alpha_V, prev_vuls, lambda_V,
                            num_services):

        if host_num == 0 or np.random.rand() < (alpha_H / (alpha_H + host_num - 1)):
            # if first host or with prob proportional to alpha_H choose new config
            new_config = self._sample_config(alpha_V, prev_vuls, lambda_V, num_services)
        else:
            # sample uniformly from previous sampled configs
            new_config = prev_configs[np.random.choice(len(prev_configs))]
        prev_configs.append(new_config)
        return new_config

    def _sample_config(self, alpha_V, prev_vuls, lambda_V, num_services):
        # no services present by default
        new_config = [False for i in range(num_services)]
        # randomly get number of times to sample using poission dist in range (0, num_services)
        # minimum 1 service running
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
        return self.topology[subnet][EXPOSED_COL] == 1

    def get_number_of_subnets(self):
        """
        Returns the number of subnets on network

        Returns:
            int num_subnets : number of subnets on network
        """
        return len(self.subnets)

    def get_subnet_depths(self):
        """
        Get the minumum depth of each subnet in the network graph in terms of steps from an exposed
        subnet to each subnet

        Returns:
            list depths : a list of depth of each subnet ordered by subnet index in topology
        """
        return min_subnet_depth(self.topology)

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
        return configs

    def _get_sensitive_addresses(self):
        """
        Get addresses of machines which contain sensitive machines, to store
        for later efficiency
        """
        sensitive_addresses = []
        for m in self.sensitive_machines:
            sensitive_addresses.append((m[0], m[1]))
        return sensitive_addresses

    def _get_machine_value(self, address):
        """
        Get the value of machine at given address
        """
        for m in self.sensitive_machines:
            if m[0] == address[0] and m[1] == address[1]:
                return float(m[2])
        return 0.0

    def __str__(self):
        output = "Network:\n"
        output += "Subnets = " + str(self.subnets) + "\n"
        output += "Topology =\n" + str(self.topology) + "\n"
        output += "Services = " + str(self.num_services) + "\n"
        output += "Sensitive machines = " + str(self.sensitive_machines)
        return output


def min_subnet_depth(topology):
    """
    Find the minumum depth of each subnet in the network graph in terms of steps from an exposed
    subnet to each subnet

    Arguments:
        2D matrix topology : An adjacency matrix representing the network, with first coloumn
                             representing connection between subnet and public

    Returns:
        list depths : a list of depth of each subnet ordered by subnet index in topology
    """
    num_subnets = len(topology)

    assert len(topology[0]) == num_subnets + 1

    depths = []
    Q = deque()
    for subnet in range(num_subnets):
        if topology[subnet][EXPOSED_COL] == 1:
            depths.append(0)
            Q.appendleft(subnet)
        else:
            depths.append(float('inf'))

    while len(Q) > 0:
        parent = Q.pop()
        for child in range(num_subnets):
            # +1 since one column for exposed connection
            if topology[parent][child + 1] == 1:
                # child is connected to parent
                if depths[child] > depths[parent] + 1:
                    depths[child] = depths[parent] + 1
                    Q.appendleft(child)
    return depths


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
