import numpy as np
from cyber_attack_simulator.envs.util import permutations
from cyber_attack_simulator.envs.machine import Machine


USER_SUBNET_SIZE = 5
EXPOSED = 0
SENSITIVE = 1


class Network(object):
    """
    A simulated network of machines belonging to different subnetworks.

    Number of subnetworks is set to , with machines distributed across each
    subnet following a set rule:
        - exposed (subnet 0) - with one machine
        - sensitive (subnet 1) - with one machine
        - user (subnet 2+) - all other machines distributed in tree of subnets
            with each subnet containing up to 5 machines.

    Two machines on network contain Sensitive documents (aka the rewards):
        - One in sensitive subnet machine = r_sensitive
        - One on machine in leaf subnet of user = r_user
        - any machine with no sensitive docs = 0

    The configurations of each machine (i.e. which services are present/absent)
    are allocated randomly but deterministically, so that a network initialized
    with the same number of machines and services will always produce the same
    network. This is done to keep consistency for agents trained on a given
    network size.

    Properties:
    - subnetworks - dictionary of subnets and the list of all machines on that
        subnet

    Note: The network and machines defined in the Network class are designed as
    immutable with the intention all attack logic (e.g whether an action can
    be used against a specific machine) is controlled by the environment, (i.e
    the CyberAttackSimulatorEnv class).
    """

    def __init__(self, num_machines, num_services, r_sensitive=9000.0,
                 r_user=5000.0, seed=1):
        """
        Initialize and generate a new Network with given number of machines and
        services for each machine

        Arguments:
            int num_machines : number of machines to include in network
                (minimum is 3)
            int num_exploits : number of exploits (and hence services) to use
                in environment (minimum is 1)
            float r_sensitive : reward for sensitive subnet documents
            float r_user : reward for user subnet documents
            int seed : random generator seed
        """
        self.num_machines = num_machines
        self.num_services = num_services
        self.r_sensitive = r_sensitive
        self.r_user = r_user
        self.seed = seed

        network = self._generate_network()
        self.subnets = network[0]
        self.address_space = network[1]
        self.reward_machines = network[2]

        self.topology = self._generate_topology()

    def _generate_network(self):
        """
        Generate the network.

        Arguments:
            list subnet_sizes : list of number of machines on each subnet
            float reward : value for sensitive documents on machine

        Returns:
            dict subnets : dictionary of lists of machine objects
            list address_space : list of machine addresses in network
            list rewarded_machiness : list of machine addresses that contain
                rewards
        """
        # set seed for consistency of networks generated
        np.random.seed(self.seed)

        subnets = {}
        address_space = []
        rewarded_machines = []

        configs = self._possible_machine_configs(self.num_services)
        subnet_sizes = self._get_subnet_sizes(self.num_machines)

        for subnet, size in enumerate(subnet_sizes):
            subnets[subnet] = []
            for m in range(size):
                cfg = configs[np.random.choice(configs.shape[0])]
                address = (subnet, m)
                address_space.append(address)
                value = 0
                if subnet == 1 and m == 0:
                    # machine on sensitive subnet
                    value = self.r_sensitive
                    rewarded_machines.append(address)
                elif subnet == len(subnet_sizes) - 1 and m == size - 1:
                    # last machine in last user subnet
                    value = self.r_user
                    rewarded_machines.append(address)
                subnets[subnet].append(Machine(address, cfg, value))
        return subnets, address_space, rewarded_machines

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

    def _get_subnet_sizes(self, nm):
        """
        Generate list of subnet sizes

        Argument:
            int nm : number of machines in network

        Returns:
            list subnet_sizes : list of number of machines in each subnet
        """
        # exposed (0) and sensitive (1) subnets both contain 1 machine
        subnet_sizes = [1, 1]
        subnet_sizes += [USER_SUBNET_SIZE] * ((nm - 2) // USER_SUBNET_SIZE)
        if ((nm - 2) % USER_SUBNET_SIZE) != 0:
            subnet_sizes.append((nm - 2) % USER_SUBNET_SIZE)
        return subnet_sizes

    def _generate_topology(self):
        """
        Generate the topology of the network, defining the connectivity between
        subnets.

        Returns:
            2D matrix topology : an adjacency matrix of subnets
        """
        topology = np.zeros((len(self.subnets), len(self.subnets)))
        # exposed subnet is connected to sensitive and first user subnet
        for i in range(3):
            for j in range(3):
                topology[i][j] = 1
        if len(self.subnets) == 3:
            return topology
        # all other subnets are part of user binary tree
        for i in range(2, len(self.subnets)):
            topology[i][i] = 1
            pos = i - 2
            if pos > 0:
                parent = ((pos - 1) // 2) + 2
                topology[i][parent] = 1
            child_left = ((2 * pos) + 1) + 2
            child_right = ((2 * pos) + 2) + 2
            if child_left < len(self.subnets):
                topology[i][child_left] = 1
            if child_right < len(self.subnets):
                topology[i][child_right] = 1
        return topology

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
        assert tgt_id <= len(self.subnets[tgt_subnet])

        # check if valid action type and service
        if not action.is_scan():
            assert 0 <= action.service and action.service < self.num_services

        # action is valid, so perform against machine
        t_machine = self.subnets[tgt_subnet][tgt_id]
        return t_machine.perform_action(action)

    def get_address_space(self):
        """
        Get a list of all machine addresses in network

        Returns:
            list address_space : a list of all machine addresses
        """
        return self.address_space

    def get_reward_machines(self):
        """
        Get addresses of machines which contain sensitive documents (rewards)

        Returns:
            list machines : a list of addresses of machines in network that
                contain sensitive documents
        """
        return self.reward_machines

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
        return self.topology[subnet_1][subnet_2] == 1

    def print_network(self):
        """
        Print a user friendly display of the network
        """
        output = "\n"
        for subnet, machines in self.subnets.items():
            if subnet == 0:
                output += "EXPOSED    "
            elif subnet == 1:
                output += "SENSITIVE  "
            else:
                output += "USER {0}     ".format(subnet - 2)
            output += "Subnet = {0} : {{".format(subnet)
            for m in machines:
                output += "{0}, ".format(m.address)
            output += "}\n"
        print(output)
