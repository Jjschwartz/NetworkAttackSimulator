import numpy as np
from machine import Machine
from util import permutations


class Network(object):
    """
    A simulated network of machines belonging to different subnetworks.

    Number of subnetworks is set to 3, with machines distributed across each
    subnet following a set rule:
        - exposed (subnet 1) - with one machine
        - sensitive (subnet 2) - with one machine for every 10 in user subnet
        - user (subnet 3) - all other machines

    Sensitive documents (aka the rewards) are also stored around network on
    every 10th machine (including 1st machine) on the sensitive and user
    subnets. Rewards are designated as follows:
        - sensitive machines with sensitive docs = +9000
        - user machines with sensitive docs = +5000
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
                 r_user=5000.0):
        """
        Initialize and generate a new Network with given number of machines and
        services for each machine

        Arguments:
        int num_machines : number of machines in network
        int num_services : number of services for machine configurations
        float r_sensitive : value for sensitive documents on machines on
            sensitive subnet
        float r_user : value for sensitive documents on machines on user subnet
        """
        self.num_machines = num_machines
        self.num_services = num_services
        self.r_sensitive = r_sensitive
        self.r_user = r_user
        self.sensitive_machines = []
        self.subnetworks = self._generate_network()

    def _generate_network(self):
        """
        Generate the network.

        Returns:
        dict subnets : dictionary of lists of machine objects
        """
        # set seed for consistency of networks generated
        np.random.seed(1)
        configs = self._possible_machine_configs(self.num_services)
        subnets = {1: [], 2: [], 3: []}
        # machine id within sensitive and user subnets
        s_id = 0
        u_id = 0
        for m in range(self.num_machines):
            cfg = configs[np.random.choice(configs.shape[0])]
            r = 0
            if m == 0:
                # exposed subnet
                subnets[1].append(Machine(1, 0, cfg))
            elif m % 11 == 2:
                # sensitive subnet
                # set reward for every 10th machine in subnet
                if s_id % 10 == 0:
                    r = self.r_sensitive
                    self.sensitive_machines.append((2, s_id))
                subnets[2].append(Machine(2, s_id, cfg, r))
                s_id += 1
            else:
                # user subnet
                # set reward for every 10th machine in subnet
                if u_id % 10 == 0:
                    r = self.r_user
                    self.sensitive_machines.append((3, u_id))
                subnets[3].append(Machine(3, u_id, cfg, r))
                u_id += 1
        return subnets

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
            each configuration is a list of bools corresponding to the presence
            or absence of a service
        """
        # remove last permutation which is all False
        configs = permutations(ns)[:-1]
        return np.asarray(configs)

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
            the services if exploit was successful or scan, otherwise an empty
            list
        """
        # check if valid target machine
        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet and tgt_subnet < 4
        assert tgt_id <= len(self.subnetworks[tgt_subnet])

        # check if valid action type and service
        if not action.is_scan():
            assert 0 <= action.service and action.service < self.num_services

        # action is valid, so perform against machine
        t_machine = self.subnetworks[tgt_subnet][tgt_id]
        return t_machine.perform_action(action)

    def get_machines(self):
        """
        Get a list of all machine addresses in network

        Returns:
            list machines : a list of all machine addresses, as (subnet, id)
                tuple, on the network
        """
        machines = []
        for subnet in self.subnetworks.values():
            for m in subnet:
                machines.append(m.address)
        return machines

    def get_sensitive_machines(self):
        """
        Get addresses of machines which contain sensitive documents

        Returns:
            list machines : a list of addresses of machines in network that
                contain sensitive documents
        """
        return self.sensitive_machines
