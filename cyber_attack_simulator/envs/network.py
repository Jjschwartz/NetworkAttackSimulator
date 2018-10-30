from collections import deque
from itertools import permutations
import numpy as np


# column in topology adjacency matrix that represents connection between subnet and public
INTERNET = 0


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
    - machines - ordered dictionary of machines in network, with address as keys and machine
        objects as values
    - firewall - a 3D matrix defining which services are allowed between source and destination
        subnets
    """

    def __init__(self, config):
        """
        Construct a new network based on provided configuration.

        Configuration must be valid and contain items:
            - subnets
            - topology
            - services
            - sensitive_machines
            - machines
            - firewall

        Arguments:
            dict config : network configuration
        """
        self.subnets = config["subnets"]
        self.topology = config["topology"]
        self.num_services = config["services"]
        self.sensitive_machines = config["sensitive_machines"]
        self.machines = config["machines"]
        self.firewall = config["firewall"]
        self.sensitive_addresses = self._get_sensitive_addresses()

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
        assert 0 < tgt_subnet and tgt_subnet < len(self.subnets)
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

    def get_machine_value(self, m):
        """
        Returns the value of a machine

        Arguments:
            (int, int) m : machine address

        Returns:
            float value : the value of machine with address m
        """
        return self.machines[m].get_value()

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

    def traffic_permitted(self, src, dest, service):
        """
        Checks whether traffic using a given service is permitted by the firewall from source
        subnet to destination subnet.

        Arguments:
            int src : id of source subnet
            int dest : id of destination subnet
            int service : service id

        Returns:
            bool permitted : True if traffic is permitted, False otherwise
        """
        return self.firewall[src][dest][service]

    def subnet_exposed(self, subnet):
        """
        Returns whether a subnet is exposed to the public or not, i.e. is in
        publicly acces DMZ and so always reachable by attacker.

        Arguments:
            int subnet : the id of subnet

        Returns:
            bool exposed : True if subnet is publicly exposed
        """
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
        """
        Returns the number of subnets on network, including the internet subnet

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

    def get_minimal_steps(self):
        """
        Get the minimum total number of steps required to reach all sensitive machines in the
        network starting from outside the network (i.e. can only reach exposed subnets).

        Returns:
            int minumum_steps : minimum number of steps to reach all sensitive machines
        """
        num_subnets = len(self.topology)
        max_value = np.iinfo(np.int32).max
        distance = np.full((num_subnets, num_subnets), max_value, dtype=np.int32)

        # set distances for each edge to 1
        for s1 in range(num_subnets):
            for s2 in range(num_subnets):
                if s1 == s2:
                    distance[s1][s2] = 0
                elif self.topology[s1][s2] == 1:
                    distance[s1][s2] = 1

        # find all pair minimum shortest path distance
        for k in range(num_subnets):
            for i in range(num_subnets):
                for j in range(num_subnets):
                    if distance[i][j] > distance[i][k] + distance[k][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]

        # get list of all subnets we need to visit
        subnets_to_visit = [INTERNET]
        for subnet, machine in self.sensitive_addresses:
            if subnet not in subnets_to_visit:
                subnets_to_visit.append(subnet)

        # find minimum shortest path that visits internet subnet and all sensitive subnets
        # by checking all possible permutations
        shortest = max_value
        for pm in permutations(subnets_to_visit):
            pm_sum = 0
            for i in range(len(pm) - 1):
                pm_sum += distance[pm[i]][pm[i+1]]
            shortest = min(shortest, pm_sum)

        return shortest

    def _get_sensitive_addresses(self):
        """
        Get addresses of machines which contain sensitive machines, to store
        for later efficiency
        """
        sensitive_addresses = []
        for m in self.sensitive_machines:
            sensitive_addresses.append((m[0], m[1]))
        return sensitive_addresses

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
        2D matrix topology : An adjacency matrix representing the network, with first subnet
                             representing the internet (i.e. exposed)

    Returns:
        list depths : a list of depth of each subnet ordered by subnet index in topology
    """
    num_subnets = len(topology)

    assert len(topology[0]) == num_subnets

    depths = []
    Q = deque()
    for subnet in range(num_subnets):
        if topology[subnet][INTERNET] == 1:
            depths.append(0)
            Q.appendleft(subnet)
        else:
            depths.append(float('inf'))

    while len(Q) > 0:
        parent = Q.pop()
        for child in range(num_subnets):
            if topology[parent][child] == 1:
                # child is connected to parent
                if depths[child] > depths[parent] + 1:
                    depths[child] = depths[parent] + 1
                    Q.appendleft(child)
    return depths
