import numpy as np
from collections import deque
from itertools import permutations


# column in topology adjacency matrix that represents connection between subnet and public
INTERNET = 0


class Network:
    """A simulated network of hosts belonging to different subnetworks.

    Properties:
    - subnets - list of subnet sizes
    - topology - adjacency matrix defining connectivity between subnets and
        public
    - services - number of possible services running on each host
    - sensitive_hosts - list of addresses of hosts that contain senstive
        info and also the value (reward) of accessing info
    - hosts - ordered dictionary of hosts in network, with address as keys and host
        objects as values
    - firewall - a 3D matrix defining which services are allowed between source and destination
        subnets
    """

    def __init__(self, scenario):
        """
        Arguments
        ---------
        scenario : Scenario
            scenario definition
        """
        self.scenario = scenario
        self.subnets = scenario.subnets
        self.topology = scenario.topology
        self.sensitive_hosts = scenario.sensitive_hosts
        self.hosts = scenario.hosts
        self.firewall = scenario.firewall
        self.address_space = scenario.address_space
        self.sensitive_addresses = self._get_sensitive_addresses()

    def reset(self):
        """Reset network to initial state.

        This only changes the compromised and reachable status of each host in network.
        """
        for host_addr, host in self.hosts.items():
            host.compromised = False
            host.reachable = self.subnet_public(host_addr[0])

    def perform_action(self, action):
        """Perform the given Action against the network.

        Arguments
        ---------
        action : Action
            the action to perform

        Returns
        -------
        success : bool
            True if action was successful, False otherwise (i.e. False if exploit failed)
        value : float
            value gained from action (0 if unsuccessful or scan), otherwise value of host
        services : list
            the list of services identified by action. This is the services if exploit
            was successful or scan, otherwise an empty list
        """
        # check if valid target host
        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet and tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        # action is valid, so perform against host
        t_host = self.hosts[action.target]
        return t_host.perform_action(action)

    def get_sensitive_hosts(self):
        """Get addresses of hosts which contain sensitive information (rewards)

        Returns
        -------
        sensitive_addresses : list
            a list of addresses of sensitive hosts in network
        """
        return self.sensitive_addresses

    def is_sensitive_host(self, host_address):
        """Returns whether a given host is sensitive or not

        Arguments
        ---------
        host_address : (int, int)
            host address

        Returns
        -------
        bool
            True if host is sensitive, False otherwise
        """
        return host_address in self.sensitive_addresses

    def reachable(self, host_addr):
        """Checks if a given host is reachable

        Arguments
        ---------
        host_addr : (int, int)
            the host address

        Returns
        -------
        bool
            True if reachable
        """
        return self.hosts[host_addr].reachable

    def compromised(self, host_addr):
        """Checks if a given host is compromised

        Arguments
        ---------
        target : (int, int)
            the host address

        Returns
        -------
        bool
            True if compromised
        """
        return self.hosts[host_addr].compromised

    def set_compromised(self, host_addr):
        """Set the target host state as compromised

        Arguments
        ---------
        host_addr : (int, int)
            the target host address
        """
        self.hosts[host_addr].compromised = True

    def set_reachable(self, host_addr):
        """Set the target host state as reachable

        Arguments
        ---------
        host_addr : (int, int)
            the target host address
        """
        self.hosts[host_addr].reachable = True

    def get_host_value(self, host_address):
        """Returns the value of a host

        Arguments
        ---------
        host_address : (int, int)
            host address

        Returns
        -------
        float
            the value of host with given address
        """
        return self.hosts[host_address].get_value()

    def host_running_service(self, host_addr, service):
        """Returns whether a host is running a service or not.

        Arguments
        ---------
        host_address : (int, int)
            host address
        service : str
            name of service

        Returns
        -------
        bool
            True if host is runnning service
        """
        return self.hosts[host_addr].service_present(service)

    def host_running_os(self, host_addr, os):
        """Returns OS of host

        Arguments
        ---------
        host_address : (int, int)
            host address
        os : str
            the host os

        Returns
        -------
        bool
            True if host is running given OS
        """
        return self.hosts[host_addr].is_running_os(os)

    def subnets_connected(self, subnet_1, subnet_2):
        """Checks whether two subnets are directly connected. A subnet is also
        connected to itself.

        Arguments
        ---------
        subnet_1 : int
            the id of first subnet
        subnet_2 : int
            the id of second subnet

        Returns
        -------
        bool
            True if subnets are directly connected
        """
        return self.topology[subnet_1][subnet_2] == 1

    def traffic_permitted(self, src, dest, service):
        """Checks whether traffic using a given service is permitted by the firewall
        from source subnet to destination subnet.

        Arguments
        ---------
        src : int
            id of source subnet
        dest : int
            id of destination subnet
        service : int
            service id

        Returns
        -------
        bool
            True if traffic is permitted, False otherwise
        """
        if src == dest:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src, dest):
            return False
        return service in self.firewall[(src, dest)]

    def subnet_public(self, subnet):
        """Returns whether a subnet is exposed to the public or not, i.e. is in
        publicly acces DMZ and so always reachable by attacker.

        Arguments
        ---------
        subnet : int
            the id of subnet

        Returns
        -------
        bool
            True if subnet is publicly exposed
        """
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
        """Returns the number of subnets on network, including the internet subnet

        Returns
        -------
        int
            number of subnets on network
        """
        return len(self.subnets)

    def get_subnet_depths(self):
        """Get the minumum depth of each subnet in the network graph in terms of
        steps from an exposed subnet to each subnet

        Returns
        -------
        depths : list
            a list of depth of each subnet ordered by subnet index in topology
        """
        return min_subnet_depth(self.topology)

    def get_minimal_steps(self):
        """Get the minimum total number of steps required to reach all sensitive
        hosts in the network starting from outside the network (i.e. can only
        reach exposed subnets).

        Returns
        -------
        int
            minimum number of steps to reach all sensitive hosts
        """
        num_subnets = len(self.topology)
        max_value = np.iinfo(np.int16).max
        distance = np.full((num_subnets, num_subnets), max_value, dtype=np.int16)

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
                    if distance[i][k] == max_value or distance[k][j] == max_value:
                        dis = max_value
                    else:
                        dis = distance[i][k] + distance[k][j]
                    if distance[i][j] > dis:
                        distance[i][j] = distance[i][k] + distance[k][j]

        # get list of all subnets we need to visit
        subnets_to_visit = [INTERNET]
        for subnet, host in self.sensitive_addresses:
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

    def get_total_sensitive_host_value(self):
        """Get the sum of the values of each sensitive host

        Returns
        -------
        float
            total value of each sensitive host on network
        """
        total_value = 0
        for host_value in self.sensitive_hosts.values():
            total_value += host_value
        return total_value

    def _get_sensitive_addresses(self):
        """Get addresses of hosts which contain sensitive hosts, to store
        for later efficiency
        """
        sensitive_addresses = []
        for h in self.sensitive_hosts.keys():
            sensitive_addresses.append(h)
        return sensitive_addresses

    def state_size(self):
        """Get the size of the state of the entire network

        N.B. This is sum of the size of all hosts on the network

        Returns
        -------
        int
            size of state network
        """
        # ther will always be a (1, 0), since it is the first host in the first subnet
        h0 = self.hosts[(1, 0)]
        return len(self.hosts)*h0.state_size

    def __str__(self):
        output = "\n--- Network ---\n"
        output += "Subnets: " + str(self.subnets) + "\n"
        output += "Topology:\n"
        for row in self.topology:
            output += f"\t{row}\n"
        output += "Sensitive hosts: \n"
        for addr, value in self.sensitive_hosts.items():
            output += f"\t{addr}: {value}\n"
        output += "Num_services: {self.scenario.num_services}\n"
        output += "Hosts:\n"
        for m in self.hosts.values():
            output += str(m) + "\n"
        output += "Firewall:\n"
        for c, a in self.firewall.items():
            output += f"\t{c}: {a}\n"
        return output


def min_subnet_depth(topology):
    """Find the minumum depth of each subnet in the network graph in terms of steps
    from an exposed subnet to each subnet

    Arguments
    ---------
    topology : 2D matrix
        An adjacency matrix representing the network, with first subnet representing
        the internet (i.e. exposed)

    Returns
    -------
    depths : list
        depth of each subnet ordered by subnet index in topology
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
