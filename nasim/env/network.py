import numpy as np
from collections import deque
from itertools import permutations

from .action_obs import ActionObservation

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
        self.compromised_subnets = None

    def reset(self):
        """Reset network to initial state.

        This only changes the compromised and reachable status of each host in network.
        """
        self.compromised_subnets = set([INTERNET])
        for host_addr, host in self.hosts.items():
            host.compromised = False
            host.reachable = self.subnet_public(host_addr[0])
            host.discovered = host.reachable

    def perform_action(self, action, fully_obs):
        """Perform the given Action against the network.

        Arguments
        ---------
        action : Action
            the action to perform

        Returns
        -------
        ActionObservation
            the result from the action
        """
        # check if valid target host
        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet and tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        if not self.host_reachable(action.target):
            # print("host not reachable")
            return ActionObservation(False, 0.0)

        if not self.host_discovered(action.target):
            # print("host not discovered")
            return ActionObservation(False, 0.0)

        if action.is_exploit() and not self.host_service_traffic_permitted(action.target, action.service):
            # print("Traffic not permitted")
            # exploit action and host and/or service blocked by firewall
            return ActionObservation(False, 0.0)

        if action.is_exploit() and self.host_compromised(action.target):
            # print("Host already compromised")
            # host already compromised so exploit will work
            t_host = self.hosts[action.target]
            return t_host.perform_action(action)

        # non-deterministic actions
        if np.random.rand() > action.prob:
            # print("Stochastic action fail")
            return ActionObservation(False, 0.0)

        # action is valid
        if action.is_subnet_scan():
            # print("Performing subnet scan")
            return self._perform_subnet_scan(action)

        t_host = self.hosts[action.target]
        action_obs = t_host.perform_action(action)
        self._update(action, action_obs, fully_obs)
        return action_obs

    def _perform_subnet_scan(self, action):
        if not self.host_compromised(action.target):
            # can only perform subnet scan from compromised host
            return ActionObservation(False, 0.0)

        discovered = {}
        target_subnet = action.target[0]
        for h_addr, host in self.hosts.items():
            if self.subnets_connected(target_subnet, h_addr[0]):
                discovered[h_addr] = True
                host.discovered = True
            else:
                discovered[h_addr] = False
        return ActionObservation(True, 0.0, discovered=discovered)

    def _update(self, action, action_obs, fully_obs):
        if action.is_exploit() and action_obs.success:
            self.compromised_subnets.add(action.target[0])
            self._update_reachable(action.target, fully_obs)

    def _update_reachable(self, compromised_addr, fully_obs):
        """Updates the reachable status of hosts on network, based on current state and newly
        exploited host
        """
        comp_subnet = compromised_addr[0]
        for addr in self.address_space:
            if self.host_reachable(addr):
                continue
            host_subnet = addr[0]
            if self.subnets_connected(comp_subnet, host_subnet):
                self.set_host_reachable(addr)
                if fully_obs:
                    self.set_host_discovered(addr)

    def get_sensitive_hosts(self):
        return self.sensitive_addresses

    def is_sensitive_host(self, host_address):
        return host_address in self.sensitive_addresses

    def host_reachable(self, host_addr):
        return self.hosts[host_addr].reachable

    def host_compromised(self, host_addr):
        return self.hosts[host_addr].compromised

    def host_discovered(self, host_addr):
        return self.hosts[host_addr].discovered

    def set_host_compromised(self, host_addr):
        self.hosts[host_addr].compromised = True

    def set_host_reachable(self, host_addr):
        self.hosts[host_addr].reachable = True

    def set_host_discovered(self, host_addr):
        self.hosts[host_addr].discovered = True

    def get_host_value(self, host_address):
        return self.hosts[host_address].get_value()

    def host_is_running_service(self, host_addr, service):
        return self.hosts[host_addr].service_present(service)

    def host_is_running_os(self, host_addr, os):
        return self.hosts[host_addr].is_running_os(os)

    def subnets_connected(self, subnet_1, subnet_2):
        return self.topology[subnet_1][subnet_2] == 1

    def subnet_traffic_permitted(self, src_subnet, dest_subnet, service):
        if src_subnet == dest_subnet:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src_subnet, dest_subnet):
            return False
        return service in self.firewall[(src_subnet, dest_subnet)]

    def host_service_traffic_permitted(self, host_addr, service):
        """Checks whether the firewall permits traffic to a given host and service,
        based on current set of compromised hosts on network.
        """
        for src in self.compromised_subnets:
            if self.subnet_traffic_permitted(src, host_addr[0], service):
                return True
        return False

    def subnet_public(self, subnet):
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
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
