import numpy as np

from .action import ActionResult
from .utils import get_minimal_steps_to_goal, min_subnet_depth, AccessLevel

# column in topology adjacency matrix that represents connection between
# subnet and public
INTERNET = 0


class Network:
    """A computer network """

    def __init__(self, scenario):
        self.hosts = scenario.hosts
        self.host_num_map = scenario.host_num_map
        self.subnets = scenario.subnets
        self.topology = scenario.topology
        self.firewall = scenario.firewall
        self.address_space = scenario.address_space
        self.address_space_bounds = scenario.address_space_bounds
        self.sensitive_addresses = scenario.sensitive_addresses
        self.sensitive_hosts = scenario.sensitive_hosts

    def reset(self, state):
        """Reset the network state to initial state """
        next_state = state.copy()
        for host_addr in self.address_space:
            host = next_state.get_host(host_addr)
            host.compromised = False
            host.access = AccessLevel.NONE
            host.reachable = self.subnet_public(host_addr[0])
            host.discovered = host.reachable
        return next_state

    def perform_action(self, state, action):
        """Perform the given Action against the network.

        Arguments
        ---------
        state : State
            the current state
        action : Action
            the action to perform

        Returns
        -------
        State
            the state after the action is performed
        ActionObservation
            the result from the action
        """
        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        next_state = state.copy()

        if action.is_noop():
            return next_state, ActionResult(True)

        if not state.host_reachable(action.target) \
           or not state.host_discovered(action.target):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        has_req_permission = self.has_required_remote_permission(state, action)
        if action.is_remote() and not has_req_permission:
            result = ActionResult(False, 0.0, permission_error=True)
            return next_state, result

        if action.is_exploit() \
           and not self.traffic_permitted(
                    state, action.target, action.service
           ):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        host_compromised = state.host_compromised(action.target)
        if action.is_privilege_escalation() and not host_compromised:
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if action.is_exploit() and host_compromised:
            # host already compromised so exploits don't fail due to randomness
            pass
        elif np.random.rand() > action.prob:
            return next_state, ActionResult(False, 0.0, undefined_error=True)

        if action.is_subnet_scan():
            return self._perform_subnet_scan(next_state, action)

        t_host = state.get_host(action.target)
        next_host_state, action_obs = t_host.perform_action(action)
        next_state.update_host(action.target, next_host_state)
        self._update(next_state, action, action_obs)
        return next_state, action_obs

    def _perform_subnet_scan(self, next_state, action):
        if not next_state.host_compromised(action.target):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if not next_state.host_has_access(action.target, action.req_access):
            result = ActionResult(False, 0.0, permission_error=True)
            return next_state, result

        discovered = {}
        newly_discovered = {}
        discovery_reward = 0
        target_subnet = action.target[0]
        for h_addr in self.address_space:
            newly_discovered[h_addr] = False
            discovered[h_addr] = False
            if self.subnets_connected(target_subnet, h_addr[0]):
                host = next_state.get_host(h_addr)
                discovered[h_addr] = True
                if not host.discovered:
                    newly_discovered[h_addr] = True
                    host.discovered = True
                    discovery_reward += host.discovery_value

        obs = ActionResult(
            True,
            discovery_reward,
            discovered=discovered,
            newly_discovered=newly_discovered
        )
        return next_state, obs

    def _update(self, state, action, action_obs):
        if action.is_exploit() and action_obs.success:
            self._update_reachable(state, action.target)

    def _update_reachable(self, state, compromised_addr):
        """Updates the reachable status of hosts on network, based on current
        state and newly exploited host
        """
        comp_subnet = compromised_addr[0]
        for addr in self.address_space:
            if state.host_reachable(addr):
                continue
            if self.subnets_connected(comp_subnet, addr[0]):
                state.set_host_reachable(addr)

    def get_sensitive_hosts(self):
        return self.sensitive_addresses

    def is_sensitive_host(self, host_address):
        return host_address in self.sensitive_addresses

    def subnets_connected(self, subnet_1, subnet_2):
        return self.topology[subnet_1][subnet_2] == 1

    def subnet_traffic_permitted(self, src_subnet, dest_subnet, service):
        if src_subnet == dest_subnet:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src_subnet, dest_subnet):
            return False
        return service in self.firewall[(src_subnet, dest_subnet)]

    def host_traffic_permitted(self, src_addr, dest_addr, service):
        dest_host = self.hosts[dest_addr]
        return dest_host.traffic_permitted(src_addr, service)

    def has_required_remote_permission(self, state, action):
        """Checks attacker has necessary permissions for remote action """
        if self.subnet_public(action.target[0]):
            return True

        for src_addr in self.address_space:
            if not state.host_compromised(src_addr):
                continue
            if action.is_scan() and \
               not self.subnets_connected(src_addr[0], action.target[0]):
                continue
            if action.is_exploit() and \
               not self.subnet_traffic_permitted(
                   src_addr[0], action.target[0], action.service
               ):
                continue
            if state.host_has_access(src_addr, action.req_access):
                return True
        return False

    def traffic_permitted(self, state, host_addr, service):
        """Checks whether the subnet and host firewalls permits traffic to a
        given host and service, based on current set of compromised hosts on
        network.
        """
        for src_addr in self.address_space:
            if not state.host_compromised(src_addr) and \
               not self.subnet_public(src_addr[0]):
                continue
            if not self.subnet_traffic_permitted(
                    src_addr[0], host_addr[0], service
            ):
                continue
            if self.host_traffic_permitted(src_addr, host_addr, service):
                return True
        return False

    def subnet_public(self, subnet):
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
        return len(self.subnets)

    def all_sensitive_hosts_compromised(self, state):
        for host_addr in self.sensitive_addresses:
            if not state.host_has_access(host_addr, AccessLevel.ROOT):
                return False
        return True

    def get_total_sensitive_host_value(self):
        total = 0
        for host_value in self.sensitive_hosts.values():
            total += host_value
        return total

    def get_total_discovery_value(self):
        total = 0
        for host in self.hosts:
            total += host.discovery_value
        return total

    def get_minimal_steps(self):
        return get_minimal_steps_to_goal(
            self.topology, self.sensitive_addresses
        )

    def get_subnet_depths(self):
        return min_subnet_depth(self.topology)

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
