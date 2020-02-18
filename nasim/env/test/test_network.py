import unittest
import numpy as np

from nasim.env.action import Exploit, ServiceScan
from nasim.env.network import Network, min_subnet_depth
from nasim.scenarios.generator import ScenarioGenerator, DMZ, USER
from nasim.scenarios import Scenario, INTERNET

A_COST = 10


class NetworkTestCase(unittest.TestCase):

    def setUp(self):
        self.r_sensitive = 1000
        self.r_user = 1000
        self.M = 3
        self.S = 3
        self.network = self.get_network(self.M, self.S)

    def find_exploit(self, network, address, valid):
        services = network.hosts[address].services
        for srv, v in services.items():
            if v == valid:
                return srv

    def get_network(self, nM, nS, **params):
        generator = ScenarioGenerator()
        scenario_dict = generator.generate(nM,
                                           nS,
                                           r_sensitive=self.r_sensitive,
                                           r_user=self.r_user,
                                           **params)
        scenario = Scenario(scenario_dict)
        network = Network(scenario)
        return network

    def get_exploit(self, addr, srv):
        return Exploit(addr, A_COST, srv)

    def test_generate_network_small(self):
        M = 3
        S = 3
        network = self.get_network(M, S)
        subnets = network.subnets
        self.assertEqual(len(subnets), USER + 1)
        for s in range(USER + 1):
            self.assertEqual(subnets[s], 1)
            if s > INTERNET:
                self.assertTrue(network.hosts[(s, 0)].address, (s, 0))

    def test_generate_network_consistency(self):
        M = [3, 6]
        S = [3, 6]
        results1 = []
        hosts1 = []
        for m in M:
            for s in S:
                network = self.get_network(m, s, **{"seed": 0})
                results1.append(network.subnets)
                hosts1.append(network.hosts)
        results2 = []
        hosts2 = []
        for m in M:
            for s in S:
                network = self.get_network(m, s, **{"seed": 0})
                results2.append(network.subnets)
                hosts2.append(network.hosts)
        self.assertEqual(results1, results2)
        self.assertEqual(hosts1, hosts2)

    def test_successful_exploit(self):
        rewards = [0, self.r_sensitive, self.r_user]
        for i in range(DMZ, USER + 1):
            subnet = i
            srv = self.find_exploit(self.network, (subnet, 0), True)
            exploit = self.get_exploit((subnet, 0), srv)
            outcome, reward, services = self.network.perform_action(exploit)
            self.assertTrue(outcome)
            self.assertEqual(reward, rewards[i - 1])

    def test_unsuccessful_exploit(self):
        rewards = [0, 0, 0]
        exp_services = np.asarray([])
        for i in range(DMZ, USER + 1):
            subnet = i
            srv = self.find_exploit(self.network, (subnet, 0), False)
            if (srv is None):
                # host vulnerable to all exploits
                continue
            exploit = self.get_exploit((subnet, 0), srv)
            outcome, reward, services = self.network.perform_action(exploit)
            self.assertFalse(outcome)
            self.assertEqual(reward, rewards[i - 1])
            self.assertTrue((services == exp_services).all())

    def test_scan(self):
        rewards = [0, 0, 0]
        for i in range(DMZ, USER + 1):
            subnet = i
            exp_services = self.network.hosts[(subnet, 0)].services
            scan = ServiceScan((subnet, 0), A_COST)
            outcome, reward, services = self.network.perform_action(scan)
            self.assertTrue(outcome)
            self.assertEqual(reward, rewards[i - 1])
            self.assertTrue(services == exp_services)

    def test_invalid_action(self):
        # invalid subnet
        scan = ServiceScan((USER + 1, 0), A_COST)
        with self.assertRaises(AssertionError):
            self.network.perform_action(scan)
        # invalid host ID
        scan = ServiceScan((DMZ, 2), A_COST)
        with self.assertRaises(AssertionError):
            self.network.perform_action(scan)
        # invalid service
        exploit = self.get_exploit((DMZ, 0), self.S + 1)
        with self.assertRaises(KeyError):
            self.network.perform_action(exploit)

    def test_topology(self):
        m = 20
        s = 1
        network = self.get_network(m, s)
        # test public explosure of DMZ subnet
        self.assertTrue(network.subnet_public(DMZ))
        # test full connectivity of first 3 subnets
        for i in range(DMZ, USER + 1):
            for j in range(DMZ, USER + 1):
                self.assertTrue(network.subnets_connected(i, j))

        # test exposed and sensitive subnets not connected to sub user subnets
        for i in range(DMZ, USER):
            for j in range(USER + 1, 7):
                self.assertFalse(network.subnets_connected(i, j))
                self.assertFalse(network.subnets_connected(j, i))

        # test user subnet connections
        # 2 & 3 connected
        self.assertTrue(network.subnets_connected(USER, USER + 1))
        self.assertTrue(network.subnets_connected(USER + 1, USER))
        self.assertTrue(network.subnets_connected(USER + 1, USER + 1))
        # 2 & 4 connected
        self.assertTrue(network.subnets_connected(USER, USER + 2))
        self.assertTrue(network.subnets_connected(USER + 2, USER))
        self.assertTrue(network.subnets_connected(USER + 2, USER + 2))
        # 3 & 4 not connected
        self.assertFalse(network.subnets_connected(USER + 1, USER + 2))
        self.assertFalse(network.subnets_connected(USER + 2, USER + 1))
        # 3 & 5 connected
        self.assertTrue(network.subnets_connected(USER + 1, USER + 3))
        self.assertTrue(network.subnets_connected(USER + 3, USER + 1))
        self.assertTrue(network.subnets_connected(USER + 3, USER + 3))
        # 4 & 5 not connected
        self.assertFalse(network.subnets_connected(USER + 2, USER + 3))
        self.assertFalse(network.subnets_connected(USER + 3, USER + 2))

    def min_subnet_depth(self):
        topology = [[1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1]]

        expected_depths = [0, 0, 1, 1, 1, 0, 2, 2]
        actual_depths = min_subnet_depth(topology)
        self.assertEqual(actual_depths, expected_depths)

    def test_correllated_configs(self):
        nM = 40
        nS = 10
        alpha_H = [0.5, 1, 2.5, 5, 10]
        alpha_V = [0.5, 1, 2.5, 5, 10]
        lambda_V = [1, 2.5, 5, 10]

        # test generating network with uniform dist of configs
        print("Test Uniform dist of configurations")
        network = self.get_network(nM, nS, **{"uniform": True})
        num_configs, max_same = self.num_similiar_configs(network)
        print(f"\tUniform: num_configs={num_configs}, max_same_configs={max_same}")
        max_vulns, avg_vulns = self.num_exploits_avail(nS, nM, network)
        print(f"\tUniform: max_vulns={max_vulns}, avg_vulns={avg_vulns}")

        # for each compare number of similar configs for range of alpha_H
        print("Test alpha_H")
        for h in alpha_H:
            network = self.get_network(nM, nS, **{"alpha_H": h})
            num_configs, max_same = self.num_similiar_configs(network)
            print(f"\talpha_H={h}, num_configs={num_configs}, max_same_configs={max_same}")

        # for each compare number of similar configs and services for range of alpha_V
        print("Test alpha_V")
        for v in alpha_V:
            network = self.get_network(nM, nS, **{"alpha_V": v})
            num_configs, max_same = self.num_similiar_configs(network)
            print(f"\talpha_V={v}, num_configs={num_configs}, max_same_configs={max_same}")

        # for each compare number of services per host for range of lambda_V
        print("Test lambda_V")
        for l in lambda_V:
            network = self.get_network(nM, nS, **{"lambda_V": l})
            max_vulns, avg_vulns = self.num_exploits_avail(nS, nM, network)
            print(f"\tlambda_V={l}, max_vulns={max_vulns}, avg_vulns={avg_vulns}")

    def test_minimum_steps(self):
        nS = 1
        nM = 3
        network = self.get_network(nM, nS, **{"uniform": True})
        actual = network.get_minimal_steps()
        expected = 3
        self.assertEqual(actual, expected)

        nM = 8
        network = self.get_network(nM, nS, **{"uniform": True})
        actual = network.get_minimal_steps()
        expected = 4
        self.assertEqual(actual, expected)

        nM = 13
        network = self.get_network(nM, nS, **{"uniform": True})
        actual = network.get_minimal_steps()
        expected = 4
        self.assertEqual(actual, expected)

        nM = 18
        network = self.get_network(nM, nS, **{"uniform": True})
        actual = network.get_minimal_steps()
        expected = 5
        self.assertEqual(actual, expected)

        nM = 38
        network = self.get_network(nM, nS, **{"uniform": True})
        actual = network.get_minimal_steps()
        expected = 6
        self.assertEqual(actual, expected)

    def num_similiar_configs(self, network):
        seen_configs = []
        seen_configs_count = []
        for m in network.hosts:
            cfg = network.hosts[m].services
            if cfg in seen_configs:
                i = seen_configs.index(cfg)
                seen_configs_count[i] += 1
            else:
                seen_configs.append(cfg)
                seen_configs_count.append(0)
        return len(seen_configs), max(seen_configs_count)

    def num_exploits_avail(self, nS, nM, network):
        vulns = np.zeros(nS)
        for m in network.hosts:
            cfg = network.hosts[m].services
            for i, v in enumerate(cfg.values()):
                if v:
                    vulns[i] += 1
        return np.max(vulns), np.average(vulns)

    def test_print(self):
        m = 20
        s = 1
        network = self.get_network(m, s)
        print()
        print(network)


if __name__ == "__main__":
    unittest.main()
