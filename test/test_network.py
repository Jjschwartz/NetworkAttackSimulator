from network import Network
from network import R_SENSITIVE
from network import R_USER
from action import Action
import numpy as np
import unittest


class NetworkTestCase(unittest.TestCase):

    def setUp(self):
        self.M = 3
        self.S = 3
        self.network = Network(self.M, self.S)

    def find_exploit(self, network, subnet_id, m_id, valid):
        services = network.subnetworks[subnet_id][m_id]._services
        for i in range(len(services)):
            if services[i] == valid:
                return i

    def test_generate_network_small(self):
        M = 3
        S = 3
        network = Network(M, S)
        subnets = network.subnetworks
        self.assertEqual(len(subnets), 3)
        for s in range(1, 4):
            self.assertEqual(len(subnets[s]), 1)
            self.assertEqual(subnets[s][0].address, (s, 0))

    def test_generate_network_consistency(self):
        M = [3, 6]
        S = [3, 6]
        results1 = []
        for m in M:
            for s in S:
                network = Network(m, s)
                results1.append(network.subnetworks)
        results2 = []
        for m in M:
            for s in S:
                network = Network(m, s)
                results2.append(network.subnetworks)
        self.assertEqual(results1, results2)

    def test_successful_exploit(self):
        rewards = [0, R_SENSITIVE, R_USER]
        for i in range(3):
            subnet = i + 1
            e = self.find_exploit(self.network, subnet, 0, True)
            exploit = Action((subnet, 0), "exploit", e)
            outcome, reward, services = self.network.perform_action(exploit)
            self.assertTrue(outcome)
            self.assertEqual(reward, rewards[i])

    def test_unsuccessful_exploit(self):
        rewards = [0, 0, 0]
        exp_services = np.asarray([])
        for i in range(3):
            subnet = i + 1
            e = self.find_exploit(self.network, subnet, 0, False)
            exploit = Action((subnet, 0), "exploit", e)
            outcome, reward, services = self.network.perform_action(exploit)
            self.assertFalse(outcome)
            self.assertEqual(reward, rewards[i])
            self.assertTrue((services == exp_services).all())

    def test_scan(self):
        rewards = [0, 0, 0]
        for i in range(3):
            subnet = i + 1
            exp_services = self.network.subnetworks[subnet][0]._services
            exploit = Action((subnet, 0), "scan", None)
            outcome, reward, services = self.network.perform_action(exploit)
            self.assertTrue(outcome)
            self.assertEqual(reward, rewards[i])
            self.assertTrue((services == exp_services).all())

    def test_invalid_action(self):
        # invalid subnet
        exploit = Action((4, 0), "scan", None)
        with self.assertRaises(AssertionError):
            self.network.perform_action(exploit)
        # invalid machine ID
        exploit = Action((1, 2), "scan", None)
        with self.assertRaises(AssertionError):
            self.network.perform_action(exploit)
        # invalid service
        exploit = Action((1, 0), "exploit", self.S + 1)
        with self.assertRaises(AssertionError):
            self.network.perform_action(exploit)


if __name__ == "__main__":
    unittest.main()
