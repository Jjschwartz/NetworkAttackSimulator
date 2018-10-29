import unittest
from cyber_attack_simulator.envs.action import Action


class ActionTestCase(unittest.TestCase):

    def setUp(self):
        self.test_exploit_cost = 100.0
        self.test_scan_cost = 100.0
        self.E = 1
        self.M = 3
        m1 = (1, 0)
        m2 = (2, 0)
        m3 = (3, 0)
        self.ads_space = [m1, m2, m3]

    def test_action_space(self):
        actual_A_space = Action.generate_action_space(self.ads_space, self.E,
                                                      self.test_exploit_cost, self.test_scan_cost)
        expected_A_space = []
        for m in self.ads_space:
            scan = Action(m, self.test_scan_cost, "scan")
            expected_A_space.append(scan)
            for e in range(self.E):
                exploit = Action(m, self.test_exploit_cost, "exploit", e)
                expected_A_space.append(exploit)
        self.assertEqual(actual_A_space, expected_A_space)

    def test_gen_action_space_mixed(self):
        actual_A_space = Action.generate_action_space(self.ads_space, self.E,
                                                      self.test_exploit_cost, self.test_scan_cost,
                                                      exploit_probs="mixed")
        for a in actual_A_space:
            if a.is_scan():
                self.assertTrue(a.prob == 1.0)
            else:
                self.assertTrue(a.prob == 0.5 or a.prob == 0.8)

        actual_A_space = Action.generate_action_space(self.ads_space, 5,
                                                      self.test_exploit_cost, self.test_scan_cost,
                                                      exploit_probs="mixed")
        for a in actual_A_space:
            if a.is_scan():
                self.assertTrue(a.prob == 1.0)
            else:
                self.assertTrue(a.prob == 0.2 or a.prob == 0.5 or a.prob == 0.8)


if __name__ == "__main__":
    unittest.main()
