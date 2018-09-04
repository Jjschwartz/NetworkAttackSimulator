import unittest
from cyber_attack_simulator.envs.action import Action


class ActionTestCase(unittest.TestCase):

    def setUp(self):
        self.E = 1
        self.M = 3
        m1 = (0, 0)
        m2 = (1, 0)
        m3 = (2, 0)
        self.ads_space = [m1, m2, m3]

    def test_action_space(self):
        actual_A_space = Action.generate_action_space(self.ads_space, self.E)
        expected_A_space = []
        for m in self.ads_space:
            scan = Action(m, "scan")
            expected_A_space.append(scan)
            for e in range(self.E):
                exploit = Action(m, "exploit", e)
                expected_A_space.append(exploit)
        self.assertEqual(actual_A_space, expected_A_space)


if __name__ == "__main__":
    unittest.main()
