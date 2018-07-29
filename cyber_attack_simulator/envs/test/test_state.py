import unittest
import copy
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv


class ActionTestCase(unittest.TestCase):

    def setUp(self):
        self.E = 1
        self.M = 3
        self.env = CyberAttackSimulatorEnv(self.M, self.E)
        self.init_state = self.env.reset()

    def test_state_deepcopy(self):
        state_copy = copy.deepcopy(self.init_state)
        self.assertFalse(state_copy is self.init_state)
        self.assertTrue(state_copy == self.init_state)

if __name__ == "__main__":
    unittest.main()
