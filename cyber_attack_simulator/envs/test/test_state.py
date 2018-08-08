import unittest
import copy
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
import cyber_attack_simulator.envs.loader as loader


class ActionTestCase(unittest.TestCase):

    def setUp(self):
        self.E = 1
        self.M = 3
        self.env = self.get_env(self.M, self.E)
        self.init_state = self.env.reset()

    def get_env(self, nM, nS):
        config = loader.generate_config(nM, nS)
        env = CyberAttackSimulatorEnv(config)
        return env

    def test_state_deepcopy(self):
        state_copy = copy.deepcopy(self.init_state)
        self.assertFalse(state_copy is self.init_state)
        self.assertTrue(state_copy == self.init_state)


if __name__ == "__main__":
    unittest.main()
