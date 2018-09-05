import unittest
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv


class ActionTestCase(unittest.TestCase):

    def setUp(self):
        self.E = 1
        self.M = 3
        self.env = self.get_env(self.M, self.E)
        self.init_state = self.env.reset()

    def get_env(self, nM, nS):
        env = CyberAttackSimulatorEnv.from_params(nM, nS)
        return env

    def test_state_copy(self):
        state_copy = self.init_state.copy()
        self.assertFalse(state_copy is self.init_state)
        self.assertTrue(state_copy == self.init_state)


if __name__ == "__main__":
    unittest.main()
