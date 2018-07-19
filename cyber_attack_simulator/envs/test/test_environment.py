import unittest
from collections import OrderedDict
import numpy as np
from cyber_attack_simulator.envs import CyberAttackSimulatorEnv
from cyber_attack_simulator.envs.environment import ServiceState
import cyber_attack_simulator.envs.environment as environment
from cyber_attack_simulator.envs.action import Action


class EnvironmentTestCase(unittest.TestCase):

    def setUp(self):
        self.E = 1
        self.M = 3
        self.env = CyberAttackSimulatorEnv(self.M, self.E)
        m1 = (1, 0)
        m2 = (2, 0)
        m3 = (3, 0)
        self.t_machines = [m1, m2, m3]

    def test_action_space(self):
        actual_A_space = self.env.action_space
        expected_A_space = set()
        for m in self.t_machines:
            expected_A_space.add(Action(m, "scan", None))
            for e in range(self.E):
                expected_A_space.add(Action(m, "exploit", e))
        self.assertSetEqual(actual_A_space, expected_A_space)

    def test_reset1(self):
        actual_obs = self.env.reset()
        expected_obs = self.get_initial_expected_obs()
        self.assertDictEqual(actual_obs, expected_obs)

    def test_reset2(self):
        t_action = Action(self.t_machines[0], "scan", None)
        o, _, _, _ = self.env.step(t_action)
        self.env.reset()
        actual_obs = self.env.reset()
        expected_obs = self.get_initial_expected_obs()
        self.assertDictEqual(actual_obs, expected_obs)

    def test_step_not_reachable(self):
        t_action = Action(self.t_machines[1], "scan", None)
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(t_action)
        self.assertEqual(r, -t_action.cost)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_scan_reachable(self):
        t_action = Action(self.t_machines[0], "scan", None)
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)
        self.assertEqual(r, -t_action.cost)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_exploit_reachable(self):
        t_action = Action(self.t_machines[0], "exploit", 0)
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)
        self.assertEqual(r, -t_action.cost)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_exploit_sensitive(self):
        t_action = Action(self.t_machines[0], "exploit", 0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action2 = Action(self.t_machines[1], "exploit", 0)
        o, r, d, _ = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertEqual(r, environment.R_SENSITIVE - t_action.cost)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_exploit_user(self):
        t_action = Action(self.t_machines[0], "exploit", 0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action2 = Action(self.t_machines[2], "exploit", 0)
        o, r, d, _ = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertEqual(r, environment.R_USER - t_action.cost)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_done(self):
        t_action = Action(self.t_machines[0], "exploit", 0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action1 = Action(self.t_machines[1], "exploit", 0)
        o, r, d, _ = self.env.step(t_action1)
        self.update_obs(t_action1, expected_obs, True)

        t_action2 = Action(self.t_machines[2], "exploit", 0)
        o, r, d, _ = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertTrue(d)
        self.assertDictEqual(o, expected_obs)

    def test_already_rewarded(self):
        t_action = Action(self.t_machines[0], "exploit", 0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action2 = Action(self.t_machines[2], "exploit", 0)
        o, r, d, _ = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        o, r, d, _ = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertEqual(r, 0 - t_action.cost)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def get_initial_expected_obs(self):
        t_service_info = np.full(self.E, ServiceState.unknown, ServiceState)
        t_compromised = False
        t_obs = {}
        for m in self.t_machines:
            t_reachable = False
            t_sensitive = True
            if m[0] == 1:
                t_reachable = True
                t_sensitive = False
            t_obs[m] = {"service_info": t_service_info,
                        "compromised": t_compromised,
                        "reachable": t_reachable,
                        "sensitive": t_sensitive}
        return OrderedDict(sorted(t_obs.items()))

    def update_obs(self, action, obs, success):
        """ Valid for test where E = 1 """
        m = action.target
        if success:
            for s in range(len(obs[m]["service_info"])):
                obs[m]["service_info"][s] = ServiceState.present
        if not action.is_scan() and success:
            obs[action.target]["compromised"] = True
            if m[0] == 1:
                for o in obs.keys():
                    obs[o]["reachable"] = True
        elif not action.is_scan() and not success:
            obs[m["service_info"][action.service]] = ServiceState.absent
        return obs


if __name__ == "__main__":
    unittest.main()
