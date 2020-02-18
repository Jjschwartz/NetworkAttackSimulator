import unittest
from nasim.env.state import State
from nasim.env.environment import NASimEnv
from nasim.env.action import ServiceScan, Exploit


class EnvironmentTestCase(unittest.TestCase):

    def setUp(self):
        self.E = 1
        self.M = 3
        self.service = "srv_0"
        self.scan_cost = 1
        self.exploit_cost = 1
        self.r_sensitive = 10
        self.r_user = 10
        self.env = self.get_env()
        self.network = self.env.network
        self.addr_space = self.network.address_space

    def get_env(self):
        env = NASimEnv.from_params(self.M, self.E,
                                   r_sensitive=self.r_sensitive,
                                   r_user=self.r_user,
                                   exploit_cost=self.exploit_cost,
                                   scan_cost=self.scan_cost)
        return env

    def get_exploit(self, host):
        return Exploit(self.addr_space[host], self.exploit_cost, self.service)

    def test_reset1(self):
        actual_obs = self.env.reset()
        expected_obs = self.get_initial_expected_obs()
        self.assertEqual(actual_obs, expected_obs)

    def test_reset2(self):
        t_action = ServiceScan(self.addr_space[0], self.scan_cost)
        o, _, _ = self.env.step(t_action)
        self.env.reset()
        actual_obs = self.env.reset()
        expected_obs = self.get_initial_expected_obs()
        self.assertEqual(actual_obs, expected_obs)

    def test_step_not_reachable(self):
        t_action = ServiceScan(self.addr_space[1], self.scan_cost)
        expected_obs = self.env.reset()
        o, r, d = self.env.step(t_action)
        self.assertEqual(r, -t_action.cost)
        self.assertFalse(d)
        self.assertEqual(o, expected_obs)

    def test_step_scan_reachable(self):
        t_action = ServiceScan(self.addr_space[0], self.scan_cost)
        expected_obs = self.env.reset()
        o, r, d = self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)
        self.assertEqual(r, -t_action.cost)
        self.assertFalse(d)
        self.assertEqual(o, expected_obs)

    def test_step_exploit_reachable(self):
        t_action = self.get_exploit(0)
        expected_obs = self.env.reset()
        o, r, d = self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)
        self.assertEqual(r, -t_action.cost)
        self.assertFalse(d)
        self.assertEqual(o, expected_obs)

    def test_step_exploit_sensitive(self):
        t_action = self.get_exploit(0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action2 = self.get_exploit(1)
        o, r, d = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertEqual(r, self.r_sensitive - t_action.cost)
        self.assertFalse(d)
        self.assertEqual(o, expected_obs)

    def test_step_exploit_user(self):
        t_action = self.get_exploit(0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action2 = self.get_exploit(2)
        o, r, d = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertEqual(r, self.r_user - t_action.cost)
        self.assertFalse(d)
        self.assertEqual(o, expected_obs)

    def test_step_done(self):
        t_action = self.get_exploit(0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action1 = self.get_exploit(1)
        o, r, d = self.env.step(t_action1)
        self.update_obs(t_action1, expected_obs, True)

        t_action2 = self.get_exploit(2)
        o, r, d = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertTrue(d)
        self.assertEqual(o, expected_obs)

    def test_already_rewarded(self):
        t_action = self.get_exploit(0)
        expected_obs = self.env.reset()
        self.env.step(t_action)
        self.update_obs(t_action, expected_obs, True)

        t_action2 = self.get_exploit(2)
        o, r, d = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        o, r, d = self.env.step(t_action2)
        self.update_obs(t_action2, expected_obs, True)

        self.assertEqual(r, 0 - t_action.cost)
        self.assertFalse(d)
        self.assertEqual(o, expected_obs)

    def get_initial_expected_obs(self):
        return State.generate_initial_state(self.network, [self.service])

    def update_obs(self, action, obs, success):
        """ Valid for test where E = 1 """
        target = action.target
        if success:
            for s in range(self.E):
                obs.update_service(target, f"srv_{s}", True)
        if not action.is_scan() and success:
            obs.set_compromised(target)
            for m in self.addr_space:
                if obs.reachable(m):
                    continue
                if self.network.subnets_connected(target[0], m[0]):
                    obs.set_reachable(m)
        return obs


if __name__ == "__main__":
    unittest.main()
