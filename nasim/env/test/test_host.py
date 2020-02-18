import unittest
import numpy as np

from nasim.env.host import Host
from nasim.env.action import Exploit, ServiceScan

A_COST = 10


class HostTestCase(unittest.TestCase):

    def setUp(self):
        self.test_r = 5000.0
        self.services = {"ssh": True,
                         "banana": False,
                         "ftp": True}
        self.test_m1 = Host((1, 1), self.services)
        self.test_m2 = Host((1, 2), self.services, self.test_r)

    def get_exploit(self, addr, srv):
        return Exploit(addr, A_COST, srv)

    def test_successful_exploit(self):
        # address is ignored as this is controlled and checked at Network level
        exploit = self.get_exploit((1, 1), "ssh")

        # Test on host with no sensitive docs (i.e. 0 value)
        outcome, reward, services = self.test_m1.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue(services == self.services)

        # Test exploit on host with sensitive docs
        outcome, reward, services = self.test_m2.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, self.test_r)
        self.assertTrue(services == self.services)

    def test_unsuccessful_exploit(self):
        # address is ignored as this is controlled and checked at Network level
        exploit = self.get_exploit((1, 1), "banana")

        # Test on host with no sensitive docs (i.e. 0 value)
        outcome, reward, services = self.test_m1.perform_action(exploit)
        self.assertFalse(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue((services == np.asarray([])).all())

        # Test exploit on host with sensitive docs
        outcome, reward, services = self.test_m2.perform_action(exploit)
        self.assertFalse(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue((services == np.asarray([])).all())

    def test_scan(self):
        # address is ignored as this is controlled and checked at Network level
        exploit = ServiceScan((1, 1), A_COST)

        # Test on host with no sensitive docs (i.e. 0 value)
        outcome, reward, services = self.test_m1.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue(services == self.services)

        # Test exploit on host with sensitive docs
        outcome, reward, services = self.test_m2.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue(services == self.services)

    def test_eq(self):
        s1 = {"ssh": True, "banana": False, "ftp": True}
        s2 = {"ssh": True, "banana": False, "ftp": True}
        m1 = Host((1, 1), s1)
        m2 = Host((1, 1), s2)
        self.assertTrue(m1 == m2)

        m3 = Host((1, 1), s1, self.test_r)
        m4 = Host((1, 1), s2, self.test_r)
        self.assertTrue(m3 == m4)


if __name__ == "__main__":
    unittest.main()
