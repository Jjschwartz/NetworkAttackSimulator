from machine import Machine
from action import Action
import numpy as np
import unittest


class MachineTestCase(unittest.TestCase):

    def setUp(self):
        self.test_r = 5000.0
        self.services = np.asarray([True, False, True])
        self.test_m1 = Machine(1, 1, self.services)
        self.test_m2 = Machine(1, 2, self.services, self.test_r)

    def test_successful_exploit(self):
        # address is ignored as this is controlled and checked at Network level
        exploit = Action((1, 1), "exploit", 0)

        # Test on machine with no sensitive docs (i.e. 0 value)
        outcome, reward, services = self.test_m1.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue((services == self.services).all())

        # Test exploit on machine with sensitive docs
        outcome, reward, services = self.test_m2.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, self.test_r)
        self.assertTrue((services == self.services).all())

    def test_unsuccessful_exploit(self):
        # address is ignored as this is controlled and checked at Network level
        exploit = Action((1, 1), "exploit", 1)

        # Test on machine with no sensitive docs (i.e. 0 value)
        outcome, reward, services = self.test_m1.perform_action(exploit)
        self.assertFalse(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue((services == np.asarray([])).all())

        # Test exploit on machine with sensitive docs
        outcome, reward, services = self.test_m2.perform_action(exploit)
        self.assertFalse(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue((services == np.asarray([])).all())

    def test_scan(self):
        # address is ignored as this is controlled and checked at Network level
        exploit = Action((1, 1), "scan", None)

        # Test on machine with no sensitive docs (i.e. 0 value)
        outcome, reward, services = self.test_m1.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue((services == self.services).all())

        # Test exploit on machine with sensitive docs
        outcome, reward, services = self.test_m2.perform_action(exploit)
        self.assertTrue(outcome)
        self.assertEqual(reward, 0)
        self.assertTrue((services == self.services).all())

    def test_eq(self):
        s1 = np.asarray([True, False, True])
        s2 = np.asarray([True, False, True])
        m1 = Machine(1, 1, s1)
        m2 = Machine(1, 1, s2)
        self.assertTrue(m1 == m2)

        m3 = Machine(1, 1, s1, self.test_r)
        m4 = Machine(1, 1, s2, self.test_r)
        self.assertTrue(m3 == m4)


if __name__ == "__main__":
    unittest.main()
