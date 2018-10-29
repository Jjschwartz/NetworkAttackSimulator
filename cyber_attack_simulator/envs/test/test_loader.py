import unittest
import yaml
import numpy as np
import cyber_attack_simulator.envs.loader as loader
from cyber_attack_simulator.envs.loader import INTERNET, DMZ, SENSITIVE, USER
from cyber_attack_simulator.envs.machine import Machine


class LoaderTestCase(unittest.TestCase):

    def test_load_yaml_file_non_existent(self):
        invalid_file = "invalid_file"
        with self.assertRaises(FileNotFoundError):
            loader.load_yaml_file(invalid_file)

    def test_load_yaml_file_invalid(self):
        invalid_file = "inputs/invalid_yaml.yaml"
        with self.assertRaises(yaml.YAMLError):
            loader.load_yaml_file(invalid_file)

    def test_load_yaml_file_valid(self):
        file = "inputs/valid_yaml.yaml"
        expected = {"brackets": []}
        actual = loader.load_yaml_file(file)
        self.assertDictEqual(expected, actual)

    def test_check_config_valid_invalid_key_number(self):
        config_small = {"invalid_key": "some_data"}
        expected_num_keys = len(loader.VALID_CONFIG_KEYS)
        config_large = {}
        for i in range(expected_num_keys + 1):
            config_large["invalid_key" + str(i)] = "some_data" + str(i)
        with self.assertRaises(KeyError):
            loader.check_config_valid(config_small)
            loader.check_config_valid(config_large)

    def test_check_config_valid_invalid_key(self):
        expected_num_keys = len(loader.VALID_CONFIG_KEYS)
        config = {}
        for i in range(expected_num_keys):
            config["invalid_key" + str(i)] = "some_data" + str(i)
        with self.assertRaises(KeyError):
            loader.check_config_valid(config)

    def test_check_config_valid_invalid_value_type(self):
        config = {}
        for k in loader.VALID_CONFIG_KEYS.keys():
            config[k] = "some_data"
        with self.assertRaises(TypeError):
            loader.check_config_valid(config)

    def test_check_config_valid_invalid_subnets(self):
        config = self.get_valid_config_dict()

        config["subnets"] = ["invalid_type"]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)

        config["subnets"] = [1.0]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)

        config["subnets"] = [-1]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)

        config["subnets"] = [0]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)

    def test_check_config_valid_invalid_topology(self):
        config = self.get_valid_config_dict()
        # invalid number of rows
        config["topology"] = [[1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # not 2D array
        config["topology"] = [1, 1, 1]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid number of columns
        config["topology"] = [[1, 1],
                              [1, 1],
                              [1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid number of columns V2
        config["topology"] = [[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value
        config["topology"] = [[1.0, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value V2
        config["topology"] = [[1, -1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value V3
        config["topology"] = [[1, 1, 1, 1],
                              [1, 1, 2, 1],
                              [1, 1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value V4
        config["topology"] = [[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, "invalid", 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)

    def test_check_config_valid_invalid_services(self):
        config = self.get_valid_config_dict()
        # invalid value
        config["services"] = -1
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value V2
        config["services"] = 0
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)

    def test_check_config_valid_invalid_sensitive_machines(self):
        config = self.get_valid_config_dict()
        # too few sensitive_machines
        config["sensitive_machines"] = []
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # too many sensitive_machines
        config["sensitive_machines"] = [[0, 0, 1],
                                        [1, 0, 1],
                                        [2, 0, 1],
                                        [3, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: too short
        config["sensitive_machines"] = [[0, 0], [1, 0, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: too long
        config["sensitive_machines"] = [[0, 0, 0, 0], [1, 0, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: subnet not int
        config["sensitive_machines"] = [["str_a", 0, 0], [1, 0, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: subnet id too small
        config["sensitive_machines"] = [[0, 0, 0], [-1, 0, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: subnet id too large
        config["sensitive_machines"] = [[0, 0, 0], [1, 0, 1], [3, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: machine id not int
        config["sensitive_machines"] = [[0, "str_a", 0], [1, 0, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: machine id too small
        config["sensitive_machines"] = [[0, 0, 0], [1, -1, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: machine id too large
        config["sensitive_machines"] = [[0, 0, 0], [1, 0, 1], [2, 2, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: value not int or float
        config["sensitive_machines"] = [[0, 0, "str_a"], [1, 0, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: value too small
        config["sensitive_machines"] = [[0, 0, 0], [1, 0, 1], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid tuple form: value too small V2
        config["sensitive_machines"] = [[0, 0, 100], [1, 0, -1.0], [2, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # duplicate machine addresses
        config["sensitive_machines"] = [[0, 0, 1], [1, 0, 1], [0, 0, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # duplicate machine addresses V2
        config["sensitive_machines"] = [[0, 0, 1], [1, 0, 1], [0, 0, 5]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)

    def test_check_config_valid_valid_config(self):
        config = self.get_valid_config_dict()
        try:
            loader.check_config_valid(config)
        except Exception:
            self.fail()

    def test_load_config_from_file_valid_config(self):
        expected = self.get_valid_config_dict()
        actual = loader.load_config("inputs/valid_config.yaml")
        self.assertDictEqual(expected, actual)

    def test_load_config_invalid_subnets(self):
        file = "inputs/invalid_config_subnets.yaml"
        with self.assertRaises(ValueError):
            loader.load_config(file)

    def test_load_config_invalid_topology(self):
        file = "inputs/invalid_config_topology.yaml"
        with self.assertRaises(ValueError):
            loader.load_config(file)

    def test_load_config_invalid_services(self):
        file = "inputs/invalid_config_services.yaml"
        with self.assertRaises(TypeError):
            loader.load_config(file)

    def test_load_config_invalid_sensitive_machines(self):
        file = "inputs/invalid_config_sensitive_machines.yaml"
        with self.assertRaises(ValueError):
            loader.load_config(file)

    def test_generate_topology(self):
        subnets = [1, 1, 1, 1]
        expected = np.asarray([[1, 1, 0, 0],
                               [1, 1, 1, 1],
                               [0, 1, 1, 1],
                               [0, 1, 1, 1]])
        actual = loader.generate_topology(subnets)

        self.assertTrue(np.array_equal(expected, actual))

        subnets = [1, 1, 1, 5, 3]
        expected = np.asarray([[1, 1, 0, 0, 0],
                               [1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1],
                               [0, 0, 0, 1, 1]])
        actual = loader.generate_topology(subnets)

        self.assertTrue(np.array_equal(expected, actual))

    def test_correllated_configs(self):
        nM = 40
        nS = 10
        alpha_H = [0.5, 1, 2.5, 5, 10]
        alpha_V = [0.5, 1, 2.5, 5, 10]
        lambda_V = [1, 2.5, 5, 10]
        subnets = loader.generate_subnets(nM)
        s_machines = loader.generate_sensitive_machines(subnets, 1, 1)

        # test generating network with uniform dist of configs
        print("\nTest Uniform dist of configurations")
        np.random.seed(1)
        machines = loader.generate_machines(subnets, nS, s_machines, uniform=True, alpha_H=0.5,
                                            alpha_V=2.0, lambda_V=1.0)
        num_configs, max_same = self.num_similiar_configs(machines)
        print("\tUniform: num_configs={0}, max_same_configs={1}".format(num_configs, max_same))
        max_vulns, avg_vulns = self.num_exploits_avail(nS, nM, machines)
        print("\tUniform: max_vulns={0}, avg_vulns={1}".format(max_vulns, avg_vulns))

        # for each compare number of similar configs for range of alpha_H
        print("Test alpha_H")
        for h in alpha_H:
            np.random.seed(1)
            machines = loader.generate_machines(subnets, nS, s_machines, uniform=False, alpha_H=h,
                                                alpha_V=2.0, lambda_V=1.0)
            num_configs, max_same = self.num_similiar_configs(machines)
            print("\talpha_H={0}, num_configs={1}, max_same_configs={2}"
                  .format(h, num_configs, max_same))

        # for each compare number of similar configs and services for range of alpha_V
        print("Test alpha_V")
        for v in alpha_V:
            np.random.seed(1)
            machines = loader.generate_machines(subnets, nS, s_machines, uniform=False, alpha_H=2.0,
                                                alpha_V=v, lambda_V=1.0)
            num_configs, max_same = self.num_similiar_configs(machines)
            print("\talpha_V={0}, num_configs={1}, max_same_configs={2}"
                  .format(v, num_configs, max_same))

        # for each compare number of services per machine for range of lambda_V
        print("Test lambda_V")
        for l in lambda_V:
            np.random.seed(1)
            machines = loader.generate_machines(subnets, nS, s_machines, uniform=False, alpha_H=2.0,
                                                alpha_V=2.0, lambda_V=l)
            max_vulns, avg_vulns = self.num_exploits_avail(nS, nM, machines)
            print("\tlambda_V={0}, max_vulns={1}, avg_vulns={2}".format(l, max_vulns, avg_vulns))

    def test_generate_firewall_one_exploit(self):
        nM = 3
        nS = 1
        config = loader.generate_config(nM, nS, 1, 1)
        firewall = config['firewall']
        num_subnets = USER + 1

        for src in range(num_subnets):
            for dest in range(num_subnets):
                services = firewall[src][dest]
                if (src != INTERNET and dest == INTERNET) or (src == INTERNET and dest > DMZ):
                    for s in services:
                        self.assertFalse(s)
                else:
                    for s in services:
                        self.assertTrue(s)

    def test_generate_firewall_two_exploits_none_restricted(self):
        nM = 3
        nS = 2
        config = loader.generate_config(nM, nS, 1, 1, restrictiveness=5)
        firewall = config['firewall']
        num_subnets = USER + 1

        for src in range(num_subnets):
            for dest in range(num_subnets):
                services = firewall[src][dest]
                if (src != INTERNET and dest == INTERNET) or (src == INTERNET and dest > DMZ):
                    for s in services:
                        self.assertFalse(s)
                else:
                    for s in services:
                        self.assertTrue(s)

    def test_generate_firewall_two_exploits_one_restricted(self):
        subnets = [1, 1, 1, 1]
        num_subnets = len(subnets)
        nS = 2
        restrictiveness = 1
        machines = {}
        machines[(DMZ, 0)] = Machine((DMZ, 0), [True, False], 0)
        machines[(SENSITIVE, 0)] = Machine((SENSITIVE, 0), [True, False], 0)
        machines[(USER, 0)] = Machine((USER, 0), [False, True], 0)
        firewall = loader.generate_firewalls(subnets, nS, machines, restrictiveness)

        expected = np.full((num_subnets, num_subnets, nS), False)
        for src in range(num_subnets):
            for dest in range(num_subnets):
                if src == dest:
                    expected[src][dest][0] = True
                    expected[src][dest][1] = True
                elif dest == INTERNET or (src == INTERNET and dest > DMZ):
                    continue
                else:
                    m = machines[(dest, 0)]
                    expected[src][dest][0] = m._services[0]
                    expected[src][dest][1] = m._services[1]

        self.assertTrue(np.equal(expected, firewall).all())

    def test_generate_firewall_two_exploits_three_restricted(self):
        subnets = [1, 1, 1, 1]
        num_subnets = len(subnets)
        nS = 2
        restrictiveness = 3
        machines = {}
        machines[(DMZ, 0)] = Machine((DMZ, 0), [True, False], 0)
        machines[(SENSITIVE, 0)] = Machine((SENSITIVE, 0), [True, False], 0)
        machines[(USER, 0)] = Machine((USER, 0), [False, True], 0)
        firewall = loader.generate_firewalls(subnets, nS, machines, restrictiveness)

        expected = np.full((num_subnets, num_subnets, nS), True)
        for src in range(num_subnets):
            for dest in range(num_subnets):
                if (src != INTERNET and dest == INTERNET) or (src == INTERNET and dest > DMZ):
                    expected[src][dest][0] = False
                    expected[src][dest][1] = False
        self.assertTrue(np.equal(expected, firewall).all())

    def num_similiar_configs(self, machines):
        seen_configs = []
        seen_configs_count = []
        for m in machines.values():
            cfg = m._services
            if cfg in seen_configs:
                i = seen_configs.index(cfg)
                seen_configs_count[i] += 1
            else:
                seen_configs.append(cfg)
                seen_configs_count.append(0)
        return len(seen_configs), max(seen_configs_count)

    def num_exploits_avail(self, nS, nM, machines):
        vulns = np.zeros(nS)
        for m in machines.values():
            cfg = m._services
            for i in range(nS):
                if cfg[i]:
                    vulns[i] += 1
        return np.max(vulns), np.average(vulns)

    def get_valid_config_dict(self):
        config = {}
        for k, v in loader.VALID_CONFIG_KEYS.items():
            if k is "subnets":
                value = [1, 1, 1]
            if k is "topology":
                value = [[1, 1, 1, 1],
                         [0, 1, 1, 1],
                         [0, 1, 1, 1]]
            if k is "services":
                value = 2
            if k is "sensitive_machines":
                value = [[1, 0, 9000], [2, 0, 5000]]
            config[k] = value
        return config

    def print_firewall(self, firewall):
        print()
        num_subnets = len(firewall)
        for src in range(num_subnets):
            for dest in range(num_subnets):
                services = firewall[src][dest]
                print("{0} -> {1}: {2}".format(src, dest, services))


if __name__ == "__main__":
    unittest.main()
