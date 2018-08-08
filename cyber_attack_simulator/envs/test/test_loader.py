import unittest
import yaml
import cyber_attack_simulator.envs.loader as loader


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

    def test_check_config_valid_invalid_subnet_connections(self):
        config = self.get_valid_config_dict()
        # invalid number of rows
        config["subnet_connections"] = [[1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # not 2D array
        config["subnet_connections"] = [1, 1, 1]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid number of columns
        config["subnet_connections"] = [[1, 1],
                                        [1, 1],
                                        [1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid number of columns V2
        config["subnet_connections"] = [[1, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value
        config["subnet_connections"] = [[1.0, 1, 1, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value V2
        config["subnet_connections"] = [[1, -1, 1, 1],
                                        [1, 1, 1, 1],
                                        [1, 1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value V3
        config["subnet_connections"] = [[1, 1, 1, 1],
                                        [1, 1, 2, 1],
                                        [1, 1, 1, 1]]
        with self.assertRaises(ValueError):
            loader.check_config_valid(config)
        # invalid value V4
        config["subnet_connections"] = [[1, 1, 1, 1],
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
        config["sensitive_machines"] = [[0, 0, 1], [1, 0, 1], [2, 0, 1],
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

    def test_load_config_invalid_subnet_connections(self):
        file = "inputs/invalid_config_subnet_connections.yaml"
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

    def get_valid_config_dict(self):
        config = {}
        for k, v in loader.VALID_CONFIG_KEYS.items():
            if k is "subnets":
                value = [1, 1, 1]
            if k is "subnet_connections":
                value = [[1, 1, 1, 1],
                         [0, 1, 1, 1],
                         [0, 1, 1, 1]]
            if k is "services":
                value = 2
            if k is "sensitive_machines":
                value = [[1, 0, 9000], [2, 0, 5000]]
            config[k] = value
        return config


if __name__ == "__main__":
    unittest.main()
