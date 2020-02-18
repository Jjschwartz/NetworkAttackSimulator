import os.path as osp


SCENARIO_DIR = osp.dirname(osp.abspath(__file__))

# default subnet address for internet
INTERNET = 0

# scenario property keys
SUBNETS = "subnets"
TOPOLOGY = "topology"
SENSITIVE_HOSTS = "sensitive_hosts"
SERVICES = "services"
EXPLOITS = "exploits"
SCAN_COST = "scan_cost"
HOST_CONFIGS = "host_configurations"
FIREWALL = "firewall"
HOSTS = "host"

# scenario exploit keys
EXPLOIT_SERVICE = "service"
EXPLOIT_PROB = "prob"
EXPLOIT_COST = "cost"
