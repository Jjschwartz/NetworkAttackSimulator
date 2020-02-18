"""A collection of definitions for generated benchmark scenarios.

Each generated scenario is defined by the parameters:
- 'hosts' = number of hosts
- 'services' = number of vulnerable services (and hence actions)
- 'restrictiveness' = how restrictive the firewalls between subnets are

There are also some parameters, which are the same for all scenarios:
- exploit_cost = 1  (cost of performing an exploit)
- scan_cost = 1     (cost of performing a scan)
- r_sensitive = 10  (reward for controlling sensitive host)
- r_user = 10   (reward for controlling user host)
- exploit_probs = 'mixed'   (distribution success probabilities of exploits)
- uniform = False

Each scenario definition also contains additional information, that is
not used for generating the scenario, but may be useful:
- 'max_score' = the thoeretical max undiscounted score possible by an attacker
"""

# generated environment constants
UNIFORM = False
EXPLOIT_PROB = "mixed"
R_SENS = R_USR = 10
COST_EXP = COST_SCAN = 1

# Generated Scenario definitions
TINY_GEN = {"hosts": 3,
            "services": 1,
            "restrictiveness": 1,
            "max_score": 17}
SMALL_GEN = {"hosts": 8,
             "services": 3,
             "restrictiveness": 2,
             "max_score": 16}
MEDIUM_GEN = {"hosts": 13,
              "services": 5,
              "restrictiveness": 3,
              "max_score": 16}
LARGE_GEN = {"hosts": 18,
             "services": 6,
             "restrictiveness": 3,
             "max_score": 15}
HUGE_GEN = {"hosts": 38,
            "services": 10,
            "restrictiveness": 3,
            "max_score": 14}

AVAIL_GEN_BENCHMARKS = {
    "tiny-gen": TINY_GEN,
    "small-gen": SMALL_GEN,
    "medium-gen": MEDIUM_GEN,
    "large-gen": LARGE_GEN,
    "huge-gen": HUGE_GEN
}
