"""A collection of definitions for generated benchmark scenarios.

Each generated scenario is defined by the a number of parameters that
control the size of the problem (see scenario.generator for more info):

There are also some parameters, where default values are used for all scenarios,
see DEFAULTS dict.
"""

# generated environment constants
DEFAULTS = dict(
    num_exploits=None,
    r_sensitive=10,
    r_user=10,
    exploit_cost=1,
    exploit_probs='mixed',
    service_scan_cost=1,
    os_scan_cost=1,
    uniform=False,
    alpha_H=2.0,
    alpha_V=2.0,
    lambda_V=1.0,
    random_goal=False
)

# Generated Scenario definitions
TINY_GEN = {**DEFAULTS,
            "num_hosts": 3,
            "num_os": 1,
            "num_services": 1,
            "restrictiveness": 1}
TINY_GEN_RGOAL = {**DEFAULTS,
                  "num_hosts": 3,
                  "num_os": 1,
                  "num_services": 1,
                  "restrictiveness": 1,
                  "random_goal": True}
SMALL_GEN = {**DEFAULTS,
             "num_hosts": 8,
             "num_os": 2,
             "num_services": 3,
             "restrictiveness": 2}
SMALL_GEN_RGOAL = {**DEFAULTS,
                   "num_hosts": 8,
                   "num_os": 2,
                   "num_services": 3,
                   "restrictiveness": 2,
                   "random_goal": True}
MEDIUM_GEN = {**DEFAULTS,
              "num_hosts": 13,
              "num_os": 2,
              "num_services": 5,
              "restrictiveness": 3}
LARGE_GEN = {**DEFAULTS,
             "num_hosts": 18,
             "num_os": 3,
             "num_services": 6,
             "restrictiveness": 3}
HUGE_GEN = {**DEFAULTS,
            "num_hosts": 38,
            "num_os": 4,
            "num_services": 10,
            "restrictiveness": 3}
POCP_1_GEN = {**DEFAULTS,
              "num_hosts": 35,
              "num_os": 2,
              "num_services": 50,
              "num_exploits": 60,
              "restrictiveness": 5}
POCP_2_GEN = {**DEFAULTS,
              "num_hosts": 95,
              "num_os": 3,
              "num_services": 10,
              "num_exploits": 30,
              "restrictiveness": 5}


AVAIL_GEN_BENCHMARKS = {
    "tiny-gen": TINY_GEN,
    "tiny-gen-rgoal": TINY_GEN_RGOAL,
    "small-gen": SMALL_GEN,
    "small-gen-rgoal": SMALL_GEN_RGOAL,
    "medium-gen": MEDIUM_GEN,
    "large-gen": LARGE_GEN,
    "huge-gen": HUGE_GEN,
    "pocp-1-gen": POCP_1_GEN,
    "pocp-2-gen": POCP_2_GEN
}
