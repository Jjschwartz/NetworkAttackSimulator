"""A collection of definitions for generated benchmark scenarios.

Each generated scenario is defined by the a number of parameters that
control the size of the problem (see scenario.generator for more info):

There are also some parameters, where default values are used for all
scenarios, see DEFAULTS dict.
"""

# generated environment constants
DEFAULTS = dict(
    num_exploits=None,
    num_privescs=None,
    r_sensitive=100,
    r_user=100,
    exploit_cost=1,
    exploit_probs='mixed',
    privesc_cost=1,
    privesc_probs=1.0,
    service_scan_cost=1,
    os_scan_cost=1,
    subnet_scan_cost=1,
    process_scan_cost=1,
    uniform=False,
    alpha_H=2.0,
    alpha_V=2.0,
    lambda_V=1.0,
    random_goal=False,
    base_host_value=1,
    host_discovery_value=1,
    step_limit=1000,
    address_space_bounds=None
)

# Generated Scenario definitions
TINY_GEN = {**DEFAULTS,
            "name": "tiny-gen",
            "num_hosts": 3,
            "num_os": 1,
            "num_services": 1,
            "num_processes": 1,
            "restrictiveness": 1}
TINY_GEN_RGOAL = {**DEFAULTS,
                  "name": "tiny-gen-rangoal",
                  "num_hosts": 3,
                  "num_os": 1,
                  "num_services": 1,
                  "num_processes": 1,
                  "restrictiveness": 1,
                  "random_goal": True}
SMALL_GEN = {**DEFAULTS,
             "name": "small-gen",
             "num_hosts": 8,
             "num_os": 2,
             "num_services": 3,
             "num_processes": 2,
             "restrictiveness": 2}
SMALL_GEN_RGOAL = {**DEFAULTS,
                   "name": "small-gen-rangoal",
                   "num_hosts": 8,
                   "num_os": 2,
                   "num_services": 3,
                   "num_processes": 2,
                   "restrictiveness": 2,
                   "random_goal": True}
MEDIUM_GEN = {**DEFAULTS,
              "name": "medium-gen",
              "num_hosts": 16,
              "num_os": 2,
              "num_services": 5,
              "num_processes": 2,
              "restrictiveness": 3,
              "step_limit": 2000}
LARGE_GEN = {**DEFAULTS,
             "name": "large-gen",
             "num_hosts": 23,
             "num_os": 3,
             "num_services": 7,
             "num_processes": 3,
             "restrictiveness": 3,
             "step_limit": 5000}
HUGE_GEN = {**DEFAULTS,
            "name": "huge-gen",
            "num_hosts": 38,
            "num_os": 4,
            "num_services": 10,
            "num_processes": 4,
            "restrictiveness": 3,
            "step_limit": 10000}
POCP_1_GEN = {**DEFAULTS,
              "name": "pocp-1-gen",
              "num_hosts": 35,
              "num_os": 2,
              "num_services": 50,
              "num_exploits": 60,
              "num_processes": 2,
              "restrictiveness": 5,
              "step_limit": 30000}
POCP_2_GEN = {**DEFAULTS,
              "name": "pocp-2-gen",
              "num_hosts": 95,
              "num_os": 3,
              "num_services": 10,
              "num_exploits": 30,
              "num_processes": 3,
              "restrictiveness": 5,
              "step_limit": 30000}


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
