"""Runs bruteforce agent on environment for different scenarios and
using different parameters to check no exceptions occur
"""

import pytest

import nasim
from nasim.scenarios.benchmark import \
    AVAIL_GEN_BENCHMARKS, AVAIL_STATIC_BENCHMARKS
from nasim.agents.bruteforce_agent import run_bruteforce_agent


@pytest.mark.parametrize("scenario", AVAIL_STATIC_BENCHMARKS)
@pytest.mark.parametrize("seed", [0, 666])
@pytest.mark.parametrize("fully_obs", [True, False])
@pytest.mark.parametrize("flat_actions", [True, False])
@pytest.mark.parametrize("flat_obs", [True, False])
def test_bruteforce_static(scenario, seed, fully_obs, flat_actions, flat_obs):
    """Tests all static benchmark scenarios using every possible environment
    setting, using bruteforce agent, checking for any errors
    """
    env = nasim.make_benchmark(scenario,
                               seed=seed,
                               fully_obs=fully_obs,
                               flat_actions=flat_actions,
                               flat_obs=flat_obs)
    run_bruteforce_agent(env, verbose=False)


@pytest.mark.parametrize("scenario", AVAIL_GEN_BENCHMARKS)
@pytest.mark.parametrize("seed", [0, 30, 666])
@pytest.mark.parametrize("fully_obs", [True, False])
@pytest.mark.parametrize("flat_actions", [True, False])
@pytest.mark.parametrize("flat_obs", [True, False])
def test_bruteforce_gen(scenario, seed, fully_obs, flat_actions, flat_obs):
    """Tests all generated benchmark scenarios using every possible environment
    setting, using bruteforce agent, checking for any errors
    """
    env = nasim.make_benchmark(scenario,
                               seed=seed,
                               fully_obs=fully_obs,
                               flat_actions=flat_actions,
                               flat_obs=flat_obs)
    run_bruteforce_agent(env, verbose=False)
