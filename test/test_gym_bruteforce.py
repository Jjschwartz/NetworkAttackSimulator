"""Runs bruteforce agent on environment for different scenarios and
using different parameters to check no exceptions occur.

Tests loading environments using gym.make()
"""
from importlib import reload

import gym
import pytest

import nasim
from nasim.scenarios.benchmark import AVAIL_BENCHMARKS
from nasim.agents.bruteforce_agent import run_bruteforce_agent


def test_gym_reload():
    """Tests there is no issue when reloading gym """
    reload(gym)
    reload(nasim)

@pytest.mark.parametrize("scenario", AVAIL_BENCHMARKS)
@pytest.mark.parametrize("po", ['', 'PO'])
@pytest.mark.parametrize("obs", ['', '2D'])
@pytest.mark.parametrize("actions", ['', 'VA'])
@pytest.mark.parametrize("v", ['v0'])
def test_bruteforce(scenario, po, obs, actions,v):
    """Tests all benchmark scenarios using every possible environment
    setting, using bruteforce agent, checking for any errors
    """
    name = ''.join([g.capitalize() for g in scenario.split("-")])
    name = f"nasim:{name}{po}{obs}{actions}-{v}"
    env = gym.make(name, new_step_api=True)
    run_bruteforce_agent(env, verbose=False)
