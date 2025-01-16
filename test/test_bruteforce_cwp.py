"""Runs bruteforce agent on environment for different static scenarios and
using credentials, vulnerabilities and wiretapping
"""

import pytest

import nasim
from nasim import load
from nasim.agents.bruteforce_agent import run_bruteforce_agent


@pytest.mark.parametrize(("scenario"), (
    ("tiny-cred"),("tiny-wire"),("tiny-post"),
    ("small-wire"),("small-post"),("small-cwp"),
    ("tiny-small-cwp"),("tiny-hard-cwp"),("medium-cwp")))

def test_bruteforce_cwp(scenario):
    
    env = load('nasim/scenarios/benchmark/' + scenario + '.yaml')
    
    run_bruteforce_agent(env, verbose = False)