"""Runs bruteforce agent on environment for different scenarios and
using different parameters to check no exceptions occur
"""

import pytest

import nasim
from nasim.scenarios.benchmark import \
    AVAIL_GEN_BENCHMARKS


@pytest.mark.parametrize("scenario", AVAIL_GEN_BENCHMARKS)
@pytest.mark.parametrize("seed", list(range(100)))
def test_generator(scenario, seed):
    """Tests generating all generated benchmark scenarios using a range of
    seeds, checking for any errors
    """
    nasim.make_benchmark(scenario, seed=seed)
