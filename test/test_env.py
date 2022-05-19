"""Runs some general tests on environment"""

import pytest

import nasim
from nasim.scenarios.benchmark import \
    AVAIL_GEN_BENCHMARKS, AVAIL_STATIC_BENCHMARKS


def test_render_error():
    env = nasim.make_benchmark("tiny")
    with pytest.raises(NotImplementedError):
        env.render(mode="a bad mode str")


def test_render_readable():
    env = nasim.make_benchmark("tiny")
    env.render(mode="readable")


def test_render_state_error():
    env = nasim.make_benchmark("tiny")
    with pytest.raises(NotImplementedError):
        env.render_state(mode="a bad mode str")


def test_render_state_readable():
    env = nasim.make_benchmark("tiny")
    env.render_state(mode="readable")


@pytest.mark.parametrize("flat_actions", [True, False])
def test_render_action(flat_actions):
    env = nasim.make_benchmark("tiny", flat_actions=flat_actions)
    env.render_action(env.action_space.sample())
