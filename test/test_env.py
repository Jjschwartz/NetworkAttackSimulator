"""Runs some general tests on environment"""

import pytest

import nasim
from nasim.scenarios.benchmark import \
    AVAIL_GEN_BENCHMARKS, AVAIL_STATIC_BENCHMARKS


def test_render_error():
    env = nasim.make_benchmark("tiny", render_mode="a bad mode str")
    env.reset()
    with pytest.raises(NotImplementedError):
        env.render()


def test_render_readable():
    env = nasim.make_benchmark("tiny", render_mode="human")
    env.reset()
    env.render()


def test_render_state_error():
    env = nasim.make_benchmark("tiny")
    env.reset()
    with pytest.raises(NotImplementedError):
        env.render_state(mode="a bad mode str")


def test_render_state_readable():
    env = nasim.make_benchmark("tiny")
    env.reset()
    env.render_state(mode="human")


@pytest.mark.parametrize("flat_actions", [True, False])
def test_render_action(flat_actions):
    env = nasim.make_benchmark("tiny", flat_actions=flat_actions)
    env.reset()
    env.render_action(env.action_space.sample())


@pytest.mark.parametrize(
    ("scenario", "expected_value"),
    [("tiny", 0.0), ("small", 0.0)]
)
def test_get_total_discovery_value(scenario, expected_value):
    env = nasim.make_benchmark(scenario)
    env.reset()
    actual_value = env.network.get_total_discovery_value()
    assert actual_value == expected_value


@pytest.mark.parametrize(
    ("scenario", "expected_value"),
    [("tiny", 200.0), ("small", 200.0)]
)
def test_get_total_sensitive_host_value(scenario, expected_value):
    env = nasim.make_benchmark(scenario)
    env.reset()
    actual_value = env.network.get_total_sensitive_host_value()
    assert actual_value == expected_value
