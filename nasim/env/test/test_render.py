from nasim.env.render import Viewer
from nasim.env.action import Exploit
from nasim.env import make_benchmark_env

# Must import matplotlib this way for compatibility with TKinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # noqa


def generate_episode(env, actions):
    episode = []
    init_state = env.reset()
    state = init_state
    for a in actions:
        new_state, r, d = env.step(a)
        episode.append((state.copy(), a, r, d))
        state = new_state
    return episode


def test_render_network_graph(env):
    init_state = env.reset()
    viewer = Viewer(env.network)
    # Render on new seperate figure
    viewer.render_graph(init_state, None, True)

    # render on passed in figure
    fig = plt.figure()
    axes = fig.add_subplot(122)
    viewer.render_graph(init_state, axes, False)
    plt.show()
    plt.close(fig)


def test_render_episode(env, actions):
    episode = generate_episode(env, actions)
    viewer = Viewer(env.network)
    viewer.render_episode(episode)
    # test for closing window and rendering a new episode
    viewer.render_episode(episode)


def test_render_asci(env, actions):
    env.render("ASCI")
    for a in actions:
        env.step(a)
        env.render("ASCI")


def test_render_readable(env, actions):
    env.render("readable")
    for a in actions:
        env.step(a)
        env.render("readable")


def get_exploit(addr, srv):
    return Exploit(addr, 10, srv)


def test_render(env):
    """Test rendering of a single episode using Viewer class in render module """
    actions = []
    service = env.scenario.services[0]
    for address in env.address_space:
        actions.append(get_exploit(address, service))

    test_render_asci(env, actions)
    test_render_readable(env, actions)
    test_render_network_graph(env)
    test_render_episode(env, actions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    env = make_benchmark_env(args.scenario_name, args.seed)
    test_render(env)
