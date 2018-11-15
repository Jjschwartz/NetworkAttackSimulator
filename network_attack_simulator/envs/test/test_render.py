from network_attack_simulator.envs.environment import NetworkAttackSimulator
from network_attack_simulator.envs.environment import EXPLOIT_COST
from network_attack_simulator.envs.action import Action
from network_attack_simulator.envs.render import Viewer
# Must import matplotlib this way for compatibility with TKinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # noqa


proj_path = "/home/jonathon/Documents/Uni/COMP6801/CyberAttackSimulator/network_attack_simulator/"
small_config = proj_path + "configs/small.yaml"
small2_config = proj_path + "configs/small2.yaml"
small_med_config = proj_path + "configs/small_med.yaml"
med_config = proj_path + "configs/medium.yaml"
med_2_config = proj_path + "configs/medium_2_exposed.yaml"


def generate_episode(env, actions):
    episode = []
    init_state = env.reset()
    state = init_state
    for a in actions:
        new_state, r, d = env.step(a)
        episode.append((state, a, r, d))
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


def main():
    """
    Test rendering of a single episode using Viewer class in render module
    """
    generated_env = True
    config_file = small_config

    actions = [Action((0, 0), EXPLOIT_COST, "exploit", 0),
               Action((1, 0), EXPLOIT_COST, "exploit", 0),
               Action((2, 1), EXPLOIT_COST, "exploit", 0),
               Action((2, 0), EXPLOIT_COST, "exploit", 0)]

    if generated_env:
        E = 1
        M = 40
        env = NetworkAttackSimulator.from_params(M, E)
    else:
        env = NetworkAttackSimulator.from_file(config_file)

    test_render_asci(env, actions)
    test_render_readable(env, actions)
    test_render_network_graph(env)
    test_render_episode(env, actions)


if __name__ == "__main__":
    main()
