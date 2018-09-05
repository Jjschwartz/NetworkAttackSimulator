from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.envs.action import Action
from cyber_attack_simulator.envs.render import Viewer
# Must import matplotlib this way for compatibility with TKinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


proj_path = "/home/jonathon/Documents/Uni/COMP6801/CyberAttackSimulator/cyber_attack_simulator/"
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


def test_render_generated_network(nM, nS):
    E = 1
    M = 40
    env = CyberAttackSimulatorEnv.from_params(M, E)
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


def test_render_episode():

    actions = [Action((0, 0), "exploit", 0),
               Action((1, 0), "exploit", 0),
               Action((2, 1), "exploit", 0),
               Action((2, 0), "exploit", 0)]

    env = CyberAttackSimulatorEnv.from_file(med_2_config)
    episode = generate_episode(env, actions)
    viewer = Viewer(env.network)
    viewer.render_episode(episode)
    viewer.render_episode(episode)


def main():
    """
    Test rendering of a single episode using Viewer class in render module
    """
    test_render_generated_network(40, 1)
    test_render_episode()


if __name__ == "__main__":
    main()
