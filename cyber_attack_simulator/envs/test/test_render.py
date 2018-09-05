from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.envs.action import Action
from cyber_attack_simulator.envs.render import Viewer


proj_path = "/home/jonathon/Documents/Uni/COMP6801/CyberAttackSimulator/cyber_attack_simulator/"
small_config = proj_path + "configs/small.yaml"
small2_config = proj_path + "configs/small2.yaml"
small_med_config = proj_path + "configs/small_med.yaml"
med_config = proj_path + "configs/medium.yaml"
med_2_config = proj_path + "configs/medium_2_exposed.yaml"


def main():
    """
    Test rendering of a single episode using Viewer class in render module
    """
    # E = 1
    # M = 40
    # env = CyberAttackSimulatorEnv.from_params(M, E)

    env = CyberAttackSimulatorEnv.from_file(med_2_config)

    episode = []
    actions = [Action((0, 0), "exploit", 0),
               Action((1, 0), "exploit", 0),
               Action((2, 1), "exploit", 0),
               Action((2, 0), "exploit", 0)]

    state = env.reset()
    for a in actions:
        new_state, r, d = env.step(a)
        episode.append((state, a, r, d))
        state = new_state

    Viewer(episode, env.network)


if __name__ == "__main__":
    main()
