from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.envs.action import Action
from cyber_attack_simulator.envs.render import Viewer
import cyber_attack_simulator.envs.loader as loader


def main():
    E = 1
    M = 40
    config = loader.generate_config(M, E)
    env = CyberAttackSimulatorEnv(config)
    episode = []

    actions = [Action((0, 0), "exploit", 0),
               Action((1, 0), "exploit", 0),
               Action((2, 1), "exploit", 0),
               Action((2, 0), "exploit", 0)]

    state = env.reset()

    for a in actions:
        new_state, r, d, _ = env.step(a)
        episode.append((state, a, r, d))
        state = new_state

    Viewer(episode, env.network)


if __name__ == "__main__":
    main()
