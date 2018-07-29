from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.envs.action import Action
from cyber_attack_simulator.envs.render import Viewer


def main():
    E = 1
    M = 40
    env = CyberAttackSimulatorEnv(M, E)
    episode = []

    actions = [Action((0, 0), "exploit", 0),
               Action((1, 0), "exploit", 0),
               Action((2, 1), "exploit", 0),
               Action((2, 0), "exploit", 0)]

    state = env.reset()

    for a in actions:
        new_state, r, d, _ = env.step(a)
        episode.append((state, a, r))
        state = new_state

    Viewer(episode, env.network)


if __name__ == "__main__":
    main()
