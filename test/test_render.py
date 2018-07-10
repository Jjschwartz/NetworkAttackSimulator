from environment import CyberAttackSimulatorEnv
from action import Action
from render import Viewer


def main():
    E = 1
    M = 14
    env = CyberAttackSimulatorEnv(M, E)
    obs_sequence = []
    obs_sequence.append(env.reset())

    t_action = Action((1, 0), "exploit", 0)
    o, r, d, _ = env.step(t_action)
    obs_sequence.append(o)

    t_action = Action((2, 0), "exploit", 0)
    o, r, d, _ = env.step(t_action)
    obs_sequence.append(o)

    t_action = Action((3, 1), "exploit", 0)
    o, r, d, _ = env.step(t_action)
    obs_sequence.append(o)

    t_action = Action((3, 0), "exploit", 0)
    o, r, d, _ = env.step(t_action)
    obs_sequence.append(o)

    Viewer(obs_sequence)


if __name__ == "__main__":
    main()
