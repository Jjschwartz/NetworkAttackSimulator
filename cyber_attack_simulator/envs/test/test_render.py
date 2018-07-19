from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.envs.action import Action
from cyber_attack_simulator.envs.render import Viewer


def main():
    E = 1
    M = 40
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
