from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from agent_test import test_agent
from sarsa import EGreedySarsaAgent
from sarsa import UCBSarsaAgent


def main():
    num_machines = 5
    num_services = 3
    exploit_probs = 1.0
    static = True
    env = CyberAttackSimulatorEnv(num_machines, num_services, exploit_probs,
                                  static)
    print(env)

    num_episodes = 500
    max_steps = 500
    alpha = 0.1
    gamma = 0.9
    greedy_agent = EGreedySarsaAgent(env, num_episodes, max_steps, alpha,
                                     gamma)
    ucb_agent = UCBSarsaAgent(env, num_episodes, max_steps, alpha, gamma)

    num_runs = 1
    test_agent(env, ucb_agent, num_runs)


if __name__ == "__main__":
    main()
