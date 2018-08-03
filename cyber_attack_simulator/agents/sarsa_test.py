from agent_test import test_agent
from sarsa import EGreedySarsaAgent
from sarsa import UCBSarsaAgent
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv


def main():
    num_machines = 10
    num_services = 1
    env = CyberAttackSimulatorEnv(num_machines, num_services)

    num_episodes = 300
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
