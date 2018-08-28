from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.agents.sarsa import SarsaAgent
from agent_test import test_agent

small_config = "../configs/small.yaml"
small_med_config = "../configs/small_med.yaml"
med_config = "../configs/medium.yaml"


def main():
    exploit_probs = 1.0
    static = True
    num_machines = 5
    num_services = 3
    env = CyberAttackSimulatorEnv.from_params(num_machines, num_services, exploit_probs, static)

    # config_path = small_config
    # config_path = small_med_config
    # config_path = med_config
    # env = CyberAttackSimulatorEnv.from_file(config_path, exploit_probs, static)

    num_episodes = 500
    max_steps = 200
    alpha = 0.1
    gamma = 0.9
    sarsa_agent = SarsaAgent("egreedy", alpha, gamma)

    num_runs = 1
    test_agent(env, sarsa_agent, num_runs, num_episodes, max_steps)


if __name__ == "__main__":
    main()
