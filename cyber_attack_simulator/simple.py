from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.agents.sarsa import SarsaAgent
from cyber_attack_simulator.agents.q_learning import QLearningAgent

small_config = "configs/small.yaml"
small_med_config = "configs/small_med.yaml"
med_config = "configs/medium.yaml"


def main():
    generate = True
    # network configuration params
    num_machines = 3
    num_services = 2
    exploit_probs = 0.8
    uniform = False
    restrictiveness = 1

    print("Simple test")
    if generate:
        print("Generating network configuration")
        print("\tnumber of machines =", num_machines)
        print("\tnumber of services =", num_services)
        print("\texploit success probability =", exploit_probs)
        print("\tuniform =", uniform)
        print("\tfirewall restrictiveness =", restrictiveness)
        env = CyberAttackSimulatorEnv.from_params(num_machines, num_services,
                                                  exploit_probs=exploit_probs,
                                                  uniform=uniform,
                                                  restrictiveness=restrictiveness)
    else:
        print("Loading network configuration from file")
        print("File={}")
        # 100 episodes, 100 steps
        # config_path = small_config
        # 100 episodes, 100 steps
        # config_path = small_med_config
        # 500 episodes, 200 steps
        config_path = med_config
        print("File={}".format(config_path))
        env = CyberAttackSimulatorEnv.from_file(config_path, exploit_probs=exploit_probs)

    num_episodes = 1000
    max_steps = 200
    visualize_policy = num_episodes // 1
    verbose = True

    alpha = 0.1
    gamma = 0.9
    c = 10.0
    ucb_sarsa = SarsaAgent("UCB", alpha, gamma, c)
    # egreedy_sarsa = SarsaAgent("egreedy", alpha, gamma)
    # ucb_q = QLearningAgent("UCB", alpha, gamma)
    # egreedy_q = QLearningAgent("egreedy", alpha, gamma)

    ucb_sarsa.train(env, num_episodes, max_steps, verbose, visualize_policy)
    exp_total_reward = ucb_sarsa.evaluate_agent(env, 100, max_steps)
    print("Expected Total Reward = {}".format(exp_total_reward))


if __name__ == "__main__":
    main()
