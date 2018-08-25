from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
import cyber_attack_simulator.envs.loader as loader
from cyber_attack_simulator.agents.sarsa import SarsaAgent
from cyber_attack_simulator.agents.q_learning import QLearningAgent

small_config = "configs/small.yaml"
small_med_config = "configs/small_med.yaml"
med_config = "configs/medium.yaml"


def main():
    exploit_probs = 1.0
    static = True
    num_machines = 15
    num_services = 5
    config = loader.generate_config(num_machines, num_services)
    # 100 episodes, 100 steps
    # config = loader.load_config(small_config)
    # 100 episodes, 100 steps
    # config = loader.load_config(small_med_config)
    # 500 episodes, 200 steps
    # config = loader.load_config(med_config)

    env = CyberAttackSimulatorEnv(config, exploit_probs, static)

    num_episodes = 1500
    max_steps = 1000
    window = 500
    verbose = True

    alpha = 0.1
    gamma = 0.9
    c = 10.0
    ucb_sarsa = SarsaAgent("UCB", alpha, gamma, c)
    # egreedy_sarsa = SarsaAgent("egreedy", alpha, gamma)
    # ucb_q = QLearningAgent("UCB", alpha, gamma)
    # egreedy_q = QLearningAgent("egreedy", alpha, gamma)

    ucb_sarsa.train(env, num_episodes, max_steps, window, verbose)


if __name__ == "__main__":
    main()
