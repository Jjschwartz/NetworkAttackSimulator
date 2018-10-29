from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.agents.sarsa import SarsaAgent
from cyber_attack_simulator.agents.q_learning import QLearningAgent
from analyser import Analyser

small_config = "configs/small.yaml"
small_med_config = "configs/small_med.yaml"
med_config = "configs/medium.yaml"


def main():
    generate_env = False
    exploit_probs = 1.0

    if generate_env:
        num_machines = 5
        num_services = 3
        env = CyberAttackSimulatorEnv.from_params(num_machines, num_services, exploit_probs)
    else:
        # 100 episodes, 100 steps
        config_path = small_config
        # 100 episodes, 100 steps
        # config_path = small_med_config
        # 500 episodes, 200 steps
        # config_path = med_config
        env = CyberAttackSimulatorEnv.from_file(config_path, exploit_probs)

    num_episodes = 500
    max_steps = 200
    num_runs = 50
    verbose = False

    alpha = 0.1
    gamma = 0.9
    ucb_sarsa = SarsaAgent("UCB", alpha, gamma)
    egreedy_sarsa = SarsaAgent("egreedy", alpha, gamma)
    ucb_q = QLearningAgent("UCB", alpha, gamma)
    egreedy_q = QLearningAgent("egreedy", alpha, gamma)
    agents = [ucb_sarsa, egreedy_sarsa, ucb_q, egreedy_q]
    # agents = [ucb_sarsa, egreedy_sarsa]
    # agents = [ucb_q]

    # agents = []
    # for i in range(1, 10):
    #     agents.append(SarsaAgent("UCB", i*alpha, gamma))

    analyser = Analyser(env, agents, num_episodes, max_steps, num_runs)
    analyser.run_analysis(verbose)
    analyser.output_results()
    analyser.plot_results()


if __name__ == "__main__":
    main()
