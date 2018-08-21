from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
import cyber_attack_simulator.envs.loader as loader
from cyber_attack_simulator.agents.sarsa import SarsaAgent
from analyser import Analyser

small_config = "configs/small.yaml"
small_med_config = "configs/small_med.yaml"
med_config = "configs/medium.yaml"


def main():
    exploit_probs = 1.0
    static = True
    # num_machines = 5
    # num_services = 3
    # config = loader.generate_config(num_machines, num_services)
    # config = loader.load_config(small_config)
    # config = loader.load_config(small_med_config)
    config = loader.load_config(med_config)

    env = CyberAttackSimulatorEnv(config, exploit_probs, static)

    num_episodes = 200
    max_steps = 500
    num_runs = 10
    verbose = True

    alpha = 0.1
    gamma = 0.9
    ucb_agent = SarsaAgent("UCB", alpha, gamma)
    egreedy_agent = SarsaAgent("egreedy", alpha, gamma)
    agents = [ucb_agent, egreedy_agent]

    analyser = Analyser(env, agents, num_episodes, max_steps, num_runs)
    analyser.run_analysis(verbose)
    analyser.plot_results()


if __name__ == "__main__":
    main()
