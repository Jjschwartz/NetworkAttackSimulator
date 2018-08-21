from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
import cyber_attack_simulator.envs.loader as loader
from agent_test import test_agent
from sarsa import SarsaAgent

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
    config = loader.load_config(small_med_config)
    # config = loader.load_config(med_config)

    env = CyberAttackSimulatorEnv(config, exploit_probs, static)

    num_episodes = 500
    max_steps = 200
    alpha = 0.1
    gamma = 0.9
    ucb_agent = SarsaAgent("egreedy", alpha, gamma)

    num_runs = 1
    test_agent(env, ucb_agent, num_runs, num_episodes, max_steps)


if __name__ == "__main__":
    main()
