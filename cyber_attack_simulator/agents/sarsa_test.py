from sarsa import EGreedySarsaAgent
from sarsa import UCBSarsaAgent
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
import matplotlib.pyplot as plt
import numpy as np


def plot_timesteps_per_epsiode(cumulative_t_per_e, optimal_action_num):

    episodes = range(len(cumulative_t_per_e))
    avg_t_per_e = []

    for i in range(10, len(cumulative_t_per_e)):
        start = cumulative_t_per_e[i - 10]
        end = cumulative_t_per_e[i]
        avg_t_per_e.append((end - start) / 10)

    plt.subplot(121)
    plt.plot(episodes, cumulative_t_per_e)
    plt.xlabel("Episodes")
    plt.ylabel("Timesteps")

    plt.subplot(122)
    avg_episodes = list(range(10, len(cumulative_t_per_e)))
    plt.plot(avg_episodes, avg_t_per_e)
    plt.plot(avg_episodes, np.full(len(avg_episodes), optimal_action_num))
    plt.xlabel("Episodes")
    plt.ylabel("Avg Timesteps")
    plt.show()


def main():
    num_machines = 10
    num_services = 3
    env = CyberAttackSimulatorEnv(num_machines, num_services)

    num_episodes = 100
    max_steps = 200
    alpha = 0.1
    gamma = 0.9
    agent = EGreedySarsaAgent(env, num_episodes, max_steps, alpha, gamma)
    # agent = UCBSarsaAgent(env, num_episodes, max_steps, alpha, gamma)

    num_runs = 1
    run_results = []
    for run in range(num_runs):
        run_results.append(agent.train())
    avg_t_per_e = np.average(run_results, axis=0)

    opt_num_actions = env.optimal_num_actions()
    print("Optimal number actions = {0}".format(opt_num_actions))

    episode = agent.generate_episode()
    env.render_episode(episode)
    plot_timesteps_per_epsiode(avg_t_per_e, opt_num_actions)


if __name__ == "__main__":
    main()
