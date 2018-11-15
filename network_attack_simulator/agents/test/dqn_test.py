from network_attack_simulator.envs.environment import NetworkAttackSimulator
from network_attack_simulator.agents.dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np


def main():
    # network configuration params
    num_machines = 3
    num_services = 1
    exploit_probs = 'mixed'
    uniform = False
    restrictiveness = 3

    print("Simple test")
    print("Generating network configuration")
    print("\tnumber of machines =", num_machines)
    print("\tnumber of services =", num_services)
    print("\texploit success probability =", exploit_probs)
    print("\tuniform =", uniform)
    print("\tfirewall restrictiveness =", restrictiveness)
    env = NetworkAttackSimulator.from_params(num_machines, num_services,
                                             r_sensitive=10, r_user=10,
                                             exploit_cost=1, scan_cost=1,
                                             exploit_probs=exploit_probs,
                                             uniform=uniform,
                                             restrictiveness=restrictiveness)

    num_episodes = 100
    max_steps = 100
    verbose = True
    agent = DQNAgent(env.get_state_size(), env.get_num_actions(),
                     epsilon_decay_lambda=0.001)

    ep_timesteps, ep_rewards, ep_times = agent.train(env, num_episodes=num_episodes,
                                                     max_steps=max_steps, verbose=verbose)

    # plot reward vs episode
    plt.subplot(131)
    plt.plot(list(range(num_episodes)), ep_rewards)
    plt.title("Rewards per episode")

    plt.subplot(132)
    plt.plot(np.cumsum(ep_times), ep_rewards)
    plt.title("Rewards over time")

    plt.subplot(133)
    plt.plot(list(range(num_episodes)), np.cumsum(ep_timesteps))
    plt.title("Cumulative Timesteps per episode")
    plt.show()


if __name__ == "__main__":
    main()
