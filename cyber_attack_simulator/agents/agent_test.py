import matplotlib.pyplot as plt
import numpy as np


def plot_episode_results(episode_timesteps, episode_rewards, optimal_actions,
                         avg_interval):

    episodes = range(len(episode_timesteps))

    avg_episode_timesteps = []
    avg_episode_rewards = []
    for i in range(avg_interval, len(episodes)):
        start = episode_timesteps[i - avg_interval]
        end = episode_timesteps[i]
        avg_episode_timesteps.append((end - start) / avg_interval)
        avg_reward = np.average(episode_rewards[i - avg_interval: i])
        avg_episode_rewards.append(avg_reward)

    plt.subplot(221)
    plt.plot(episodes, episode_timesteps)
    plt.xlabel("Episodes")
    plt.ylabel("Timesteps")

    plt.subplot(222)
    avg_episodes = list(range(avg_interval, len(episodes)))
    plt.plot(avg_episodes, avg_episode_timesteps)
    plt.plot(avg_episodes, np.full(len(avg_episodes), optimal_actions))
    plt.xlabel("Episodes")
    plt.ylabel("Average Timesteps")

    plt.subplot(223)
    plt.plot(episodes, episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")

    plt.subplot(224)
    avg_episodes = list(range(avg_interval, len(episodes)))
    plt.plot(avg_episodes, avg_episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Rewards")

    plt.show()


def test_agent(env, agent, num_runs, avg_interval=10):

    run_results_timesteps = []
    run_results_rewards = []
    for run in range(num_runs):
        t, r = agent.train()
        run_results_timesteps.append(t)
        run_results_rewards.append(r)
    avg_run_timesteps = np.average(run_results_timesteps, axis=0)
    avg_run_rewards = np.average(run_results_rewards, axis=0)

    opt_num_actions = env.optimal_num_actions()
    print("Optimal number actions = {0}".format(opt_num_actions))

    episode = agent.generate_episode()
    env.render_episode(episode)
    plot_episode_results(avg_run_timesteps, avg_run_rewards, opt_num_actions,
                         avg_interval)
