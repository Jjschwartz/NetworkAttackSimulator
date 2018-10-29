"""
For plotting the csv output from experiment.py
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# smoothing window
WINDOW = 100


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def average_data_over_runs(df):
    """ Average the data for each scenario and agent over runs """
    avg_df = df.groupby(["scenario", "agent", "episode"]).mean().reset_index()
    return avg_df


def get_scenario_df(df, scenario):
    """ Returns dataframe containing only data for given scenario """
    scenario_df = df.loc[df["scenario"] == scenario]
    return scenario_df


def smooth_reward(rewards, I):
    """ Smooth the rewards by averaging over the last I episodes """
    episodes = range(len(rewards))
    avg_rewards = []
    for i in range(I, len(episodes)):
        avg_reward = np.average(rewards[i - I: i])
        avg_rewards.append(avg_reward)
    return avg_rewards


def plot_average_reward_per_episode(ax, scenario_df):

    ax.set_title("Average reward per episode")
    agents = scenario_df.agent.unique()
    for agent in agents:
        agent_df = scenario_df.loc[scenario_df["agent"] == agent]
        rewards = agent_df["rewards"]
        avg_rewards = smooth_reward(rewards, WINDOW)
        episodes = list(range(WINDOW, len(rewards)))
        ax.plot(episodes, avg_rewards, label=agent)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.legend()


def plot_average_timesteps_per_episode(ax, scenario_df):
    pass


def plot_average_reward_vs_time(ax, scenario_df):

    ax.set_title("Average reward vs time")
    agents = scenario_df.agent.unique()
    for agent in agents:
        agent_df = scenario_df.loc[scenario_df["agent"] == agent]
        rewards = agent_df["rewards"]
        avg_rewards = smooth_reward(rewards, WINDOW)
        episodes = list(range(WINDOW, len(rewards)))
        ax.plot(episodes, avg_rewards, label=agent)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.legend()


def main():

    if len(sys.argv) != 2:
        print("Usage: python result_plotter.py <result_file>.csv")
        return 1

    print("Feast your eyes on the wonders of the future!")
    results_df = import_data(sys.argv[1])
    avg_df = average_data_over_runs(results_df)
    # smooth data
    scenarios = avg_df.scenario.unique()
    fig, axes = plt.subplots(nrows=len(scenarios), ncols=1, squeeze=False)

    for i, scenario in enumerate(scenarios):
        ax = axes[i, 0]
        scenario_df = get_scenario_df(avg_df, scenario)
        plot_average_reward_per_episode(ax, scenario_df)

    plt.show()


if __name__ == "__main__":
    main()
