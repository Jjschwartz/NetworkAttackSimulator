"""
For plotting the csv output from agent_perf_exp.py

Specifically, mean episode reward vs episode for different agents
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cyber_attack_simulator.experiments.experiment_util import get_agent_label
from cyber_attack_simulator.experiments.experiment_util import get_scenario_max

# smoothing window
WINDOW = 10


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
    avg_rewards = [rewards[0]]
    for i in range(1, len(episodes)):
        interval = min(i, I)
        avg_reward = np.average(rewards[i - interval: i])
        avg_rewards.append(avg_reward)
    return avg_rewards


def plot_average_reward_per_episode(ax, scenario_df, max_reward, show_legend):

    # ax.set_title("Average reward per episode")
    agents = scenario_df.agent.unique()
    max_episodes = 0
    for agent in agents:
        agent_df = scenario_df.loc[scenario_df["agent"] == agent]
        avg_df = average_data_over_runs(agent_df)
        rewards = avg_df["rewards"]
        avg_rewards = smooth_reward(rewards, WINDOW)
        err = agent_df.groupby(["scenario", "agent", "episode"]).sem().reset_index()["rewards"]
        episodes = list(range(0, len(rewards)))
        if len(episodes) > max_episodes:
            max_episodes = len(episodes)
        ax.plot(episodes, avg_rewards, label=get_agent_label(agent))
        ax.fill_between(episodes, avg_rewards-err, avg_rewards+err, alpha=0.4)

    episodes = list(range(0, max_episodes))
    max_rewards = np.full(max_episodes, max_reward)
    ax.plot(episodes, max_rewards, label="Theoretical Max", linestyle="--")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Mean episode reward")
    # if show_legend:
    #     ax.legend(loc=4)


def main():

    if len(sys.argv) != 2:
        print("Usage: python agent_perf_plotter.py <result_file>.csv")
        return 1

    print("Feast your eyes on the wonders of the future!")
    results_df = import_data(sys.argv[1])
    scenarios = results_df.scenario.unique()
    # fig, axes = plt.subplots(nrows=len(scenarios), ncols=1, squeeze=False)

    fig = plt.figure(1, figsize=(8, 8))
    # +1 for legend
    plot_count = len(scenarios) + 1
    rows = (plot_count + 1) // 2
    cols = 1 if plot_count == 1 else 2
    title_vals = ["a)", "b)", "c)", "d)", "e)", "f)"]

    for i, scenario in enumerate(scenarios):
        if i == 0:
            ax = fig.add_subplot(rows, cols, i + 1)
        else:
            ax = fig.add_subplot(rows, cols, i + 1)
        # ax = axes[i, 0]
        ax.set_title((title_vals[i]), loc='left')
        show_legend = True if i == 0 else False
        scenario_df = get_scenario_df(results_df, scenario)
        scenario_max = get_scenario_max(scenario)
        plot_average_reward_per_episode(ax, scenario_df, scenario_max, show_legend)

    handles, labels = ax.get_legend_handles_labels()
    ax_end = fig.add_subplot(rows, cols, plot_count)
    ax_end.legend(handles, labels, loc="upper center")
    ax_end.axis('off')
    # fig.legend(handles, labels, loc='lower right')

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
