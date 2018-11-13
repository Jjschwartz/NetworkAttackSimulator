"""
For summarizing and displaying the results generating by evaluating trained agents
using agent_perf_exp.py

Specifically, displays mean episode timesteps and reward for each agent for a given
scenario

Timesteps and rewards are averaged over the number of evaluation runs (evalrun) and then
over each run of the given scenario
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cyber_attack_simulator.experiments.experiment_util import get_scenario
from cyber_attack_simulator.experiments.experiment_util import get_agent_label


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def average_data(df):
    """ Average data over eval runs then scenario runs """
    df = df.astype({"solved": int})
    avg_eval_df = df.groupby(["scenario", "agent", "run"]).mean().reset_index()
    eval_err = df.groupby(["scenario", "agent", "run"]).sem().reset_index()
    # run_err = avg_eval_df.groupby(["scenario", "agent"]).sem().reset_index()
    run_avg = avg_eval_df.groupby(["scenario", "agent"]).mean().reset_index()
    return run_avg, eval_err


def max_data_over_eval_runs(df):
    df = df.astype({"solved": int})
    max_eval_df = df.groupby(["scenario", "agent", "run"]).max().reset_index()
    eval_err = df.groupby(["scenario", "agent", "run"]).sem().reset_index()
    run_max = max_eval_df.groupby(["scenario", "agent"]).max().reset_index()
    return run_max, eval_err


def get_solved_proportions(df):

    agents = df.agent.unique()
    scenario = df.scenario.unique()[0]
    runs = df.run.unique()
    max_steps = get_scenario(scenario)["steps"]

    print("Scenario={} max steps={}".format(scenario, max_steps))
    for agent in agents:
        solved = 0
        agent_df = df[df["agent"] == agent]
        for run in runs:
            run_df = agent_df[agent_df["run"] == run]
            solved_runs = run_df[run_df["timesteps"] < 50]
            if solved_runs.shape[0] > 0:
                solved += 1
        print("Agent={} solved={} proportion={}".format(agent, solved, solved / len(runs)))


def plot_solve_proportions(ax, df):

    scenarios = df.scenario.unique()
    agents = df.agent.unique()

    # the width of the bars
    width = 0.2
    # the x locations for the groups
    ind = np.arange(len(scenarios))
    rects = []
    labels = []

    for i, agent in enumerate(agents):
        print(agent)
        agent_df = df.loc[df["agent"] == agent]
        mean_agent_df, err_agent_df = average_data(agent_df)
        err = err_agent_df["solved"]
        solved = mean_agent_df["solved"]
        rect = ax.bar(ind + (i * width), solved, width, yerr=err)
        label = get_agent_label(agent)
        rects.append(rect)
        labels.append(label)

    ax.set_ylabel("Mean solved proportion")
    ax.set_xlabel("Scenario")
    ax.set_xticks(ind + (len(scenarios) * width) / 2)
    ax.set_xticklabels(scenarios)
    # ax.legend(rects, labels)


def plot_mean_rewards(ax, df):

    scenarios = df.scenario.unique()
    agents = df.agent.unique()

    # the width of the bars
    width = 0.2
    # the x locations for the groups
    ind = np.arange(len(scenarios))
    rects = []
    labels = []

    for i, agent in enumerate(agents):
        print(agent)
        agent_df = df.loc[df["agent"] == agent]
        # mean_agent_df, err_agent_df = average_data(agent_df)
        # err = err_agent_df["reward"]
        # solved = mean_agent_df["reward"]
        # rect = ax.bar(ind + (i * width), solved, width, yerr=err)
        max_agent_df, err_agent_df = max_data_over_eval_runs(agent_df)
        err = err_agent_df["reward"]
        reward = max_agent_df["reward"]
        rect = ax.bar(ind + (i * width), reward, width, yerr=err)
        label = get_agent_label(agent)
        rects.append(rect)
        labels.append(label)

    ax.set_ylabel("Max reward")
    ax.set_xlabel("Scenario")
    ax.set_xticks(ind + (len(scenarios) * width) / 2)
    ax.set_xticklabels(scenarios)
    # ax.legend(rects, labels)


def main():
    if len(sys.argv) != 2:
        print("Usage: python agent_perf_eval_plotter.py <eval_file>")

    print("Witness the power of my highly trained agents, muhahaha!")
    results_df = import_data(sys.argv[1])

    # avg_df, err_df = average_data(results_df)
    # print("\nAverages:\n", avg_df)
    # print("\nStd errors:\n", err_df)

    # get_solved_proportions(results_df)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    plot_solve_proportions(ax1, results_df)

    ax2 = fig.add_subplot(122)
    plot_mean_rewards(ax2, results_df)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
