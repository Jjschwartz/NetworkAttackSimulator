"""
For plotting the csv output from scaling.py
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cyber_attack_simulator.experiments.experiment_util import get_agent_label


def import_data(file_name):
    df = pd.read_csv(file_name)
    # df = df.loc[df.solved == True]      # noqa E3712
    return df


def average_data_over_eval_runs(df):
    avg_eval_df = df.groupby(["M", "S", "agent", "run"]).mean().reset_index()
    run_avg = avg_eval_df.groupby(["M", "S", "agent"]).mean().reset_index()
    run_err = avg_eval_df.groupby(["M", "S", "agent"]).sem().reset_index()
    return run_avg, run_err


def plot_solve_proportion(ax, df, var, label):

    agents = df.agent.unique()
    for agent in agents:
        print(agent)
        agent_df = df.loc[df["agent"] == agent]
        mean_agent_df, err_agent_df = average_data_over_eval_runs(agent_df)
        err = err_agent_df["solved"]
        solved = mean_agent_df["solved"]
        machines = mean_agent_df[var]
        agent_label = get_agent_label(agent)
        ax.plot(machines, solved, label=agent_label)
        ax.fill_between(machines, solved-err, solved+err, alpha=0.3)

    ax.set_xlabel(label)
    ax.set_ylabel("Mean solved proportion")
    # ax.set_ylim(top=int(df.mean_reward.max())+2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_solve_reward(ax, df, var, label):

    agents = df.agent.unique()
    for agent in agents:
        print(agent)
        agent_df = df.loc[df["agent"] == agent]
        mean_agent_df, err_agent_df = average_data_over_eval_runs(agent_df)
        err = err_agent_df["reward"]
        reward = mean_agent_df["reward"]
        machines = mean_agent_df[var]
        agent_label = get_agent_label(agent)
        ax.plot(machines, reward, label=agent_label)
        ax.fill_between(machines, reward-err, reward+err, alpha=0.3)

    ax.set_xlabel(label)
    ax.set_ylabel("Mean reward")
    # ax.set_ylim(top=int(df.mean_reward.max())+2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.legend()


def main():

    if len(sys.argv) != 3:
        print("Usage: python scaling_plotter.py <result_file>.csv plot_M")
        return 1

    print("Watch as it grows and grows and grows!")
    results_df = import_data(sys.argv[1])
    plot_machines = bool(int(sys.argv[2]))

    if plot_machines:
        var = "M"
        label = "Machines"
    else:
        var = "S"
        label = "Exploitable Services"

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(121)
    plot_solve_proportion(ax1, results_df, var, label)

    ax2 = fig.add_subplot(122)
    plot_solve_reward(ax2, results_df, var, label)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')

    fig.subplots_adjust(left=0.1, bottom=0.37, right=0.97, top=0.96, wspace=0.4, hspace=0.2)
    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
