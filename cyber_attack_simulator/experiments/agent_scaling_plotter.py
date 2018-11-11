"""
For plotting the csv output from scaling.py
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def import_data(file_name):
    df = pd.read_csv(file_name)
    df = df.loc[df.solved == True]      # noqa E3712
    return df


def average_data_over_runs(df):
    avg_df = df.groupby(["agent", "M"]).mean().reset_index()
    return avg_df


def get_agent_label(agent_name):
    if agent_name == "dqn":
        return "DQN"
    elif agent_name == "td_egreedy":
        return r'Tabular Q-Learning $\epsilon$-greedy'
    elif agent_name == "td_ucb":
        return "Tabular Q-Learning UCB"
    else:
        return "unknown"


def plot_solve_time_vs_machines(ax, df):

    agents = df.agent.unique()
    for agent in agents:
        agent_df = df.loc[df["agent"] == agent]
        mean_agent_df = average_data_over_runs(agent_df)
        solve_times = mean_agent_df["solve_time"]
        err = agent_df["solve_time"].sem()
        machines = mean_agent_df["M"]
        label = get_agent_label(agent)
        ax.plot(machines, solve_times, label=label)
        ax.fill_between(machines, solve_times-err, solve_times+err, alpha=0.3)

    ax.set_xlabel("Machines", fontsize=12)
    ax.set_ylabel("Mean Solve Time (Sec)", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=12)


def plot_solve_reward_vs_machines(ax, df):

    agents = df.agent.unique()
    for agent in agents:
        agent_df = df.loc[df["agent"] == agent]
        mean_agent_df = average_data_over_runs(agent_df)
        mean_reward = mean_agent_df["mean_reward"]
        err_reward = agent_df["mean_reward"].sem()
        machines = mean_agent_df["M"]
        label = get_agent_label(agent)
        ax.plot(machines, mean_reward, label=label)
        ax.fill_between(machines, mean_reward-err_reward, mean_reward+err_reward,
                        alpha=0.3)

    ax.set_xlabel("Machines")
    ax.set_ylabel("Mean solution reward")
    ax.set_ylim(bottom=0, top=int(df.mean_reward.max())+2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()


def main():

    if len(sys.argv) != 2:
        print("Usage: python scaling_plotter.py <result_file>.csv")
        return 1

    print("Watch as it grows and grows and grows!")
    results_df = import_data(sys.argv[1])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot_solve_time_vs_machines(ax1, results_df)

    # ax2 = fig.add_subplot(122)
    # plot_solve_reward_vs_machines(ax2, results_df)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
