"""
For plotting the csv output from experiment.py
"""
import sys
import pandas as pd


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def average_data_over_runs(df):
    """ Average the data for each scenario and agent over runs """
    avg_df = df.groupby(["scenario", "agent", "episode"]).mean().reset_index()
    return avg_df


def smooth_reward(df, I):
    """ Smooth the rewards by averaging over the last I episodes """
    pass


def plot_average_reward_per_episode(ax, scenario_df):

    ax.set_title("Average reward per episode")
    agents = scenario_df.agent.unique()
    for agent in agents:
        agent_df = scenario_df.loc[scenario_df["agent"] == agent]
        rewards = agent_df["rewards"]
        episodes = len(rewards)
        ax.plot(episodes, rewards, label=agent)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average reward")
    ax.legend()


def plot_average_timesteps_per_episode():
    pass


def plot_average_reward_vs_time():
    pass



def main():

    if len(sys.argv) != 2:
        print("Usage: python result_plotter.py <result_file>.csv")
        return 1

    print("Feast your eyes on the wonders of the future!")
    result_file = open(sys.argv[1], "r")



if __name__ == "__main__":
    main()
