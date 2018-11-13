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
from cyber_attack_simulator.experiments.experiment_util import get_scenario


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def average_data(df):
    """ Average data over eval runs then scenario runs """
    avg_eval_df = df.groupby(["scenario", "agent", "run"]).mean().reset_index()
    run_df = avg_eval_df.groupby(["scenario", "agent"])
    run_err = run_df.sem().reset_index()
    run_avg = run_df.mean().reset_index()
    return run_avg, run_err


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


def main():
    if len(sys.argv) != 2:
        print("Usage: python agent_perf_eval_plotter.py <eval_file>")

    print("Witness the power of my highly trained agents, muhahaha!")
    results_df = import_data(sys.argv[1])

    avg_df, err_df = average_data(results_df)
    print("\nAverages:\n", avg_df)
    print("\nStd errors:\n", err_df)

    get_solved_proportions(results_df)


if __name__ == "__main__":
    main()
