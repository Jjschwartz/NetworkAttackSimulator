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


def import_data(file_name):
    df = pd.read_csv(file_name)
    return df


def average_data(df):
    """ Average data over eval runs then scenario runs """
    avg_eval_df = df.groupby(["scenario", "agent", "run"]).mean().reset_index()
    run_df = avg_eval_df.groupby(["scenario", "agent"])
    run_err = run_df.sem().reset_index()
    print(run_df["reward"].sem())
    print(run_df["timesteps"].sem())
    print(run_df["reward"].max().reset_index())
    run_avg = run_df.mean().reset_index()
    return run_avg, run_err


def main():
    if len(sys.argv) != 2:
        print("Usage: python agent_perf_eval_plotter.py <eval_file>")

    print("Witness the power of my highly trained agents, muhahaha!")
    results_df = import_data(sys.argv[1])

    avg_df, err_df = average_data(results_df)
    print("\nAverages:\n", avg_df)
    print("\nStd errors:\n", err_df)


if __name__ == "__main__":
    main()
