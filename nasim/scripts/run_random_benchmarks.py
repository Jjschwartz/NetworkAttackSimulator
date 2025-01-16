"""This script runs the random agent for all benchmarks scenarios

The mean (+/- stdev) steps and reward are reported in table to stdout
(and to optional CSV file)

Usage
-----
$ python run_random_benchmarks.py [-n --num_cpus NUM_CPUS]
     [-o --output OUTPUT_FILENAME] [-s --num_seeds NUM_SEEDS]

"""
import os
import numpy as np
import multiprocessing as mp
from prettytable import PrettyTable

import nasim
from nasim.agents.random_agent import run_random_agent
from nasim.scenarios.benchmark import AVAIL_BENCHMARKS


def print_msg(msg):
    print(f"[PID={os.getpid()}] {msg}")


class Result:

    def __init__(self, name):
        self.name = name
        self.run_steps = []
        self.run_rewards = []

    def add(self, steps, reward):
        self.run_steps.append(steps)
        self.run_rewards.append(reward)

    def summarize(self):
        steps_mean = np.mean(self.run_steps)
        steps_std = np.std(self.run_steps)
        reward_mean = np.mean(self.run_rewards)
        reward_std = np.std(self.run_rewards)
        return steps_mean, steps_std, reward_mean, reward_std

    def get_formatted_summary(self):
        steps_mean, steps_std, reward_mean, reward_std = self.summarize()
        return (
            f"{steps_mean:.2f} +/- {steps_std:.2f}",
            f"{reward_mean:.2f} +/- {reward_std:.2f}"
        )


def run_scenario(args):
    scenario_name, seed = args
    print_msg(f"Running '{scenario_name}' scenario with seed={seed}")
    env = nasim.make_benchmark(scenario_name, seed, False, True, True)
    steps, total_reward, done = run_random_agent(env, verbose=False)
    return {
        "Name": scenario_name,
        "Seed": seed,
        "Steps": steps,
        "Total reward": total_reward
    }


def collate_results(results):
    scenario_results = {}
    for res in results:
        name = res["Name"]
        if name not in scenario_results:
            scenario_results[name] = Result(name)
        scenario_results[name].add(res["Steps"], res["Total reward"])
    return scenario_results


def output_results(results, output=None):
    headers = ["Scenario Name", "Steps", "Total Reward"]
    rows = []
    for name in AVAIL_BENCHMARKS:
        rows.append([
            name, *results[name].get_formatted_summary()
        ])

    table = PrettyTable(headers)
    for row in rows:
        table.add_row(row)

    if output is not None:
        with open(output, "w") as fout:
            fout.write(",".join(headers) + "\n")
            for row in rows:
                fout.write(",".join(row) + "\n")


def run_random_benchmark(num_cpus=1, num_seeds=10, output=None):
    run_args_list = []
    for name in AVAIL_BENCHMARKS:
        for seed in range(num_seeds):
            run_args_list.append((name, seed))

    with mp.Pool(num_cpus) as p:
        results = p.map(run_scenario, run_args_list)

    results = collate_results(results)
    output_results(results, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_cpus", type=int, default=1,
                        help="Number of CPUS to use in parallel (default=1)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="File name to output as CSV too")
    parser.add_argument("-s", "--num_seeds", type=int, default=10,
                        help=("Number of seeds to run for each scenario"
                              " (default=10)"))
    args = parser.parse_args()

    run_random_benchmark(**vars(args))
