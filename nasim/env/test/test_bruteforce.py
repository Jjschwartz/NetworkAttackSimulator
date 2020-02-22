"""Runs bruteforce agent on environment using different seeds and
reports outcomes
"""
from nasim.env import make_benchmark_env
from nasim.agents.bruteforce_agent import run_bruteforce_agent

STEP_LIMIT = 1e6
NUM_SEEDS = 10
VERBOSE = False
LINE_BREAK = "-"*60


def run_test(env_name):
    complete = 0

    for seed in range(NUM_SEEDS):
        env = make_benchmark_env(env_name, seed)
        t, total_reward, done = run_bruteforce_agent(env, STEP_LIMIT, VERBOSE)
        complete += int(done)

    print(LINE_BREAK)
    print("TEST COMPLETE:")
    print(f"Runs complete = {complete} out of {NUM_SEEDS}")
    print(LINE_BREAK)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    args = parser.parse_args()

    run_test(args.env_name)
