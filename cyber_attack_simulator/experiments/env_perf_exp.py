"""
Experiment for testing the performance of the environment.

Performance metrics:
    - Actions per second
    - Load time (i.e. average time long it takes to start a new episode)

Variables:
    - Number of machines
    - Number of services
    - Number of subnets
"""
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv as Cyber
import numpy as np
import time
import sys


# Environment constants used for all experiments
UNIFORM = False
EXPLOIT_PROB = "mixed"
RVE = 3


def get_random_action(action_space):
    return np.random.choice(action_space)


def load_env(M, S, seed):
    return Cyber.from_params(M, S, restrictiveness=RVE, exploit_probs=EXPLOIT_PROB, seed=seed)


def run_experiment(M, S, actions_per_run, run):

    print("\tRun = {}".format(run))
    load_time_start = time.time()
    env = load_env(M, S, run)
    load_time = time.time() - load_time_start
    action_space = np.array(env.action_space)
    # action_space = np.random.choice(action_space, S)
    start_time = time.time()
    s = env.reset()

    for t in range(actions_per_run):
        a = get_random_action(action_space)
        s, _, done = env.step(a)
        if done:
            env.reset()
        if t % (actions_per_run // 5) == 0:
            print("\t\tActions performed = {}".format(t))

    a_per_sec = actions_per_run / (time.time() - start_time)
    return a_per_sec, load_time


def write_result(M, S, run, a_per_sec, load_time, result_file):
    result_file.write("{},{},{},{},{}\n".format(M, S, run, a_per_sec, load_time))


def main():

    if len(sys.argv) != 10 and len(sys.argv) != 11:
        print("Usage: python env_perf_exp.py outfile minM maxM intM minS maxS "
              + "intS action_per_run runs [1/0]")
        print("Where the [1/0] is for whether to append to file or not")
        return 1

    print("Welcome to your friendly environment performance experiment runner")
    minM = int(sys.argv[2])
    maxM = int(sys.argv[3])
    intM = int(sys.argv[4])
    minS = int(sys.argv[5])
    maxS = int(sys.argv[6])
    intS = int(sys.argv[7])
    actions_per_run = int(sys.argv[8])
    runs = int(sys.argv[9])
    print("Running with the following parameters:")
    print("\tmachine min={} max={} interval={}".format(minM, maxM, intM))
    print("\tservices min={} max={} interval={}".format(minS, maxS, intS))
    print("\tactions per run={} runs={}".format(actions_per_run, runs))

    if len(sys.argv) == 8 and sys.argv[7] == "1":
        print("Appending to", sys.argv[1])
        result_file = open(sys.argv[1], 'a+')
    else:
        print("Writing to new file", sys.argv[1])
        result_file = open(sys.argv[1], 'w')
        # write header line
        write_result("M", "S", "run", "a_per_sec", "load_time", result_file)

    for M in range(minM, maxM + 1, intM):
        for S in range(minS, maxS + 1, intS):
            print("\nRunning experiment with M={} S={}".format(M, S))
            run_results_actions = np.empty(runs, dtype=float)
            run_results_load = np.empty(runs, dtype=float)
            for run in range(runs):
                a_per_sec, load_time = run_experiment(M, S, actions_per_run, run)
                write_result(M, S, run, a_per_sec, load_time, result_file)
                run_results_actions[run] = a_per_sec
                run_results_load[run] = load_time
            avg_a_per_sec = np.mean(run_results_actions)
            avg_load_time = np.mean(run_results_load)
            print("\tAverage actions per action = {:.6f}".format(avg_a_per_sec))
            print("\tAverage load time = {:.6f}".format(avg_load_time))

    return 0


if __name__ == "__main__":
    main()
