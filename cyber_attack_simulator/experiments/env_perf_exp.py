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
UNIFORM = True
EXPLOIT_PROB = 1.0
# allow all services
RVE = 2000


def get_random_action(action_space):
    return np.random.choice(action_space)


def load_env(M, S, seed):
    return Cyber.from_params(M, S, restrictiveness=RVE, exploit_probs=EXPLOIT_PROB, seed=seed)


def run_experiment(M, S, actions_per_run, run):

    # so actions performed are the same for each run
    np.random.seed(1)

    print("\tRun = {}".format(run))
    load_time_start = time.time()
    env = load_env(M, S, run)
    load_time = time.time() - load_time_start
    action_space = np.array(env.action_space)
    # actions = np.random.choice(action_space, size=actions_per_run)
    actions = action_space
    s = env.reset()
    reset_time = 0

    start_time = time.time()
    for t in range(actions_per_run):
        # a = actions[t]
        a = actions[t % len(actions)]
        s, _, done = env.step(a)
        if done:
            reset_start_time = time.time()
            env.reset()
            reset_time += (time.time() - reset_start_time)
        # if t % (actions_per_run // 5) == 0:
        #     print("\t\tActions performed = {}".format(t))

    a_per_sec = actions_per_run / ((time.time() - start_time) - reset_time)
    t_per_a = ((time.time() - start_time) - reset_time) / actions_per_run

    print("Not reachable count {} {:.4f}".format(env.not_reachable_count,
          env.not_reachable_count / actions_per_run))
    print("Failure count {} {:.4f}".format(env.failure_count,
          env.failure_count / actions_per_run))
    print("perform time {:.4f}".format(env.perform_time))
    print("update time {:.4f}".format(env.update_time))

    return a_per_sec, t_per_a, load_time


def write_result(M, S, run, a_per_sec, t_per_a, load_time, result_file):
    result_file.write("{},{},{},{},{},{}\n".format(M, S, run, a_per_sec, t_per_a, load_time))


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

    if len(sys.argv) == 11 and sys.argv[10] == "1":
        print("Appending to", sys.argv[1])
        result_file = open(sys.argv[1], 'a+', buffering=1)
    else:
        print("Writing to new file", sys.argv[1])
        result_file = open(sys.argv[1], 'w', buffering=1)
        # write header line
        write_result("M", "S", "run", "a_per_sec", "t_per_a", "load_time", result_file)

    for M in range(minM, maxM + 1, intM):
        if M != minM:
            M -= (M % intM)
        for S in range(minS, maxS + 1, intS):
            if S != minS:
                S -= (S % intS)
            print("\nRunning experiment with M={} S={}".format(M, S))
            run_results_actions = np.empty(runs, dtype=float)
            run_results_time = np.empty(runs, dtype=float)
            run_results_load = np.empty(runs, dtype=float)
            for run in range(runs):
                results = run_experiment(M, S, actions_per_run, run)
                write_result(M, S, run, results[0], results[1], results[2], result_file)
                run_results_actions[run] = results[0]
                run_results_time[run] = results[1]
                run_results_load[run] = results[2]
            avg_a_per_sec = np.mean(run_results_actions)
            avg_t_per_a = np.mean(run_results_time)
            avg_load_time = np.mean(run_results_load)
            print("\tAverage actions per second = {:.6f}".format(avg_a_per_sec))
            print("\tAverage time per action = {:.6f}".format(avg_t_per_a))
            print("\tAverage load time = {:.6f}".format(avg_load_time))

    return 0


if __name__ == "__main__":
    main()
