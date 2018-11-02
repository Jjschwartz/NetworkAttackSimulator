from env_perf_exp import run_experiment
import cProfile
import sys


if __name__ == "__main__":

    M = int(sys.argv[1])
    S = int(sys.argv[2])
    actions_per_run = int(sys.argv[3])

    pr = cProfile.Profile()
    pr.enable()
    run_experiment(M, S, actions_per_run, 1)
    pr.disable()
    pr.print_stats()
