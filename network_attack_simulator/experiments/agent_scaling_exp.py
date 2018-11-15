"""
This module contains functions for performing an analysis of the performance of different
algorithms versus the problem size (number of machines and exploits)
"""
from network_attack_simulator.envs.environment import CyberAttackSimulatorEnv as Cyber
from network_attack_simulator.experiments.experiment_util import get_agent
import time
import sys
import numpy as np


# Experiment agents to run
agent_list = []
agent_list.append("td_egreedy")
agent_list.append("td_ucb")
agent_list.append("dqn")
# agent_list.append("random")

# Experiment parameters
RUNS = 5
MACHINE_MIN = 3
MACHINE_MAX = 43
MACHINE_INTERVAL = 5
SERVICE_MIN = 5
SERVICE_MAX = 5
SERVICE_INTERVAL = 1
MAX_EPISODES = 1000000
EPISODE_INTERVALS = 100
MAX_STEPS = 500
EVAL_RUNS = 100
OPTIMALITY_PROPORTION = 7.0
OPTIMAL_SOLVE_PROPORTION = 1.0  # proportion of trials to solve to be considered solved
TIMEOUT = 120
VERBOSE = False

# Environment constants
# SERVICES = 5
RVE = 3         # restrictiveness
UNIFORM = False
EXPLOIT_PROB = 0.7
R_SENS = R_USR = 10
COST_EXP = COST_SCAN = 1


def run_scaling_analysis(agent_type, result_file):

    print("\nRunning scaling analysis for agent: \n\t {0}".format(str(agent_type)))
    for m in range(MACHINE_MIN, MACHINE_MAX + 1, MACHINE_INTERVAL):
        for s in range(SERVICE_MIN, SERVICE_MAX + 1, SERVICE_INTERVAL):
            solve_times = []
            one_run_solved = False
            print("\n>> Machines={0} Services={1}".format(m, s))
            for t in range(RUNS):
                print("\tRun {0} of {1}".format(t+1, RUNS))
                env = Cyber.from_params(m, s,
                                        r_sensitive=R_SENS,  r_user=R_USR,
                                        exploit_cost=COST_EXP, scan_cost=COST_SCAN,
                                        restrictiveness=RVE, exploit_probs=EXPLOIT_PROB, seed=t)
                agent = get_agent(agent_type, "default", env)
                solved, solve_time, mean_reward = run_till_solved(agent, env)
                solve_times.append(solve_time)
                write_results(result_file, agent_type, m, s, t, solved, solve_time, mean_reward)
                print("\t\tsolved={} -- solve_time={:.2f} -- mean_reward={:.2f}"
                      .format(solved, solve_time, mean_reward))
                if solved:
                    one_run_solved = True
            print(">> Average solve time = {:.4f}".format(np.mean(solve_times)))
            if not one_run_solved:
                print(">> No environments solved so not testing larger sizes")
                break


def write_results(result_file, agent, M, S, run, solved, solve_time, mean_reward):
    """ Write results to file """
    # agent,machines,services,run,solved,solve_time, total_reward
    result_file.write("{0},{1},{2},{3},{4},{5:.2f},{6}\n"
                      .format(agent, M, S, run, solved, solve_time, mean_reward))


def run_till_solved(agent, env):
    """ Run agent until it produces a policy that solves environment, or timesout """

    solved = False
    episodes = 0
    start_time = time.time()
    mean_solution_reward = 0
    while not solved and episodes < MAX_EPISODES:
        agent.train(env, EPISODE_INTERVALS, MAX_STEPS, timeout=TIMEOUT, verbose=VERBOSE)
        episodes += EPISODE_INTERVALS
        solved, mean_solution_reward = environment_solved(agent, env)
        if time.time() - start_time > TIMEOUT:
            break
    run_time = time.time() - start_time
    return solved, run_time, mean_solution_reward


def environment_solved(agent, env):
    """ Evaluate trained agent against environment to see if it has found a solution """
    min_actions = env.get_minimum_actions()
    optimal_threshold = int(round(min_actions * OPTIMALITY_PROPORTION))
    solved_runs = 0
    total_reward = 0
    for r in range(EVAL_RUNS):
        episode = agent.generate_episode(env, optimal_threshold)
        if len(episode) < optimal_threshold:
            solved_runs += 1
            total_reward += get_episode_reward(episode)

    if solved_runs == 0:
        mean_reward = 0
    else:
        mean_reward = total_reward / solved_runs

    return solved_runs / EVAL_RUNS >= OPTIMAL_SOLVE_PROPORTION, mean_reward


def get_episode_reward(episode):
    return episode[-1][2]


def main():

    if len(sys.argv) != 3:
        print("Usage: python analysis_scaling.py outfile [1/0]")
        print("Where the [1/0] is for whether to append to file or not")
        return 1

    print("Welcome to your friendly scaling experiment runner")
    if sys.argv[2] != "0":
        print("Appending to", sys.argv[1])
        result_file = open(sys.argv[1], 'a+')
    else:
        print("Writing to new file", sys.argv[1])
        result_file = open(sys.argv[1], 'w')
        # write header line
        result_file.write("agent,M,S,run,solved,solve_time,mean_reward\n")

    for agent_type in agent_list:
        run_scaling_analysis(agent_type, result_file)

    result_file.close()


if __name__ == "__main__":
    main()
