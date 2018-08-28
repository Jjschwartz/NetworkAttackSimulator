"""
This module contains functions for performing an analysis of the performance of different
algorithms versus the problem size (number of machines and exploits)
"""
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.agents.sarsa import SarsaAgent
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


RESULTDIR = "results/"


def run_scaling_analysis(agent, machine_range, exploit_range, num_episodes,
                         max_steps, num_runs, window):

    print("Running scaling analysis for agent: \n\t {0}".format(str(agent)))
    results = []
    for m in machine_range:
        for e in exploit_range:
            print("\n>> Machines={0}, Exploits={1}".format(m, e))
            run_rewards = []
            run_times = []
            env = CyberAttackSimulatorEnv.from_params(m, e)
            for run in range(num_runs):
                print("Run {0} of {1}".format(run, num_runs))
                agent.reset()
                start_time = time.time()
                agent.train(env, num_episodes, max_steps, window)
                run_time = time.time() - start_time
                run_reward = agent.evaluate_agent(env, max_steps)
                run_rewards.append(run_reward)
                run_times.append(run_time)
            avg_run_rewards = np.average(run_rewards)
            avg_run_times = np.average(run_times)
            results.append((m, e, avg_run_rewards, avg_run_times))
            print("\n-- Average Rewards={0}".format(avg_run_rewards))
            print("-- Average Time={0}\n".format(avg_run_times))
    return results


def output_results(results, file_name):
    fout = open(file_name, "w")
    fout.write("m,e,reward,time\n")
    for i in range(len(results)):
        fout.write("{0}{1}{2}{3}\n".format(results[0], results[1], results[2], results[3]))
    fout.close()


def plot_reward(m, e, reward):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(m, e, reward, cmap=cm.coolwarm)
    ax.set_xlabel('Machines')
    ax.set_ylabel('Exploits')
    ax.set_zlabel('Average Reward')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))


def plot_time(m, e, time):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(m, e, time, cmap=cm.coolwarm)
    ax.set_xlabel('Machines')
    ax.set_ylabel('Exploits')
    ax.set_zlabel('Average Time')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))


def main():
    min_machines = 3
    max_machines = 10
    machine_range = list(range(min_machines, max_machines + 1))
    min_exploits = 1
    max_exploits = 5
    exploit_range = list(range(min_exploits, max_exploits + 1))
    max_episodes = 1000
    max_steps = 1000
    num_runs = 5
    window = 50

    alpha = 0.1
    gamma = 0.9
    c = 1.0
    ucb_sarsa = SarsaAgent("UCB", alpha, gamma, c)

    results = run_scaling_analysis(ucb_sarsa, machine_range, exploit_range, max_episodes,
                                   max_steps, num_runs, window)

    file_name = "{0}{1}_{2}_{3}".format(RESULTDIR,
                                        (min_machines, max_machines),
                                        (min_exploits, max_exploits),
                                        num_runs)

    output_results(results, file_name)

    m = []
    e = []
    r = []
    t = []
    for res in results:
        m.append(res[0])
        e.append(res[1])
        r.append(res[2])
        t.append(res[3])

    plot_reward(m, e, r)
    plot_time(m, e, t)
    plt.show()


if __name__ == "__main__":
    main()
