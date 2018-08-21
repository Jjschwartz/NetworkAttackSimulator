"""
This module contains functionallity for analysing and comparing agent
performance on Cyber Attack Simulator environments
"""
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
import numpy as np
import matplotlib.pyplot as plt


class Analyser(object):

    def __init__(self, env, agents, num_episodes, max_steps, num_runs):
        """
        Initialize for given network configuraiton and set of agents

        Arguments:
            CyberAttackSimulatorEnv env : environment object to test on
            list agents : list of Agent class instances
            int num_episodes : number of episodes to train agent for
            int max_steps : max number of steps per episode
            int num_runs : number of runs (i.e. set of episodes) to average
                results over
        """
        self.env = env
        self.agents = agents
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_runs = num_runs

        self.results = {}

    def run_analysis(self, verbose=True):
        """
        Run analysis for agents and environment

        Arguments:
            bool verbose : whether to print progress messages or not

        Returns:
            dict results : dict with agent type as key and values being
                npmatrix of rewards vs timesteps averaged over runs
        """
        self.results = {}
        for agent in self.agents:
            print("Running analysis for agent {0}".format(str(agent)))
            run_rewards = []
            for run in range(self.num_runs):
                print("Run {0} of {1}".format(run, self.num_runs))
                agent.reset()
                t, r = agent.train(self.env, self.num_episodes, self.max_steps,
                                   verbose)
                run_rewards.append(r)
            self.results[str(agent)] = np.average(run_rewards, axis=0)

    def plot_results(self):
        """
        Plot results of last analysis run
        """
        for agent, result in self.results.items():
            plt.plot(range(self.num_episodes), result, label=agent)

        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.show()
