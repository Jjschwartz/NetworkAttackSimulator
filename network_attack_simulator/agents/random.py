import numpy as np
from network_attack_simulator.agents.agent import Agent


class RandomAgent(Agent):
    """
    A random agent for the Cyber Attack Simulator environment that selects actions
    uniformly at random at each step
    """

    def _choose_greedy_action(self, state, action_space, epsilon=0.05):
        return np.random.randint(0, len(action_space))

    def _process_state(self, s):
        return s

    def __str__(self):
        return "Random Agent"
