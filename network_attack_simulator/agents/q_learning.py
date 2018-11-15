import numpy as np
from network_attack_simulator.agents.td_agent import TDAgent


class QLearningAgent(TDAgent):
    """
    An Agent for the Cyber Attack Simulator environment that uses the
    Q-learning method to find an optimal policy.

    Implements the TD abstract class
    """

    def __init__(self, type="UCB", alpha=0.1, gamma=0.9, max_epsilon=1.0,
                 min_epsilon=0.02, epsilon_decay_lambda=0.001, c=1.0):
        """
        Initialize a new Q-learning agent
        """
        super(QLearningAgent, self).__init__(type, alpha, gamma, max_epsilon, min_epsilon,
                                             epsilon_decay_lambda, c)
        self.algorithm = "Q-learning"

    def _run_episode(self, env, max_steps, param):
        """
        Train the agent for a single episode using Q-learning algorithm
        """
        a_space = env.action_space
        s = self._process_state(env.reset())
        # s = env.reset().copy()
        ep_reward = 0
        ep_timesteps = 0
        for _ in range(max_steps):
            a = self._choose_action(s, a_space, param)
            # increment state-action pair visit count
            self._n(s, a_space)[a] += 1
            new_s, reward, done = env.step(a_space[a])
            new_s = self._process_state(new_s)
            self._q_update(s, a, new_s, reward, a_space)
            s = new_s
            # s = new_s.copy()
            ep_reward += reward
            ep_timesteps += 1
            if done:
                break
        return ep_timesteps, ep_reward

    def _process_state(self, s):
        """
        Convert a state into the form stored in q_table
        """
        return s.get_hashable()

    def _q_update(self, s, a, s_new, r, action_space):
        """
        Update state-action action value using Q-learning update

        Arguments:
            State s: current state
            int a: current action
            State s_new: next state
            float r: reward from performing a in s (current step)
            list action_space : list of possible actions for state
        """
        current_q = self._q(s, action_space, a)
        new_q = np.max(self._q(s_new, action_space))
        td_error = r + self.gamma * new_q - current_q
        self._q(s, action_space)[a] = current_q + self.alpha * td_error

    def __str__(self):
        return ("{0}: alpha={1}, gamma={2}, c={3}".format(
                self.name(), self.alpha, self.gamma, self.c))
