from cyber_attack_simulator.agents.td_agent import TDAgent


class SarsaAgent(TDAgent):
    """
    An Agent for the Cyber Attack Simulator environment that uses the Sarsa
    method to find an optimal policy
    """

    def __init__(self, type="UCB", alpha=0.1, gamma=0.9, max_epsilon=1.0,
                 min_epsilon=0.02, c=1.0):
        """
        Initialize a new Q-learning agent
        """
        super(SarsaAgent, self).__init__(type, alpha, gamma, max_epsilon,
                                         min_epsilon, c)
        self.algorithm = "Sarsa"

    def _run_episode(self, env, max_steps, param):
        """
        Train the agent for a single episode using Sarsa algorithm.
        """
        a_space = env.action_space
        s = env.reset()
        a = self._choose_action(s, a_space, param)
        ep_reward = 0
        ep_timesteps = 0
        for _ in range(max_steps):
            # increment state-action pair visit count
            self._n(s, a_space)[a] += 1
            new_s, reward, done = env.step(a_space[a])
            new_a = self._choose_action(new_s, a_space, param)
            self._q_update(s, a, new_s, new_a, reward, a_space)
            s = new_s
            a = new_a
            ep_reward += reward
            ep_timesteps += 1
            if done:
                break
        return ep_timesteps, ep_reward

    def _q_update(self, s, a, s_new, a_new, r, action_space):
        """
        Update state-action action value using Sarsa update

        Arguments:
            State s: current state
            int a: current action
            State s_new: next state
            int a_new: next action
            float r: reward from performing a in s (current step)
            list action_space : list of possible actions for state
        """
        current_q = self._q(s, action_space, a)
        new_q = self._q(s_new, action_space, a_new)
        td_error = r + self.gamma * new_q - current_q
        self._q(s, action_space)[a] = current_q + self.alpha * td_error

    def __str__(self):
        return ("{0}: alpha={1}, gamma={2}, min_epsilon={3}, max_epsilon={4}".format(
                self.name(), self.alpha, self.gamma, self.min_epsilon, self.max_epsilon))
