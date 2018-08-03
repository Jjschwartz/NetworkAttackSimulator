import numpy as np
from agent import Agent


class SarsaAgent(Agent):
    """
    An Agent for the Cyber Attack Simulator environment that uses the Sarsa
    method to find an optimal policy
    """

    def __init__(self, env, num_episodes=100, max_steps=100, alpha=0.1,
                 gamma=0.9):
        """
        Initialize a new TDAgent instance

        Arguments:
            CyberAttackSimulatorEnv env: the environment to solve
            int num_episodes: number of episodes to run agent for
            int max_steps: max number of steps per episode
            float alpha: the value update step-size for the agent
            float gamma: the discount rate
            float min_epsilon: minimum epsilon value
        """
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma

        self.action_space = env.action_space
        self.q_table = dict()

    def _q_update(self, s, a, s_new, a_new, r):
        """
        Update state-action action value

        Arguments:
            State s: current state
            int a: current action
            State s_new: next state
            int a_new: next action
            float r: reward from performing a in s (current step)
        """
        current_q = self._q(s, a)
        new_q = self._q(s_new, a_new)
        td_error = r + self.gamma * new_q - current_q
        self._q(s)[a] = current_q + self.alpha * td_error

    def _q(self, state, action=None):
        """
        Returns the Q-value for a state-action pair or returns the list of
        Q-values for a state if no action is provided

        Arguments:
            State state : environment state
            int action : index of action performed

        Returns:
            list q_values : list of action values for state
            or
            float q_value : state-action pair value
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.action_space))

        if action is None:
            return self.q_table[state]
        return self.q_table[state][action]


class EGreedySarsaAgent(SarsaAgent):
    """
    A Sarsa based agent for the Cyber Attack Simulator environment that uses
    epsilon-greedy action selection
    """

    def __init__(self, env, num_episodes=100, max_steps=100, alpha=0.1,
                 gamma=0.9, min_epsilon=0.02):
        """
        Initialize a new TDAgent instance

        Arguments:
            CyberAttackSimulatorEnv env: the environment to solve
            int num_episodes: number of episodes to run agent for
            int max_steps: max number of steps per episode
            float alpha: the value update step-size for the agent
            float gamma: the discount rate
            float min_epsilon: minimum epsilon value
        """
        self.min_epsilon = min_epsilon
        super().__init__(env, num_episodes, max_steps, alpha, gamma)

    def train(self):
        print("Starting training for {0} episodes".format(self.num_episodes))

        epsilon = 1.0
        epsilon_schedule = np.linspace(epsilon, self.min_epsilon,
                                       self.num_episodes)
        # stores total timestep at end of each episode
        total_timesteps = 0
        timeteps_per_episode = []

        for e in range(self.num_episodes):
            # reset environment
            state = self.env.reset()
            # get initial action
            action = self._choose_action(state, epsilon)

            for _ in range(self.max_steps):
                # perform action
                result = self.env.step(self.action_space[action])
                new_state, reward, done, _ = result
                # get action for new state
                new_action = self._choose_action(new_state, epsilon)
                # update state-action q_value
                self._q_update(state, action, new_state, new_action, reward)
                state = new_state
                action = new_action
                total_timesteps += 1
                if done:
                    break
            # slowly decrease exploration
            epsilon = epsilon_schedule[e]

            timeteps_per_episode.append(total_timesteps)
            self.report_progress(e, self.num_episodes / 10, total_timesteps)

        print("Training complete after {0} episodes".format(e))
        return timeteps_per_episode

    def _choose_action(self, state, epsilon=0.0):
        """
        Choose action to take from given state using epsilon-greedy policy
        If epsilon not provided then defaults to greedy policy

        Arguments:
            State state : the state to choose action for
            float epsilon : random action choice hyperparameter

        Returns:
            int action : index of chosen action to take
        """
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, len(self.action_space))
        else:
            return np.argmax(self._q(state))

    def _choose_greedy_action(self, state):
        return self._choose_action(state)


class UCBSarsaAgent(SarsaAgent):
    """
    A Sarsa based agent for the Cyber Attack Simulator environment that uses
    Upper-confidence-bound action selection (UCB1)
    """

    def __init__(self, env, num_episodes=100, max_steps=100, alpha=0.1,
                 gamma=0.9, c=1.0):
        """
        Initialize a new TDAgent instance

        Arguments:
            CyberAttackSimulatorEnv env: the environment to solve
            int num_episodes: number of episodes to run agent for
            int max_steps: max number of steps per episode
            float alpha: the value update step-size for the agent
            float gamma: the discount rate
            float c: the exploration hyperparameter
        """
        self.c = c
        # dictionary for storing state-action visits
        self.n_table = dict()
        super().__init__(env, num_episodes, max_steps, alpha, gamma)

    def train(self):
        print("Starting training for {0} episodes".format(self.num_episodes))

        # stores total timestep at end of each episode
        total_timesteps = 0
        episode_timesteps = []
        episode_rewards = []

        for e in range(self.num_episodes):
            # reset environment
            state = self.env.reset()
            # get initial action
            action = self._choose_action(state, self.c)

            episode_reward = 0
            for _ in range(self.max_steps):
                # increment state-action pair visit count
                self._n(state)[action] += 1
                # perform action
                result = self.env.step(self.action_space[action])
                new_state, reward, done, _ = result
                # get action for new state
                new_action = self._choose_action(new_state, self.c)
                # update state-action q_value
                self._q_update(state, action, new_state, new_action, reward)
                state = new_state
                action = new_action
                episode_reward += reward
                total_timesteps += 1
                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_timesteps.append(total_timesteps)
            self.report_progress(e, self.num_episodes / 10, episode_timesteps)

        print("Training complete after {0} episodes".format(e))
        return episode_timesteps, episode_rewards

    def _n(self, state, action=None):
        """
        Helper method for dealing with state-action pair visit counts

        Arguments:
            State state : environment state

        Returns:
            list n : list of state-action visits to if action=None
            or
            int n : total visits to state-action pair
        """
        if state not in self.n_table:
            self.n_table[state] = np.zeros(len(self.action_space))
        if action is None:
            return self.n_table[state]
        return self.n_table[state][action]

    def _choose_action(self, state, c=0.0):
        """
        Choose action to take from given state using Upper-Confidence-Bound
        action selection. If c hyperparameter not provided then defaults to
        greedy policy.

        Arguments:
            State state : the state to choose action for
            float c : the exploration hyperparameter

        Returns:
            int action : index of chosen action to take
        """
        n_state_action = self._n(state)
        if np.any(n_state_action == 0):
            # if any state-action pair has not been visited, choose action
            return np.argwhere(n_state_action == 0).flatten()[0]
        log_state_visits = np.log(np.sum(n_state_action))
        bonuses = c * np.sqrt(2 * (log_state_visits / n_state_action))

        q_values = self._q(state)
        adj_q_values = q_values + bonuses
        return np.argmax(adj_q_values)

    def _choose_greedy_action(self, state):
        return self._choose_action(state)
