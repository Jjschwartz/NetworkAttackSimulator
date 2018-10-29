import numpy as np
import time
import math
from cyber_attack_simulator.agents.agent import Agent


class TDAgent(Agent):
    """
    A Temporal Difference Agent abstract class for the Cyber Attack Simulator environment that uses
    a TD method to find an optimal policy

    Supported action-selection methods:
        - Upper-confidence bounds - UCB
        - epsilon-greedy - egreedy
    """
    types = ["UCB", "egreedy"]
    algorithm = "TD"

    def __init__(self, type="UCB", alpha=0.1, gamma=0.9, max_epsilon=1.0, min_epsilon=0.01,
                 epsilon_decay_lambda=0.001, c=1.0):
        """
        Initialize a new TD agent

        Arguments:
            str type : type of Sarsa agent
            float alpha: the value update step-size for the agent
            float gamma: the discount rate
            float max_epsilon : initial max epsilon value (for e-greedy agent)
            float min_epsilon: minimum epsilon value (for e-greedy agent)
            float epsilon_decay_lamda : lambda for exponential epsilon decay
            float c: the exploration hyperparameter (used for UCB agent)
        """
        if type not in self.types:
            raise ValueError("type must be in: {0}".format(self.types))
        self.type = type
        self.alpha = alpha
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_lambda = epsilon_decay_lambda
        self.c = c

        self.reset()

    def reset(self):
        # dictionary for storing state-action visits
        self.n_table = dict()
        self.q_table = dict()

        if self.type == "UCB":
            self.param = self.c
        else:
            self.param = self.max_epsilon

    def train(self, env, num_episodes=100, max_steps=100, timeout=None,
              verbose=False, **kwargs):

        if "visualize_policy" in kwargs:
            visualize_policy = kwargs["visualize_policy"]
        else:
            visualize_policy = 0

        self._print_message("Starting training for {} episodes".format(num_episodes), verbose)

        # stores timesteps, rewards and time taken for each episode
        episode_timesteps = []
        episode_rewards = []
        episode_times = []
        steps = 0

        training_start_time = time.time()

        for e in range(num_episodes):
            start_time = time.time()
            timesteps, reward = self._run_episode(env, max_steps, self.param)
            ep_time = time.time() - start_time
            episode_rewards.append(reward)
            episode_timesteps.append(timesteps)
            episode_times.append(ep_time)
            steps += timesteps

            # slowly decrease exploration for egreedy agent
            if self.type == "egreedy":
                self.param = self._epsilon_decay(steps)

            # reports progress every 1/10th of total episdes run
            self.report_progress(e, num_episodes / 10, episode_timesteps, verbose)

            if e > 0 and visualize_policy != 0 and e % visualize_policy == 0:
                gen_episode = self.generate_episode(env, max_steps)
                self._print_message("Visualizing current policy. Episode length = {0}"
                                    .format(len(gen_episode)), verbose)
                env.render_episode(gen_episode)

            # check for timeout
            if timeout is not None and time.time() - training_start_time > timeout:
                self._print_message("Timed out after {} sec on episode {}".format(timeout, e),
                                    verbose)
                break

        total_training_time = time.time() - training_start_time
        self._print_message("Training complete after {} episodes and {:.2f} sec"
                            .format(e, total_training_time), verbose)

        return episode_timesteps, episode_rewards, episode_times

    def _print_message(self, message, verbose):
        if verbose:
            print("{} {} agent: {}".format(self.algorithm, self.type, message))

    def _run_episode(self, env, max_steps, param):
        """
        Train the agent for a single episode using TD-algorithm.

        Arguments:
            CyberAttackSimulatorEnv env : the environment
            int max_steps : max number of steps to run episode for
            float param : the action selection hyperparameter value for agent

        Returns:
            int ep_timesteps : number of timesteps taken for epidode
            int ep_reward : reward achieved for episode
        """
        raise NotImplementedError

    def _choose_action(self, state, action_space, param=0.0):
        """
        Choose action to take from given state using agent defined action selection. If
        hyperparameter not provided then defaults to greedy policy.

        Arguments:
            State state : the state to choose action for
            list action_space : list of possible actions for state
            float param : the action selection hyperparameter

        Returns:
            int action : index of chosen action to take
        """
        if self.type == "UCB":
            return self._choose_action_ucb(state, action_space, param)
        else:
            return self._choose_action_egreedy(state, action_space, param)

    def _choose_action_egreedy(self, state, action_space, epsilon=0.0):
        """
        Choose action to take from given state using epsilon-greedy policy.
        If epsilon not provided then defaults to greedy policy

        Arguments:
            State state : the state to choose action for
            list action_space : list of possible actions for state
            float epsilon : random action choice hyperparameter

        Returns:
            int action : index of chosen action to take
        """
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, len(action_space))
        else:
            return np.argmax(self._q(state, action_space))

    def _choose_action_ucb(self, state, action_space, c=0.0):
        """
        Choose action to take from given state using Upper-Confidence-Bound action selection.
        If c hyperparameter not provided then defaults to greedy policy.

        Arguments:
            State state : the state to choose action for
            list action_space : list of possible actions for state
            float c : the exploration hyperparameter

        Returns:
            int action : index of chosen action to take
        """
        n_state_action = self._n(state, action_space)
        if np.any(n_state_action == 0):
            # if any state-action pair has not been visited, choose action
            return np.argwhere(n_state_action == 0).flatten()[0]
        log_state_visits = np.log(np.sum(n_state_action))
        bonuses = c * np.sqrt(2 * (log_state_visits / n_state_action))

        q_values = self._q(state, action_space)
        adj_q_values = q_values + bonuses
        return np.argmax(adj_q_values)

    def _q(self, state, action_space, action=None):
        """
        Returns the Q-value for a state-action pair or returns the list of Q-values for a state if
        no action is provided

        Arguments:
            State state : environment state
            list action_space : list of possible actions for state
            int action : index of action performed

        Returns:
            list q_values : list of action values for state
            or
            float q_value : state-action pair value
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(action_space))
        if action is None:
            return self.q_table[state]
        return self.q_table[state][action]

    def _n(self, state, action_space, action=None):
        """
        Helper method for dealing with state-action pair visit counts

        Arguments:
            State state : environment state
            list action_space : list of possible actions for state
            int action = index of action to take or None

        Returns:
            list n : list of state-action visits to if action=None
            or
            int n : total visits to state-action pair
        """
        if state not in self.n_table:
            self.n_table[state] = np.zeros(len(action_space))
        if action is None:
            return self.n_table[state]
        return self.n_table[state][action]

    def _epsilon_decay(self, steps):
        """
        Decay epsilon value.
        """
        temp = (self.max_epsilon - self.min_epsilon) * math.exp(-self.epsilon_decay_lambda * steps)
        return self.min_epsilon + temp
        # step = (self.max_epsilon - self.min_epsilon) / num_episodes
        # return self.max_epsilon - (step * e)

    def _choose_greedy_action(self, state, action_space):
        return self._choose_action(state, action_space)

    def __str__(self):
        return ("{0}: alpha={1}, gamma={2}, c={3}".format(
                self.name(), self.alpha, self.gamma, self.c))

    def name(self):
        return "{0} {1}".format(self.algorithm, self.type)
