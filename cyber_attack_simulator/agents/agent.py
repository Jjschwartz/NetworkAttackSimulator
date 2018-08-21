import math


class Agent(object):
    """
    Abstract Agent class.

    Attributes:
        - env - environment to solve
        -

    Methods users need to implemented:
        - train
        - generate_episode
        - _choose_greedy_action

    Implemented Methods
        - generate_episode
        - report_progress
    """

    def train(self, env, num_episodes=100, max_steps=100):
        """
        The main training method

        Arguments:
            CyberAttackSimulatorEnv env : the environment to solve
            int num_episodes : number of episodes to run agent for
            int max_steps : max number of steps per episode

        Returns:
            list(int) timeteps_per_episode : cumulative timesteps per episode
        """
        raise NotImplementedError

    def generate_episode(self, env, max_steps=100):
        """
        Generate and episode following the current greed-policy.
        This method is used to check the current learned policy.

        Arguments:


        Returns:
            list episode : ordered list of (State, Action, reward) tuples
                from each timestep taken during episode
        """
        episode = []
        state = env.reset()
        action_space = env.action_space
        reward_sum = 0
        for t in range(max_steps):
            action = self._choose_greedy_action(state, action_space)
            result = env.step(action_space[action])
            new_state, reward, done, _ = result
            episode.append((state, action_space[action], reward, False))
            reward_sum += reward
            if done:
                episode.append((new_state, None, reward_sum, done))
                break
            state = new_state
        return episode

    def _choose_greedy_action(self, state, action_space):
        """
        Choose the best action for given state according to current policy

        Arguments:
            State state : the state to choose action for
            list action_space : list of possible actions for state

        Returns:
            int action : index of chosen action to take
        """
        raise NotImplementedError

    def report_progress(self, episode_num, interval, episodes):
        """
        Print a progress message to standard out, reporting on current
        episode number and average timesteps per episode

        Arguments:
            int episode_num : current episode number
            int interval : reporting interval (how often to report)
            list[int] episodes : list of cumulative total of timesteps for each
                episode upto current episode
        """
        message = "Episode = {0} - avg timesteps for last {1} episodes = {2}"
        interval = int(math.ceil(interval))
        if episode_num % interval == 0:
            if episode_num > 0:
                episode_avg = episodes[-1] - episodes[-1 - interval]
                episode_avg /= interval
                print(message.format(episode_num, interval, episode_avg))
            else:
                print(message.format(episode_num, interval, episodes[0]))
