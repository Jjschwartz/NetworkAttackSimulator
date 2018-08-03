import math


class Agent(object):
    """
    Abstract Agent class.

    Methods users need to implemented:
        - train
        - generate_episode
        - _choose_greedy_action

    Implemented Methods
        - generate_episode
        - report_progress
    """

    def train(self):
        """
        The main training method
        """
        raise NotImplementedError

    def generate_episode(self):
        """
        Generate and episode following the current greed-policy.
        This method is used to check the current learned policy.

        Returns:
            list episode : ordered list of (State, Action, reward) tuples
                from each timestep taken during episode
        """
        episode = []
        state = self.env.reset()
        for t in range(self.max_steps):
            action = self._choose_greedy_action(state)
            result = self.env.step(self.action_space[action])
            new_state, reward, done, _ = result
            episode.append((state, self.action_space[action], reward))
            if done:
                break
            state = new_state
        return episode

    def _choose_greedy_action(self, state):
        """
        Choose the best action for given state according to current policy

        Arguments:
            State state : the state to choose action for

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
