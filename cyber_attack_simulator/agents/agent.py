import math


class Agent(object):
    """
    Abstract Agent class.

    Attributes:
        - env - environment to solve
        -

    Methods users need to implemented:
        - train
        - reset
        - _choose_greedy_action
        - __str__

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

    def reset(self):
        """
        Resets the agent to an initial not trained state.
        Used to perform multiple runs without having to initialize new agent.
        """
        raise NotImplementedError

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

    def generate_episode(self, env, max_steps=100):
        """
        Generate and episode following the current greed-policy.
        This method is used to check the current learned policy.

        Arguments:
            CyberAttackSimulatorEnv env : environment to generate episode for
            int max_steps : max number of steps allowed for episode

        Returns:
            list episode : ordered list of (State, Action, reward) tuples
                from each timestep taken during episode
        """
        episode = []
        state = env.reset()
        action_space = env.action_space
        reward_sum = 0
        steps = 0
        for t in range(max_steps):
            action = self._choose_greedy_action(state, action_space)
            new_state, reward, done = env.step(action_space[action])
            episode.append((state, action_space[action], reward, False))
            reward_sum += reward
            steps += 1
            if done:
                episode.append((new_state, None, reward_sum, done))
                break
            state = new_state
        return episode

    def evaluate_agent(self, env, max_steps=100):
        """
        Evaluate the current agents policy by running an episode using greedy
        policy.

        Arguments:
            CyberAttackSimulatorEnv env : environment to generate episode for
            int max_steps : max number of steps allowed for episode

        Returns:
            int reward : total reward recieved for episode
        """
        state = env.reset()
        action_space = env.action_space
        reward_sum = 0
        for t in range(max_steps):
            action = self._choose_greedy_action(state, action_space)
            new_state, reward, done = env.step(action_space[action])
            reward_sum += reward
            if done:
                break
            state = new_state
        return reward_sum

    def report_progress(self, episode_num, interval, episodes):
        """
        Print a progress message to standard out, reporting on current episode number and average
        timesteps per episode

        Arguments:
            int episode_num : current episode number
            int interval : reporting interval (how often to report)
            list[int] episodes : list of timesteps for each episode up to current episode
        """
        message = "Episode = {0} - avg timesteps for last {1} episodes = {2}"
        interval = int(math.ceil(interval))
        if episode_num % interval == 0:
            if episode_num > 0:
                episode_avg = sum(episodes[-interval:])
                episode_avg /= interval
                print(message.format(episode_num, interval, episode_avg))
            else:
                print(message.format(episode_num, interval, episodes[0]))

    def __str__(self):
        raise NotImplemented
