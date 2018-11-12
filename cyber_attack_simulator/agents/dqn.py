"""
A Deep Q-learning agent implementation for the CyberAttackSimulator
"""
from cyber_attack_simulator.agents.agent import Agent
import random
import numpy as np
import time
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf


def huber_loss(y_true, y_pred, loss_delta=1.0):
    err = y_true - y_pred

    cond = K.abs(err) < loss_delta
    L2 = 0.5 * K.square(err)
    L1 = loss_delta * (K.abs(err) - 0.5 * loss_delta)
    loss = tf.where(cond, L2, L1)
    output = K.mean(loss)
    return output


class Brain:
    """ Fully connected single-layer neural network"""

    def __init__(self, state_size, num_actions, hidden_units, learning_rate):
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        self.model = self._create_model()
        self.target_model = self._create_model()

    def _create_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_units, activation='relu', input_dim=self.state_size))
        model.add(Dense(self.num_actions, activation='linear'))
        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss=huber_loss, optimizer=opt)
        return model

    def train(self, x, y, batch_size=64, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.target_model.predict(s)
        return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.state_size), target=target).flatten()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def reset(self):
        self.model = self._create_model()
        self.target_model = self._create_model()


class Memory:
    """Replay buffer"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = []
        self.reset()

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def reset(self):
        self.samples = []


class DQNAgent(Agent):

    def __init__(self, state_size, num_actions, hidden_units=256,
                 gamma=0.99,
                 min_epsilon=0.02,
                 max_epsilon=1.0,
                 epsilon_decay_lambda=0.0001,
                 learning_rate=0.00025,
                 memory_capacity=10000,
                 batch_size=64,
                 update_target_freq=1000):
        """
        Initialize a new Deep Q-network agent

        Parameters:
            int state_size : size of a state (i.e. input into network)
            int num_actions : size of action space (i.e. output of network)
            int hidden_units : number of hidden units of hidden Q-network layer
            float gamma : discount
            float min_epsilon : minimum exploration probability
            float max_epsilon : maximum exploration probability
            float epsilon_decay_lambda : lambda for exponential epsilon decay
            float learning_rate : RMSprop learning rate
            int memory_capacity : capacity of replay buffer
            int batch_size : Q-network training batch size
            int update_target_freq : target model update frequency in terms of number of steps
        """
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon_decay_lambda = epsilon_decay_lambda
        self.epsilon = max_epsilon
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq

        self.brain = Brain(state_size, num_actions, hidden_units, learning_rate)
        self.memory = Memory(memory_capacity)

        self.steps = 0

    def train(self, env, num_episodes=100, max_steps=100, timeout=None, verbose=False, **kwargs):

        if "visualize_policy" in kwargs:
            visualize_policy = kwargs["visualize_policy"]
        else:
            visualize_policy = 0

        self.print_message("Starting training for {} episodes".format(num_episodes), verbose)

        # stores timesteps and rewards for each episode
        episode_timesteps = []
        episode_rewards = []
        episode_times = []

        training_start_time = time.time()
        steps_since_update = 0

        reporting_window = min(num_episodes / 10, 10)

        for e in range(num_episodes):
            start_time = time.time()
            timesteps, reward = self._run_episode(env, max_steps)
            ep_time = time.time() - start_time
            episode_rewards.append(reward)
            episode_timesteps.append(timesteps)
            episode_times.append(ep_time)

            self.epsilon = self.epsilon_decay()
            steps_since_update += timesteps

            if steps_since_update > self.update_target_freq:
                self.brain.update_target_model()
                steps_since_update = 0

            self.report_progress(e, reporting_window, episode_timesteps, verbose)

            if e > 0 and visualize_policy != 0 and e % visualize_policy == 0:
                gen_episode = self.generate_episode(env, max_steps)
                self.print_message("Visualizing current policy. Episode length = {0}"
                                   .format(len(gen_episode)), verbose)
                env.render_episode(gen_episode)

            if timeout is not None and time.time() - training_start_time > timeout:
                self.print_message("Timed out after {} sec on episode {:.2f}".format(timeout, e),
                                   verbose)
                break

        total_training_time = time.time() - training_start_time
        self.print_message("Training complete after {} episodes and {:.2f} sec"
                           .format(e, total_training_time), verbose)

        return episode_timesteps, episode_rewards, episode_times

    def reset(self):
        self.brain.reset()
        self.memory.reset()
        self.epsilon = self.max_epsilon
        self.steps = 0

    def _run_episode(self, env, max_steps):
        """
        Train the agent for a single episode using Q-learning algorithm
        """
        a_space = env.action_space
        s = self._process_state(env.reset())
        ep_reward = 0
        ep_timesteps = 0

        for _ in range(max_steps):
            # interact with environment
            a = self.act(s)
            ns, r, done = env.step(a_space[a])
            ns = self._process_state(ns)

            if done:
                ns = None

            # train agent
            self.observe((s, a, r, ns))
            self.replay()

            s = ns
            ep_reward += r
            ep_timesteps += 1

            self.steps += 1

            if done:
                break

        return ep_timesteps, ep_reward

    def _process_state(self, s):
        """ Convert state into format that can be handled by NN"""
        return s.flatten()

    def act(self, s):
        """ Choose action using epsilon greedy action selection """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def _choose_greedy_action(self, state, action_space):
        return np.argmax(self.brain.predictOne(self._process_state(state)))

    def observe(self, sample):
        """ Add an observation to replay memory and also adjust epsilon """
        self.memory.add(sample)

    def epsilon_decay(self):
        """ Decay the epsilon value based on episode number """
        # exponential decay by steps
        diff = self.max_epsilon - self.min_epsilon
        temp = diff * math.exp(-self.epsilon_decay_lambda * self.steps)
        return self.min_epsilon + temp

    def replay(self):
        """ Perform parameter updates """
        # each sample is observation = (s, a, r, ns) tuple
        batch = self.memory.sample(self.batch_size)
        batch_len = len(batch)

        no_state = np.zeros(self.state_size)

        states = np.array([o[0] for o in batch])
        next_states = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        next_p = self.brain.predict(next_states, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.num_actions))

        for i in range(batch_len):
            o = batch[i]
            s, a, r, ns = o

            # set target for given s, a, r, ns observation
            t = p[i]
            if ns is None:
                # terminal state
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(next_p[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y, self.batch_size)

    def print_message(self, message, verbose):
        if verbose:
            print("DQN Agent: {}".format(message))

    def __str__(self):
        return "DQNAgent"
