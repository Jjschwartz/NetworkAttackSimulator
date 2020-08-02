"""An example DQN Agent.

It uses pytorch 1.5+ and tensorboard libraries (HINT: these dependencies can
be installed by running pip install nasim[dqn])

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python dqn_agent tiny

To see detailed results using tensorboard:

$ tensorboard --logdir runs/

To see available hyperparameters:

$ python --help

Notes
-----

This is by no means a state of the art implementation of DQN, but is designed
to be an example implementation that can be used as a reference for building
your own agents.
"""
import random
import numpy as np
from gym import error
from pprint import pprint

import nasim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install dqn_agent dependencies by running "
        "'pip install nasim[dqn]'.)"
    )


class ReplayMemory:

    def __init__(self, capacity, s_dims, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [self.s_buf[sample_idxs],
                 self.a_buf[sample_idxs],
                 self.next_s_buf[sample_idxs],
                 self.r_buf[sample_idxs],
                 self.done_buf[sample_idxs]]
        return [torch.from_numpy(buf).to(self.device) for buf in batch]


class DQN(nn.Module):

    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))
        self.out = nn.Linear(layers[-1], num_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]


class DQNAgent:

    def __init__(self,
                 env,
                 seed=None,
                 lr=0.001,
                 training_steps=10000,
                 batch_size=32,
                 replay_size=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 hidden_sizes=[64, 64],
                 target_update_freq=1000,
                 **kwargs):

        # This DQN implementation only works for flat actions
        assert env.flat_actions
        print(f"\nRunning DQN with config:")
        pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # envirnment setup
        self.env = env

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        # logger setup
        self.logger = SummaryWriter()

        # Training related attributes
        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0,
                                            self.final_epsilon,
                                            self.exploration_steps)
        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0

        # Neural Network related attributes
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.dqn = DQN(self.obs_dim,
                       hidden_sizes,
                       self.num_actions).to(self.device)
        print("\nUsing Neural Network:")
        print(self.dqn)

        self.target_dqn = DQN(self.obs_dim,
                              hidden_sizes,
                              self.num_actions).to(self.device)
        self.target_update_freq = target_update_freq

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # replay setup
        self.replay = ReplayMemory(replay_size,
                                   self.obs_dim,
                                   self.device)

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            o = torch.from_numpy(o).float().to(self.device)
            return self.dqn.get_action(o).cpu().item()
        return random.randint(0, self.num_actions-1)

    def optimize(self):
        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        # get target q val = max val of next state
        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val = target_q_val_raw.max(1)[0]
            target = r_batch + self.discount*(1-d_batch)*target_q_val

        # calculate loss
        loss = self.loss_fn(q_vals, target)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        return loss.item(), mean_v

    def train(self):
        print("\nStarting training")
        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_episode(training_steps_remaining)
            ep_return, ep_steps, goal = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar("epsilon",
                                   self.get_epsilon(),
                                   self.steps_done)
            self.logger.add_scalar("episode_return",
                                   ep_return,
                                   self.steps_done)
            self.logger.add_scalar("episode_steps",
                                   ep_steps,
                                   self.steps_done)
            self.logger.add_scalar("episode_goal_reached",
                                   int(goal),
                                   self.steps_done)

            if num_episodes % 10 == 0:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")

        self.logger.close()
        print("Training complete")
        print(f"\nEpisode {num_episodes}:")
        print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
        print(f"\treturn = {ep_return}")
        print(f"\tgoal = {goal}")

    def run_episode(self, step_limit):
        o = self.env.reset()
        done = False

        steps = 0
        episode_return = 0

        while not done and steps < step_limit:
            a = self.get_egreedy_action(o, self.get_epsilon())
            next_o, r, done, _ = self.env.step(a)
            self.replay.store(o, a, next_o, r, done)
            self.steps_done += 1

            loss, mean_v = self.optimize()
            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("mean_v", mean_v, self.steps_done)

            o = next_o
            episode_return += r
            steps += 1

        return episode_return, steps, self.env.goal_reached()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[64, 64],
                        help="(default=[64. 64])")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("--training_steps", type=int, default=10000,
                        help="training steps (default=10000)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="(default=32)")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="(default=1000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="(default=100000)")
    parser.add_argument("--final_epsilon", type=float, default=0.05,
                        help="(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0,
                        help="(default=1.0)")
    parser.add_argument("--exploration_steps", type=int, default=10000,
                        help="(default=10000)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="(default=0.99)")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=True,
                               flat_obs=True)
    dqn_agent = DQNAgent(env, **vars(args))
    dqn_agent.train()
