"""An example Tabular, epsilon greedy Q-Learning Agent using experience replay.

The replay can help improve learning stability and speed (in terms of learning
per training step), at the cost of increased memory and computation use.

It uses pytorch 1.5+ tensorboard library for logging (HINT: these dependencies
can be installed by running pip install nasim[dqn])

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python ql_replay_agent.py tiny

To see detailed results using tensorboard:

$ tensorboard --logdir runs/

To see available hyperparameters:

$ python ql_replay_agent.py --help

Notes
-----

This is by no means a state of the art implementation of Tabular Q-Learning.
It is designed to be an example implementation that can be used as a reference
for building your own agents and for simple experimental comparisons.
"""
import random
from pprint import pprint

import numpy as np

import nasim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    from gym import error
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install tabular_q_learning_agent dependencies "
        "by running 'pip install nasim[dqn]'.)"
    )


class ReplayMemory:
    """Experience Replay for Tabular Q-Learning agent """

    def __init__(self, capacity, s_dims):
        self.capacity = capacity
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int32)
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
        return batch


class TabularQFunction:
    """Tabular Q-Function """

    def __init__(self, num_actions):
        self.q_func = dict()
        self.num_actions = num_actions

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = str(x.astype(np.int))
        if x not in self.q_func:
            self.q_func[x] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_func[x]

    def forward_batch(self, x_batch):
        return np.asarray([self.forward(x) for x in x_batch])

    def update(self, s_batch, a_batch, delta_batch):
        for s, a, delta in zip(s_batch, a_batch, delta_batch):
            q_vals = self.forward(s)
            q_vals[a] += delta

    def get_action(self, x):
        return int(self.forward(x).argmax())

    def display(self):
        pprint(self.q_func)


class TabularQLearningAgent:
    """A Tabular. epsilon greedy Q-Learning Agent using Experience Replay """

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
                 verbose=True,
                 **kwargs):

        # This implementation only works for flat actions
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print("\nRunning Tabular Q-Learning with config:")
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
        self.epsilon_schedule = np.linspace(
            1.0, self.final_epsilon, self.exploration_steps
        )
        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0

        # Q-Function
        self.qfunc = TabularQFunction(self.num_actions)

        # replay setup
        self.replay = ReplayMemory(replay_size, self.obs_dim)

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            return self.qfunc.get_action(o)
        return random.randint(0, self.num_actions-1)

    def optimize(self):
        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.qfunc.forward_batch(s_batch)
        q_vals = np.take_along_axis(q_vals_raw, a_batch, axis=1).squeeze()

        # get target q val = max val of next state
        target_q_val_raw = self.qfunc.forward_batch(next_s_batch)
        target_q_val = target_q_val_raw.max(axis=1)
        target = r_batch + self.discount*(1-d_batch)*target_q_val

        # calculate error and update
        td_error = target - q_vals
        td_delta = self.lr * td_error

        # optimize the model
        self.qfunc.update(s_batch, a_batch, td_delta)

        q_vals_max = q_vals_raw.max(axis=1)
        mean_v = q_vals_max.mean().item()
        mean_td_error = np.absolute(td_error).mean().item()
        return mean_td_error, mean_v

    def train(self):
        if self.verbose:
            print("\nStarting training")

        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)
            ep_return, ep_steps, goal = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar(
                "epsilon", self.get_epsilon(), self.steps_done
            )
            self.logger.add_scalar(
                "episode_return", ep_return, self.steps_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.steps_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.steps_done
            )

            if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")

    def run_train_episode(self, step_limit):
        o = self.env.reset()
        done = False

        steps = 0
        episode_return = 0

        while not done and steps < step_limit:
            a = self.get_egreedy_action(o, self.get_epsilon())

            next_o, r, done, _ = self.env.step(a)
            self.replay.store(o, a, next_o, r, done)
            self.steps_done += 1
            mean_td_error, mean_v = self.optimize()
            self.logger.add_scalar(
                "mean_td_error", mean_td_error, self.steps_done
            )
            self.logger.add_scalar("mean_v", mean_v, self.steps_done)

            o = next_o
            episode_return += r
            steps += 1

        return episode_return, steps, self.env.goal_reached()

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
                         render_mode="readable"):
        if env is None:
            env = self.env
        o = env.reset()
        done = False

        steps = 0
        episode_return = 0

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render(render_mode)
            input("Initial state. Press enter to continue..")

        while not done:
            a = self.get_egreedy_action(o, eval_epsilon)
            next_o, r, done, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render(render_mode)
                print(f"Reward = {r}")
                print(f"Done = {done}")
                input("Press enter to continue..")

                if done:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        return episode_return, steps, env.goal_reached()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--render_eval", action="store_true",
                        help="Renders final policy")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=10000,
                        help="training steps (default=10000)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="(default=32)")
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
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=True,
                               flat_actions=True,
                               flat_obs=True)
    ql_agent = TabularQLearningAgent(
        env, verbose=args.quite, **vars(args)
    )
    ql_agent.train()
    ql_agent.run_eval_episode(render=args.render_eval)
