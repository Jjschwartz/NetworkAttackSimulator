"""A script for training a DQN agent and storing best policy """

import nasim
from nasim.agents.dqn_agent import DQNAgent


class BestDQN(DQNAgent):
    """A DQN Agent which saves best policy found during training """

    def __init__(self,
                 env,
                 save_path,
                 eval_epsilon=0.01,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.save_path = save_path
        self.eval_epsilon = eval_epsilon
        self.best_score = -float("inf")

    def run_train_episode(self, step_limit):
        ep_ret, steps, goal_reached = super().run_train_episode(step_limit)

        if self.steps_done > self.exploration_steps:
            eval_ret, _, _ = self.run_eval_episode(
                eval_epsilon=self.eval_epsilon
            )
            if eval_ret > self.best_score:
                print(f"Saving New Best Score = {ep_ret}")
                self.best_score = eval_ret
                self.save(self.save_path)

        return ep_ret, steps, goal_reached


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("save_path", type=str, help="save path for agent")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("--eval_epsilon", type=float, default=0.01,
                        help="Epsilon to use for evaluation (default=0.01)")
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
    parser.add_argument("--exploration_steps", type=int, default=5000,
                        help="(default=5000)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="(default=0.99)")
    args = parser.parse_args()
    assert args.training_steps > args.exploration_steps

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=True,
                               flat_obs=True)
    dqn_agent = BestDQN(env, **vars(args))
    dqn_agent.train()

    print(f"\n{'-'*60}\nDone\n{'-'*60}")
    print(f"Best Policy score = {dqn_agent.best_score}")
    print(f"Policy saved to: {dqn_agent.save_path}")
