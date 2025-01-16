"""A script for running a pre-trained DQN agent

Note, user must ensure the DQN policy matches the NASim
Environment used to train it in terms of size.

E.g. A policy trained on the 'tiny-gen' env can be tested
against the 'tiny' env since they both have the same Action
and Observation spaces.

But a policy trained on 'tiny-gen' could not be used on the
'small' environment (or any non-'tiny' environment for that
matter)
"""

import nasim
from nasim.agents.dqn_agent import DQNAgent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("policy_path", type=str, help="path to policy")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("--eval_eps", type=int, default=1,
                        help="Number of episodes to run (default=1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default=0)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help=("Epsilon (i.e. random action probability) to use"
                              "(default=0.05)"))
    parser.add_argument("--render", action="store_true",
                        help="Render the episode/s")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=True,
                               flat_obs=True)
    dqn_agent = DQNAgent(env, verbose=False, **vars(args))
    dqn_agent.load(args.policy_path)

    total_ret = 0
    total_steps = 0
    goals = 0
    print(f"\n{'-'*60}\nRunning DQN Policy:\n\t{args.policy_path}\n{'-'*60}")
    for i in range(args.eval_eps):
        ret, steps, goal = dqn_agent.run_eval_episode(
            env, args.render, args.epsilon
        )
        print(f"Episode {i} return={ret}, steps={steps}, goal reached={goal}")
        total_ret += ret
        total_steps += steps
        goals += int(goal)

    print(f"\n{'-'*60}\nDone\n{'-'*60}")
    print(f"Average Return = {total_ret / args.eval_eps:.2f}")
    print(f"Average Steps = {total_steps / args.eval_eps:.2f}")
    print(f"Goals = {goals} / {args.eval_eps}")
