"""A random agent that selects a random action at each step

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python random_agent.py tiny

This will run the agent and display progress and final results to stdout.

To see available running arguments:

$ python random_agent.py --help
"""

import numpy as np

import nasim

LINE_BREAK = "-"*60


def run_random_agent(env, step_limit=1e6, verbose=True):
    if verbose:
        print(LINE_BREAK)
        print("STARTING EPISODE")
        print(LINE_BREAK)
        print(f"t: Reward")

    env.reset()
    total_reward = 0
    done = False
    t = 0
    a = 0

    while not done and t < step_limit:
        a = env.action_space.sample()
        _, r, done, _ = env.step(a)
        total_reward += r
        if (t+1) % 100 == 0 and verbose:
            print(f"{t}: {total_reward}")
        t += 1

    if done and verbose:
        print(LINE_BREAK)
        print("EPISODE FINISHED")
        print(LINE_BREAK)
        print(f"Total steps = {t}")
        print(f"Total reward = {total_reward}")
    elif verbose:
        print(LINE_BREAK)
        print("STEP LIMIT REACHED")
        print(LINE_BREAK)

    if done:
        done = env.goal_reached()

    return t, total_reward, done


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str,
                        help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="number of random runs to perform (default=1)")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("-p", "--param_actions", action="store_true",
                        help="Use Parameterised action space")
    parser.add_argument("-f", "--box_obs", action="store_true",
                        help="Use 2D observation space")
    args = parser.parse_args()

    seed = args.seed
    run_steps = []
    run_rewards = []
    run_goals = 0
    for i in range(args.runs):
        env = nasim.make_benchmark(args.env_name,
                                   seed,
                                   not args.partially_obs,
                                   not args.param_actions,
                                   not args.box_obs)
        steps, reward, done = run_random_agent(env, verbose=False)
        run_steps.append(steps)
        run_rewards.append(reward)
        run_goals += int(done)
        seed += 1

        if args.runs > 1:
            print(f"Run {i}:")
            print(f"\tSteps = {steps}")
            print(f"\tReward = {reward}")
            print(f"\tGoal reached = {done}")

    run_steps = np.array(run_steps)
    run_rewards = np.array(run_rewards)

    print(LINE_BREAK)
    print("Random Agent Runs Complete")
    print(LINE_BREAK)
    print(f"Mean steps = {run_steps.mean():.2f} +/- {run_steps.std():.2f}")
    print(f"Mean rewards = {run_rewards.mean():.2f} "
          f"+/- {run_rewards.std():.2f}")
    print(f"Goals reached = {run_goals} / {args.runs}")
