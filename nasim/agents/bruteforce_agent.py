from itertools import product

import nasim

line_break = "-"*60


def run_bruteforce_agent(env, step_limit=1e6, verbose=True):
    if verbose:
        print(line_break)
        print("STARTING EPISODE")
        print(line_break)
        print(f"t: Reward")

    env.reset()
    total_reward = 0
    done = False
    t = 0
    cycle_complete = False

    if env.flat_actions:
        a = 0
    else:
        a_iter = product(*[range(n) for n in env.action_space.nvec])

    while not done and t < step_limit:
        if env.flat_actions:
            a = (a + 1) % env.action_space.n
            cycle_complete = (t > 0 and a == 0)
        else:
            try:
                a = next(a_iter)
                cycle_complete = False
            except StopIteration:
                a_iter = product(*[range(n) for n in env.action_space.nvec])
                a = next(a_iter)
                cycle_complete = True

        _, r, done, _ = env.step(a)
        total_reward += r

        if cycle_complete and verbose:
            print(f"{t}: {total_reward}")
        t += 1

    if done and verbose:
        print(line_break)
        print("EPISODE FINISHED")
        print(line_break)
        print(f"Goal reached = {env.goal_reached()}")
        print(f"Total steps = {t}")
        print(f"Total reward = {total_reward}")
    elif verbose:
        print(line_break)
        print("STEP LIMIT REACHED")
        print(line_break)

    if done:
        done = env.goal_reached()

    return t, total_reward, done


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("-p", "--param_actions", action="store_true",
                        help="Use Parameterised action space")
    parser.add_argument("-f", "--box_obs", action="store_true",
                        help="Use 2D observation space")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               not args.partially_obs,
                               not args.param_actions,
                               not args.box_obs)
    if not args.param_actions:
        print(env.action_space.n)
    else:
        print(env.action_space.nvec)
    run_bruteforce_agent(env)
