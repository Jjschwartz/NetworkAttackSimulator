import nasim

line_break = "-"*60


def run_random_agent(env, step_limit=1e6, verbose=True):
    if verbose:
        print(line_break)
        print("STARTING EPISODE")
        print(line_break)

    env.reset()
    total_reward = 0
    done = False
    t = 0
    a = 0
    print(f"t: Reward")
    while not done and t < step_limit:
        # print("Choose action")
        a = env.action_space.sample()
        # print("Take step")
        _, r, done, _ = env.step(a)
        # print("Step done")
        total_reward += r
        if (t+1) % 100 == 0 and verbose:
            print(f"{t}: {total_reward}")
        t += 1

    if done and verbose:
        print(line_break)
        print("EPISODE FINISHED")
        print(line_break)
        print(f"Total reward = {total_reward}")
    elif verbose:
        print(line_break)
        print("STEP LIMIT REACHED")
        print(line_break)
    return t, total_reward, done


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str,
                        help="benchmark scenario name")
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
    print(env.action_space)
    run_random_agent(env)
