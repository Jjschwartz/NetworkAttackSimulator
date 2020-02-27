from nasim.env import make_benchmark_env

line_break = "-"*60


def choose_action(env, last_a):
    return (last_a + 1) % len(env.action_space)


def run_bruteforce_agent(env, step_limit=1e6, verbose=True):
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
        a = choose_action(env, a)
        # print("Take step")
        _, r, done, _ = env.step(a)
        # print("Step done")
        total_reward += r
        if (t+1) % len(env.action_space) == 0 and verbose:
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
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument("-o", "--partially_obs", action="store_true", help="Partially Observable Mode")
    args = parser.parse_args()

    env = make_benchmark_env(args.env_name, args.seed, args.partially_obs)
    print("Max score:", env.get_best_possible_score())
    run_bruteforce_agent(env)
