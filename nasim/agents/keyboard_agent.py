"""Play and scenario using key board """
from nasim.env import make_benchmark_env

line_break = "-"*60


def print_actions(action_space):
    print(line_break)
    print("CHOOSE ACTION")
    print(line_break)
    for i, a in enumerate(action_space):
        print(f"{i} {a}")
    print(line_break)


def choose_action(env):
    print_actions(env.action_space)
    while True:
        try:
            idx = int(input("Choose index: "))
            action = env.action_space[idx]
            print(f"Performing: {action}")
            return action
        except Exception:
            print("Invalid choice. Try again.")


def run_keyboard_agent(env):
    print(line_break)
    print("STARTING EPISODE")
    print(line_break)

    o = env.reset()
    total_reward = 0
    done = False
    while not done:
        # print(s)
        print(o._tensor)
        # env.render()
        a = choose_action(env)
        o, r, done, _ = env.step(a)
        total_reward += r

    print(line_break)
    print("EPISODE FINISHED")
    print(line_break)
    print(f"Total reward = {total_reward}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument("-o", "--partially_obs", action="store_true", help="Partially Observable Mode")
    args = parser.parse_args()

    env = make_benchmark_env(args.scenario_name, args.seed, args.partially_obs)
    print("Max score:", env.get_best_possible_score())
    run_keyboard_agent(env)
