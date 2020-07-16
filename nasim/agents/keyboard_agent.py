"""Play and scenario using key board """

import nasim
from nasim.env.action import Exploit


line_break = "-"*60
line_break2 = "="*60


def print_actions(action_space):
    for a in range(action_space.n):
        print(f"{a} {action_space.get_action(a)}")
    print(line_break)


def choose_flat_action(env):
    print_actions(env.action_space)
    while True:
        try:
            idx = int(input("Choose index: "))
            action = env.action_space.get_action(idx)
            print(f"Performing: {action}")
            return action
        except Exception:
            print("Invalid choice. Try again.")


def choose_param_action(env):
    print("1. Choose Action Type:")
    print("----------------------")
    for i, atype in enumerate(env.action_space.action_types):
        print(f"{i} {atype.__name__}")
    while True:
        try:
            atype_idx = int(input("Choose index: "))
            # check idx valid
            atype = env.action_space.action_types[atype_idx]
            break
        except Exception:
            print("Invalid choice. Try again.")

    print("-----------------")
    print("2. Choose Subnet:")
    print("-----------------")
    num_subnets = env.action_space.nvec[1]
    while True:
        try:
            subnet = int(input(f"Choose subnet in [1, {num_subnets}]: "))
            if subnet < 1 or subnet > num_subnets:
                raise ValueError()
            break
        except Exception:
            print("Invalid choice. Try again.")

    print("---------------")
    print("3. Choose Host:")
    print("---------------")
    num_hosts = env.scenario.subnets[subnet]
    while True:
        try:
            host = int(input(f"Choose host in [0, {num_hosts-1}]: "))
            if host < 0 or host > num_hosts-1:
                raise ValueError()
            break
        except Exception:
            print("Invalid choice. Try again.")

    # subnet-1, since action_space handles exclusion of internet subnet
    avec = [atype_idx, subnet-1, host, 0, 0]
    if atype != Exploit:
        action = env.action_space.get_action(avec)
        print("----------------")
        print(f"ACTION SELECTED: {action}")
        return action

    print("------------------")
    print("4. Choose Exploit:")
    print("------------------")
    exploits = env.scenario.exploits
    exploit_names = list(exploits)
    for i, e_name in enumerate(exploit_names):
        e_def = exploits[e_name]
        e_srv = e_def['service']
        e_os = e_def['os']
        e_prob = e_def['prob']
        e_cost = e_def['cost']
        print(f"{i} {e_name}: service={e_srv} os={e_os} "
              f"prob={e_prob} cost={e_cost}")
    while True:
        try:
            e_idx = int(input("Choose index: "))
            e_name = exploit_names[e_idx]
            break
        except Exception:
            print("Invalid choice. Try again.")

    target = (subnet, host)
    action = Exploit(name=e_name, target=target, **exploits[e_name])
    print("----------------")
    print(f"ACTION SELECTED: {action}")
    return action


def choose_action(env):
    input("Press enter to choose next action..")
    print(line_break2)
    print("CHOOSE ACTION")
    print(line_break2)
    if env.flat_actions:
        return choose_flat_action(env)
    return choose_param_action(env)


def run_keyboard_agent(env):
    """Run Keyboard agent

    Parameters
    ----------
    env : NASimEnv
        the environment
    """
    print(line_break2)
    print("STARTING EPISODE")
    print(line_break2)

    o = env.reset()
    env.render("readable")
    total_reward = 0
    done = False
    while not done:
        a = choose_action(env)
        o, r, done, _ = env.step(a)
        total_reward += r
        print(line_break2)
        print("OBSERVATION RECIEVED")
        print(line_break2)
        env.render("readable")
        print(f"Reward={r}")
        print(f"Done={done}")
        print(line_break)

    if done:
        done = env.goal_reached()

    print(line_break2)
    print("EPISODE FINISHED")
    print(line_break)
    print(f"Goal reached = {done}")
    print(f"Total reward = {total_reward}")


def run_generative_keyboard_agent(env):
    """Run Keyboard agent in generative mode.

    The experience is the same as the normal mode, this is mainly useful
    for testing.

    Parameters
    ----------
    env : NASimEnv
        the environment
    """
    print(line_break2)
    print("STARTING EPISODE")
    print(line_break2)

    o = env.reset()
    s = env.current_state
    env.render_state("readable", s)
    env.render("readable", o)

    total_reward = 0
    done = False
    while not done:
        a = choose_action(env)
        ns, o, r, done, info = env.generative_step(s, a)
        total_reward += r
        print(line_break2)
        print("NEXT STATE")
        print(line_break2)
        env.render_state("readable", ns)
        print(line_break2)
        print("OBSERVATION RECIEVED")
        print(line_break2)
        env.render("readable", o)
        print(f"Reward={r}")
        print(f"Done={done}")
        print(line_break)
        s = ns

    if done:
        done = env.goal_reached()

    print(line_break)
    print("EPISODE FINISHED")
    print(line_break)
    print(f"Goal reached = {done}")
    print(f"Total reward = {total_reward}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str,
                        help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="random seed (default=None)")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("-p", "--param_actions", action="store_true",
                        help="Use Parameterised action space")
    parser.add_argument("-g", "--use_generative", action="store_true",
                        help="Generative environment mode")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=not args.param_actions,
                               flat_obs=True)
    if args.use_generative:
        run_generative_keyboard_agent(env)
    else:
        run_keyboard_agent(env)
