"""An agent that lets the user interact with NASim using the keyboard.

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python keyboard_agent.py tiny

This will run the agent and display the game in stdout.

To see available running arguments:

$ python keyboard_agent.py--help
"""
import nasim
from nasim.envs.action import Exploit, PrivilegeEscalation


LINE_BREAK = "-"*60
LINE_BREAK2 = "="*60


def print_actions(action_space):
    for a in range(action_space.n):
        print(f"{a} {action_space.get_action(a)}")
    print(LINE_BREAK)


def choose_flat_action(env):
    print_actions(env.action_space)
    while True:
        try:
            idx = int(input("Choose action number: "))
            action = env.action_space.get_action(idx)
            print(f"Performing: {action}")
            return action
        except Exception:
            print("Invalid choice. Try again.")


def display_actions(actions):
    action_names = list(actions)
    for i, name in enumerate(action_names):
        a_def = actions[name]
        output = [f"{i} {name}:"]
        output.extend([f"{k}={v}" for k, v in a_def.items()])
        print(" ".join(output))


def choose_item(items):
    while True:
        try:
            idx = int(input("Choose number: "))
            return items[idx]
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

    print("------------------------")
    print("2. Choose Target Subnet:")
    print("------------------------")
    num_subnets = env.action_space.nvec[1]
    while True:
        try:
            subnet = int(input(f"Choose subnet in [1, {num_subnets}]: "))
            if subnet < 1 or subnet > num_subnets:
                raise ValueError()
            break
        except Exception:
            print("Invalid choice. Try again.")

    print("----------------------")
    print("3. Choose Target Host:")
    print("----------------------")
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
    if atype not in (Exploit, PrivilegeEscalation):
        action = env.action_space.get_action(avec)
        print("----------------")
        print(f"ACTION SELECTED: {action}")
        return action

    target = (subnet, host)
    if atype == Exploit:
        print("------------------")
        print("4. Choose Exploit:")
        print("------------------")
        exploits = env.scenario.exploits
        display_actions(exploits)
        e_name = choose_item(list(exploits))
        action = Exploit(name=e_name, target=target, **exploits[e_name])
    else:
        print("------------------")
        print("4. Choose Privilege Escalation:")
        print("------------------")
        privescs = env.scenario.privescs
        display_actions(privescs)
        pe_name = choose_item(list(privescs))
        action = PrivilegeEscalation(
            name=pe_name, target=target, **privescs[pe_name]
        )

    print("----------------")
    print(f"ACTION SELECTED: {action}")
    return action


def choose_action(env):
    input("Press enter to choose next action..")
    print("\n" + LINE_BREAK2)
    print("CHOOSE ACTION")
    print(LINE_BREAK2)
    if env.flat_actions:
        return choose_flat_action(env)
    return choose_param_action(env)


def run_keyboard_agent(env, render_mode="readable"):
    """Run Keyboard agent

    Parameters
    ----------
    env : NASimEnv
        the environment
    render_mode : str, optional
        display mode for environment (default="readable")

    Returns
    -------
    int
        final return
    int
        steps taken
    bool
        whether goal reached or not
    """
    print(LINE_BREAK2)
    print("STARTING EPISODE")
    print(LINE_BREAK2)

    o = env.reset()
    env.render(render_mode)
    total_reward = 0
    total_steps = 0
    done = False
    while not done:
        a = choose_action(env)
        o, r, done, _ = env.step(a)
        total_reward += r
        total_steps += 1
        print("\n" + LINE_BREAK2)
        print("OBSERVATION RECIEVED")
        print(LINE_BREAK2)
        env.render(render_mode)
        print(f"Reward={r}")
        print(f"Done={done}")
        print(LINE_BREAK)

    if done:
        done = env.goal_reached()

    return total_reward, total_steps, done


def run_generative_keyboard_agent(env, render_mode="readable"):
    """Run Keyboard agent in generative mode.

    The experience is the same as the normal mode, this is mainly useful
    for testing.

    Parameters
    ----------
    env : NASimEnv
        the environment
    render_mode : str, optional
        display mode for environment (default="readable")

    Returns
    -------
    int
        final return
    int
        steps taken
    bool
        whether goal reached or not
    """
    print(LINE_BREAK2)
    print("STARTING EPISODE")
    print(LINE_BREAK2)

    o = env.reset()
    s = env.current_state
    env.render_state(render_mode, s)
    env.render(render_mode, o)

    total_reward = 0
    total_steps = 0
    done = False
    while not done:
        a = choose_action(env)
        ns, o, r, done, _ = env.generative_step(s, a)
        total_reward += r
        total_steps += 1
        print(LINE_BREAK2)
        print("NEXT STATE")
        print(LINE_BREAK2)
        env.render_state(render_mode, ns)
        print("\n" + LINE_BREAK2)
        print("OBSERVATION RECIEVED")
        print(LINE_BREAK2)
        env.render(render_mode, o)
        print(f"Reward={r}")
        print(f"Done={done}")
        print(LINE_BREAK)
        s = ns

    if done:
        done = env.goal_reached()

    return total_reward, total_steps, done


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
                        help=("Generative environment mode. This makes no"
                              " difference for the player, but is useful"
                              " for testing."))
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=not args.param_actions,
                               flat_obs=True)
    if args.use_generative:
        total_reward, steps, goal = run_generative_keyboard_agent(env)
    else:
        total_reward, steps, goal = run_keyboard_agent(env)

    print(LINE_BREAK2)
    print("EPISODE FINISHED")
    print(LINE_BREAK)
    print(f"Goal reached = {goal}")
    print(f"Total reward = {total_reward}")
    print(f"Steps taken = {steps}")
