"""An bruteforce agent that repeatedly cycles through all available actions in
order.

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python bruteforce_agent.py tiny

This will run the agent and display progress and final results to stdout.

To see available running arguments:

$ python bruteforce_agent.py --help
"""

from itertools import product

import nasim

LINE_BREAK = "-"*60


def run_bruteforce_agent(env, step_limit=1e6, verbose=True):
    """Run bruteforce agent on nasim environment.

    Parameters
    ----------
    env : nasim.NASimEnv
        the nasim environment to run agent on
    step_limit : int, optional
        the maximum number of steps to run agent for (default=1e6)
    verbose : bool, optional
        whether to print out progress messages or not (default=True)

    Returns
    -------
    int
        timesteps agent ran for
    float
        the total reward recieved by agent
    bool
        whether the goal was reached or not
    """
    if verbose:
        print(LINE_BREAK)
        print("STARTING EPISODE")
        print(LINE_BREAK)
        print("t: Reward")

    env.reset()
    total_reward = 0
    done = False
    steps = 0
    cycle_complete = False

    if env.flat_actions:
        act = 0
    else:
        act_iter = product(*[range(n) for n in env.action_space.nvec])

    while not done and steps < step_limit:
        if env.flat_actions:
            act = (act + 1) % env.action_space.n
            cycle_complete = (steps > 0 and act == 0)
        else:
            try:
                act = next(act_iter)
                cycle_complete = False
            except StopIteration:
                act_iter = product(*[range(n) for n in env.action_space.nvec])
                act = next(act_iter)
                cycle_complete = True

        _, rew, done, _ = env.step(act)
        total_reward += rew

        if cycle_complete and verbose:
            print(f"{steps}: {total_reward}")
        steps += 1

    if done and verbose:
        print(LINE_BREAK)
        print("EPISODE FINISHED")
        print(LINE_BREAK)
        print(f"Goal reached = {env.goal_reached()}")
        print(f"Total steps = {steps}")
        print(f"Total reward = {total_reward}")
    elif verbose:
        print(LINE_BREAK)
        print("STEP LIMIT REACHED")
        print(LINE_BREAK)

    if done:
        done = env.goal_reached()

    return steps, total_reward, done


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

    nasimenv = nasim.make_benchmark(
        args.env_name,
        args.seed,
        not args.partially_obs,
        not args.param_actions,
        not args.box_obs
    )
    if not args.param_actions:
        print(nasimenv.action_space.n)
    else:
        print(nasimenv.action_space.nvec)
    run_bruteforce_agent(nasimenv)
