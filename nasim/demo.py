"""Script for running NASim demo

Usage
-----

$ python demo [-ai] [-h] env_name
"""

import os.path as osp

import nasim
from nasim.agents.dqn_agent import DQNAgent
from nasim.agents.keyboard_agent import run_keyboard_agent


DQN_POLICY_DIR = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "agents",
    "policies"
)
DQN_POLICIES = {
    "tiny": osp.join(DQN_POLICY_DIR, "dqn_tiny.pt"),
    "small": osp.join(DQN_POLICY_DIR, "dqn_small.pt")
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "NASim demo. Play as the hacker, trying to gain access"
            " to sensitive information on the network, or run a pre-trained"
            " AI hacker."
        )
    )
    parser.add_argument("env_name", type=str,
                        help="benchmark scenario name")
    parser.add_argument("-ai", "--run_ai", action="store_true",
                        help=("Run AI policy (currently ony supported for"
                              " 'tiny' and 'small' environments"))
    args = parser.parse_args()

    if args.run_ai:
        assert args.env_name in DQN_POLICIES, \
            ("AI demo only supported for the following environments:"
             f" {list(DQN_POLICIES)}")

    env = nasim.make_benchmark(
        args.env_name,
        fully_obs=True,
        flat_actions=True,
        flat_obs=True
    )

    line_break = f"\n{'-'*60}"
    print(line_break)
    print(f"Running Demo on {args.env_name} environment")
    if args.run_ai:
        print("Using AI policy")
        print(line_break)
        dqn_agent = DQNAgent(env, verbose=False, **vars(args))
        dqn_agent.load(DQN_POLICIES[args.env_name])
        ret, steps, goal = dqn_agent.run_eval_episode(
            env, True, 0.01, "readable"
        )
    else:
        print("Player controlled")
        print(line_break)
        ret, steps, goal = run_keyboard_agent(env, "readable")

    print(line_break)
    print(f"Episode Complete")
    print(line_break)
    if goal:
        print("Goal accomplished. Sensitive data retrieved!")
    print(f"Final Score={ret}")
    print(f"Steps taken={steps}")
