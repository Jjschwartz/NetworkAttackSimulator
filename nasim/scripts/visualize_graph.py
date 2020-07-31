"""Environment network graph visualizer

This script allows the user to visualize the network graph for a chosen
benchmark scenario.
"""

import nasim


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str,
                        help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random seed (default=0)")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.scenario_name, args.seed)
    env.render_network_graph(show=True)
