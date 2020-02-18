from nasim.env import make_benchmark_env


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    env = make_benchmark_env(args.scenario_name, args.seed)
    print("Max score:", env.get_best_possible_score())
    env.render_network_graph(show=True)
