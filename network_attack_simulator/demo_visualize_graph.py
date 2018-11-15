from network_attack_simulator.envs.environment import NetworkAttackSimulator
from network_attack_simulator.experiments.experiment_util import get_scenario
import sys


def main():

    if len(sys.argv) != 3:
        print("Usage: python demo_visualize_graph.py <scenario> <generate>")
        return 1

    scenario_name = sys.argv[1]
    generate = bool(int(sys.argv[2]))
    print(generate)
    print("Displaying {} scenario".format(scenario_name))
    if generate:
        print("Generating network configuration")
        scenario = get_scenario(scenario_name)
        if scenario is None:
            return 1
        num_machines = scenario["machines"]
        num_services = scenario["services"]
        restrictiveness = scenario["restrictiveness"]

        print("\tnumber of machines =", num_machines)
        print("\tnumber of services =", num_services)
        print("\tfirewall restrictiveness =", restrictiveness)
        env = NetworkAttackSimulator.from_params(num_machines, num_services,
                                                 restrictiveness=restrictiveness)
    else:
        print("Loading network configuration")
        env = NetworkAttackSimulator.from_file(scenario_name)

    print("Max score:", env.get_best_possible_score())
    env.render_network_graph(show=True)


if __name__ == "__main__":
    main()
