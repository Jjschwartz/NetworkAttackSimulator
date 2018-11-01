from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from cyber_attack_simulator.experiments.experiment_util import get_scenario
import sys


def main():

    if len(sys.argv) != 2:
        print("Usage: python demo_visualize_graph.py <scenario>")
        return 1

    scenario_name = sys.argv[1]
    scenario = get_scenario(scenario_name)
    if scenario is None:
        return 1

    print("Displaying {} scenario".format(scenario_name))

    num_machines = scenario["machines"]
    num_services = scenario["services"]
    restrictiveness = scenario["restrictiveness"]

    print("Generating network configuration")
    print("\tnumber of machines =", num_machines)
    print("\tnumber of services =", num_services)
    print("\tfirewall restrictiveness =", restrictiveness)
    print("\tnumber of subnets =", 2 + (num_machines - 2) // 5)
    # env = CyberAttackSimulatorEnv.from_params(num_machines, num_services,
    #                                           restrictiveness=restrictiveness)
    env = CyberAttackSimulatorEnv.from_file("configs/small.yaml")

    env.render_network_graph(show=True)


if __name__ == "__main__":
    main()
