import sys
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv


def print_state(obs):
    output = ""
    for m, vals in obs.items():
        output += "Machine = " + str(m) + " =>\n"

        output += "\tServices:\n"
        for i, s in enumerate(vals["service_info"]):
            output += "\t\t{0} = {1}".format(i, str(s))
            output += "\n"

        output += "\treachable: {0}\n".format(vals["reachable"])
        output += "\tcompromised: {0}\n".format(vals["compromised"])
        output += "\tsensitive: {0}\n".format(vals["sensitive"])
    sys.stdout.write(output)


def print_actions(action_space):
    print("\nSelect action by number and press enter: ")
    for i, a in enumerate(action_space):
        print("\t", i, a)
    print()


def get_action(action_space):
    print_actions(action_space)

    while True:
        raw_input = input()
        try:
            num = int(raw_input)
            if num < 0 or num >= len(action_space):
                print("Not a valid action number, please try again...")
            else:
                break
        except ValueError:
            print("Not a valid action number, please try again...")
    return action_space[num]


if __name__ == "__main__":
    num_machines = 3
    num_services = 2
    env = CyberAttackSimulatorEnv(num_machines, num_services)
    action_space = sorted(list(env.action_space))
    obs = env.reset()

    done = False
    total_reward = 0
    while not done:
        print_state(obs)
        action = get_action(action_space)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print("Complete")
    print("Total reward = {0}".format(total_reward))
