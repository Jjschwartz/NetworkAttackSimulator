from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv


def print_actions(action_space):
    print("\nSelect action by number and press enter: ")
    for i, a in enumerate(action_space):
        print("\t", i, a)
    print()


def get_action(action_space):
    actions = list(action_space.keys())
    print_actions(actions)

    while True:
        raw_input = input()
        try:
            num = int(raw_input)
            if num < 0 or num >= len(actions):
                print("Not a valid action number, please try again...")
            else:
                break
        except ValueError:
            print("Not a valid action number, please try again...")
    return action_space[actions[num]]


if __name__ == "__main__":
    num_machines = 4
    num_services = 2
    env = CyberAttackSimulatorEnv(num_machines, num_services)
    action_space = env.action_space
    obs = env.reset()

    done = False
    total_reward = 0
    while not done:
        env.render("ASCI")
        env.render("readable")
        action = get_action(action_space)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print("Complete")
    env.render("ASCI")
    print("Total reward = {0}".format(total_reward))
