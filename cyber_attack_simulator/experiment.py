"""
Run Experimental scenarios
"""
import sys
from collections import OrderedDict
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv as Cyber
from cyber_attack_simulator.agents.sarsa import SarsaAgent
from cyber_attack_simulator.agents.q_learning import QLearningAgent

# Experiment network size scenarios
# (|M|, |E|)
scenarios = OrderedDict()
scenarios["smallest"] = (3, 1)
scenarios["small"] = (5, 3)
# scenarios["med"] = (7, 4)
# scenarios["large"] = (13, 5)

# Values for other experiment paramaters
restrictiveness = ["high", "medium", "none"]
exploit_probs = ["deterministic", "stochastic"]

# Experiment constants
timeout = 180   # training time limit in seconds
training_runs = 5      # number of training runs to do
eval_period = 100      # number of runs to evaluate policy over

# Agent params
qlearning = True    # use qlearning or sarsa agent


def get_restrictiveness(level, E):
    if level == 0 or level == restrictiveness[0]:
        # none
        return E
    elif level == 1 or level == restrictiveness[1]:
        # medium
        return 3
    else:
        # high
        return 1


def get_exploit_prob(level):
    if level == 0 or level == exploit_probs[0]:
        return 1.0
    else:
        return "mixed"


def get_training_params(M, E):
    num_episodes = M ** 2 * 100
    max_steps = E * 2 * 100
    return num_episodes, max_steps


def is_path_found(ep_tsteps, max_steps):
    # path found if episode timesteps for last eval_period episodes is < max_steps
    for t in ep_tsteps[-eval_period:]:
        if t >= max_steps:
            return False
    return True


def get_expected_rewards(ep_rewards):
    return sum(ep_rewards[-eval_period:]) / eval_period


def run_experiment(agent, M, E, R, P):

    print("Running experiment: M={}, E={}, R={}, P={}".format(M, E, R, P))

    num_episodes, max_steps = get_training_params(M, E)
    paths_found = []
    exp_rewards = []

    P = get_exploit_prob(P)
    for t in range(training_runs):
        env = Cyber.from_params(M, E, restrictiveness=R, exploit_probs=P, seed=t)
        ep_tsteps, ep_rews, ep_times = agent.train(env, num_episodes, max_steps)
        # path = agent.generate_episode(env, max_steps)
        # path_found = len(path) < max_steps
        path_found = is_path_found(ep_tsteps, max_steps)
        paths_found.append(path_found)
        # exp_reward = agent.evaluate_agent(env, exp_reward_runs, max_steps)
        exp_reward = get_expected_rewards(ep_rews)
        exp_rewards.append(exp_reward)
        print("\tTraining run {0} - path_found={1} - exp_reward={2}"
              .format(t, path_found, exp_reward))

    path_found_prob = paths_found.count(True) / training_runs
    exp_reward = sum(exp_rewards) / training_runs

    return path_found_prob, exp_reward


def main():

    if len(sys.argv) != 2:
        print("Usage: python experiment.py outfile")
        return 1

    print("Welcome to your friendly experiment runner")
    result_file = open(sys.argv[1], 'w')
    # write header line
    result_file.write("M,E,R,P,path_found,exp_reward\n")

    alpha = 0.1
    gamma = 0.9
    c = 10.0
    if qlearning:
        agent = QLearningAgent("UCB", alpha, gamma, c)
    else:
        agent = SarsaAgent("UCB", alpha, gamma, c)

    for k, v in scenarios.items():
        M = v[0]
        E = v[1]
        for r in restrictiveness:
            R = get_restrictiveness(r, E)
            if R > E or (E == 1 and r != restrictiveness[0]) \
                    or (E == 3 and r == restrictiveness[1]):
                continue
            for P in exploit_probs:
                path_found_prob, exp_reward = run_experiment(agent, M, E, R, P)
                result_file.write("{0},{1},{2},{3},{4:.2f},{5}\n"
                                  .format(M, E, R, P, path_found_prob, exp_reward))

    result_file.close()


if __name__ == "__main__":
    main()
