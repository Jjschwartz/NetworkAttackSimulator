"""
This module contains some functionallity for finding the best hyperparameter choice
for different problems.
"""
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv as Cyber
from cyber_attack_simulator.agents.dqn import DQNAgent
from cyber_attack_simulator.agents.q_learning import QLearningAgent

import sys
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


# whether to just automatically select hyperparam values based on mean total reward
# or show results and ask user to choose
AUTOMODE = True

# list of agents and their hyperparameters
# hyperparameter list is in order of priority for tuning
AGENTS = {"DQN": {"constructor": DQNAgent,
                  "hyperparameters": ["hidden_units", "gamma", "epsilon_decay_lambda"]},
          "TD-Q-e-greedy": {"constructor": QLearningAgent,
                            "hyperparameters": ["alpha", "gamma", "epsilon_decay_lambda"]},
          "TD-Q-UCB": {"constructor": QLearningAgent,
                       "hyperparameters": ["alpha", "gamma", "c"]}}

# dictionary of hyperparameters and ranges
# default values are first entry
# list values to 4 decimal places for when doing float comparisons
H_PARAMS = {}
H_PARAMS["alpha"] = [0.1000, 0.0100, 0.0500, 0.3000, 0.5000, 0.7000, 0.9000]
# H_PARAMS["alpha"] = [0.1000]
H_PARAMS["gamma"] = [0.9000, 0.5000, 0.8000, 0.9900, 0.9990]
H_PARAMS["c"] = [1.0000, 2.0000, 5.0000, 10.0000]
H_PARAMS["hidden_units"] = [64, 128, 256, 512]
H_PARAMS["epsilon_decay_lambda"] = [0.0100, 0.0001, 0.0010, 0.1000, 0.2000, 0.5000]

# Experiment scenarios
scenarios = OrderedDict()
scenarios["tiny"] = {"machines": 3,
                     "services": 1,
                     "restrictiveness": 1,
                     "episodes": 1000,
                     "steps": 200,
                     "timeout": 300}
scenarios["small"] = {"machines": 8,
                      "services": 3,
                      "restrictiveness": 2,
                      "episodes": 2000,
                      "steps": 600,
                      "timeout": 600}
scenarios["medium"] = {"machines": 13,
                       "services": 5,
                       "restrictiveness": 3,
                       "episodes": 2000,
                       "steps": 900,
                       "timeout": 900}
scenarios["large"] = {"machines": 18,
                      "services": 6,
                      "restrictiveness": 3,
                      "episodes": 4000,
                      "steps": 1200,
                      "timeout": 1800}
scenarios["huge"] = {"machines": 37,
                     "services": 10,
                     "restrictiveness": 3,
                     "episodes": 10000,
                     "steps": 2400,
                     "timeout": 3600}

# Environment parameters
RUNS = 5      # number of runs to do per hyperparameter configuration
EVAL_WINDOW = 100      # number of episodes to evaluate policy over
LEARNING_START = 100    # from which episode to start tracking total rewards

# Environment constants
UNIFORM = True
EXPLOIT_PROB = "mixed"
R_SENS = R_USR = 10
COST_EXP = COST_SCAN = 1

# To control progress message printing for individual episodes within runs
VERBOSE = False

# Experiment scenarios to tune agent for
scenarios_list = ["tiny"]
scenarios_list = ["large", "huge"]


def load_agent(agent_type, env, agent_params):
    if agent_type == "TD-Q-UCB":
        agent_params["type"] = "UCB"
    elif agent_type == "TD-Q-e-greedy":
        agent_params["type"] = "egreedy"
    elif agent_type == "DQN":
        agent_params["state_size"] = env.get_state_size()
        agent_params["num_actions"] = env.get_num_actions()
    return AGENTS[agent_type]["constructor"](**agent_params)


def get_default_agent_params(agent_type):
    """ Get the default hyperparam dictionary for agent type
    Dictionary is ordered by priority of tuning """
    agent_params = OrderedDict()
    for h in AGENTS[agent_type]["hyperparameters"]:
        agent_params[h] = H_PARAMS[h][0]
    return agent_params


def run_agent(scenario, agent_type, agent_params):
    """
    Run agent on given environment for set number of runs and return averaged
    rewards (averaged over runs)
    """
    scenario_params = scenarios[scenario]
    M = scenario_params["machines"]
    S = scenario_params["services"]
    rve = scenario_params["restrictiveness"]
    num_episodes = scenario_params["episodes"]
    max_steps = scenario_params["steps"]
    timeout = scenario_params["timeout"]

    run_rewards = []

    for t in range(RUNS):
        env = Cyber.from_params(M, S,
                                r_sensitive=R_SENS,  r_user=R_USR,
                                exploit_cost=COST_EXP, scan_cost=COST_SCAN,
                                restrictiveness=rve, exploit_probs=EXPLOIT_PROB, seed=t)
        agent = load_agent(agent_type, env, agent_params)
        ep_tsteps, ep_rews, ep_times = agent.train(env, num_episodes, max_steps, timeout, VERBOSE)
        run_rewards.append(ep_rews)
        print("\t\tRun {} - total reward = {}".format(t, sum(ep_rews)))

    return np.mean(run_rewards, axis=0)


def test_hyperparam(scenario, agent_type, agent_params, tune_param):
    """ """

    print("\nTuning hyperparam {}".format(tune_param))
    print("Using scenario hyperparameters {}".format(agent_params))

    h_results = []
    h_values = H_PARAMS[tune_param]

    # a. for each value of given hyperparameter
    for h_val in h_values:
        print("\t{} = {}".format(tune_param, h_val))
        test_params = deepcopy(agent_params)
        test_params[tune_param] = h_val
        rew = run_agent(scenario, agent_type, test_params)
        h_results.append((h_val, rew))
        print("\t{} - Mean total reward = {}".format(tune_param, sum(rew)))

    return h_results


def report_results(scenario, agent_type, h_param, results):
    """ report the result of running test on given hyperpara
        returns the best hyperparam value
    """
    print("\nReporting results for scenario={}, agent={}, hyperparam={}"
          .format(scenario, agent_type, h_param))

    for res in results:
        print("\t{} = {}".format(res[0], sum(res[1])))

    for res in results:
        episodes = list(range(len(res[1])))
        plt.plot(episodes, np.cumsum(res[1]), label=res[0])

    plt.ylabel("Cumulative reward")
    plt.xlabel("Episode")
    plt.legend()
    plt.show()


def get_best_hyperparam_value(results):
    best_val = 0
    max_mean_total_reward = float('-inf')
    for res in results:
        res_sum = sum(res[1])
        if res_sum > max_mean_total_reward:
            best_val = res[0]
            max_mean_total_reward = res_sum
    return best_val


def get_user_choice(h_param):
    """ """

    h_values = H_PARAMS[h_param]
    good_choice = False
    message = "Please select value to use for {}: ".format(h_param)

    while not good_choice:
        choice = input(message)
        try:
            # attempt to convert user input into correct format
            new_val = type(h_values[0])(choice)
            if type(new_val) is float:
                new_val = round(new_val, 4)
            if new_val in h_values:
                good_choice = True
        except Exception:
            print("Not a valid choice, please select a value from: {}".format(h_values))

    return new_val


def main():

    if len(sys.argv) != 2:
        print("Usage: python tuner.py <agent type>")
        print("Available agent types: {}".format(AGENTS.keys()))
        return 1

    agent_type = sys.argv[1]
    if agent_type not in AGENTS:
        print("Invalid agent type: {}. Agent type must be one of: {}"
              .format(agent_type, AGENTS.keys()))
        return 1

    print("Tuning {} agent".format(agent_type))

    agent_params = get_default_agent_params(agent_type)
    print("Default params = {}".format(agent_params))

    for scenario in scenarios_list:
        print("Tuning for scenario = {}".format(scenario))
        print("Aaaaand awaaaay weeee goooo!!")
        scenario_params = deepcopy(agent_params)
        for h in scenario_params.keys():
            results = test_hyperparam(scenario, agent_type, scenario_params, h)
            if not AUTOMODE:
                report_results(scenario, agent_type, h, results)
                new_val = get_user_choice(h)
            else:
                new_val = get_best_hyperparam_value(results)
            print("Selected {}={} for new value".format(h, new_val))
            scenario_params[h] = new_val
        # 2. print final hyperparam values for given scenario
        print("\n\nFinished tuning params for scenario={} and agent={}"
              .format(scenario, agent_type))
        print("Final Hyperam values:")
        for h, v in scenario_params.items():
            print("\t{} = {}".format(h, v))
        print("\nRemember to write these down somewhere safe")
        if not AUTOMODE:
            input("Press any key to continue onto the next scenario (if there is one...)")


if __name__ == "__main__":
    main()
