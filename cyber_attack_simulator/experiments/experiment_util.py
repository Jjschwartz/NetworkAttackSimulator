from collections import OrderedDict
from cyber_attack_simulator.agents.q_learning import QLearningAgent


# experiment scenarios
scenarios = OrderedDict()
scenarios["tiny"] = {"machines": 3,
                     "services": 1,
                     "restrictiveness": 1,
                     "episodes": 1000,
                     "steps": 100,
                     "timeout": 300}
scenarios["small"] = {"machines": 8,
                      "services": 3,
                      "restrictiveness": 2,
                      "episodes": 1000,
                      "steps": 200,
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
                      "episodes": 29000,
                      "steps": 1200,
                      "timeout": 1800}
scenarios["huge"] = {"machines": 37,
                     "services": 10,
                     "restrictiveness": 3,
                     "episodes": 110000,
                     "steps": 2400,
                     "timeout": 3600}


# Experiment agents
agents = OrderedDict()
agents["td_egreedy"] = {
    "tiny": {"type": "egreedy", "alpha": 0.05, "gamma": 0.9, "epsilon_decay_lambda": 0.01},
    "small": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001},
    "medium": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001},
    "large": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001},
    "huge": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001}
    }
agents["td_ucb"] = {
    "tiny": {"type": "UCB", "alpha": 0.05, "gamma": 0.9, "c": 1.0},
    "small": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0},
    "medium": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0},
    "large": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0},
    "huge": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0}
    }
agents["dqn"] = {
    "tiny": {"hidden_units": 256, "gamma": 0.5, "epsilon_decay_lambda": 0.01},
    "small": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001},
    "medium": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001},
    "large": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001},
    "huge": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001}
     }


def is_valid_scenario(scenario_name, verbose=False):
    if scenario_name not in scenarios.keys():
        if verbose:
            print("Scenario must be one of: {}".format(list(scenarios.keys())))
        return False
    return True


def get_scenarios():
    return scenarios


def get_scenario(scenario_name):
    if is_valid_scenario(scenario_name, verbose=True):
        return scenarios[scenario_name]
    return None


def is_valid_agent(agent_name, verbose=False):
    if agent_name not in agents.keys():
        if verbose:
            print("Agent must be one of: {}".format(list(agents.keys())))
        return False
    return True


def get_agents():
    return agents


def get_agent(agent_name, scenario_name, env):
    """ Returns a new agent instance """
    if not is_valid_agent(agent_name, verbose=True):
        return None
    if not is_valid_scenario(scenario_name, verbose=True):
        return None

    agent_params = agents[agent_name][scenario_name]
    if agent_name == "dqn":
        # only import when necessary
        from cyber_attack_simulator.agents.dqn import DQNAgent
        state_size = env.get_state_size()
        num_actions = env.get_num_actions()
        return DQNAgent(state_size, num_actions, **agent_params)
    return QLearningAgent(**agent_params)
