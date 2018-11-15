from collections import OrderedDict
from network_attack_simulator.envs.environment import NetworkAttackSimulator as Cyber
from network_attack_simulator.agents.q_learning import QLearningAgent
from network_attack_simulator.agents.random import RandomAgent


# generated environment constants
UNIFORM = False
EXPLOIT_PROB = "mixed"
R_SENS = R_USR = 10
COST_EXP = COST_SCAN = 1


# experiment scenarios
scenarios = OrderedDict()
scenarios["tiny"] = {"machines": 3,
                     "services": 1,
                     "restrictiveness": 1,
                     "episodes": 1000,
                     "steps": 500,
                     "timeout": 600,
                     "max_score": 17,
                     "generate": True}
scenarios["small"] = {"machines": 8,
                      "services": 3,
                      "restrictiveness": 2,
                      "episodes": 1000,
                      "steps": 500,
                      "timeout": 300,
                      "max_score": 16,
                      "generate": True}
scenarios["medium"] = {"machines": 13,
                       "services": 5,
                       "restrictiveness": 3,
                       "episodes": 1000,
                       "steps": 500,
                       "timeout": 300,
                       "max_score": 16,
                       "generate": True}
scenarios["large"] = {"machines": 18,
                      "services": 6,
                      "restrictiveness": 3,
                      "episodes": 1000,
                      "steps": 500,
                      "timeout": 300,
                      "max_score": 15,
                      "generate": True}
scenarios["huge"] = {"machines": 38,
                     "services": 10,
                     "restrictiveness": 3,
                     "episodes": 1000,
                     "steps": 500,
                     "timeout": 300,
                     "max_score": 14,
                     "generate": True}
scenarios["multi"] = {"file": "/home/jonathon/Documents/Uni/COMP6801/CyberAttackSimulator/network_attack_simulator/configs/multi_site.yaml", # noqa
                      "episodes": 1000000,
                      "steps": 500,
                      "timeout": 120,
                      "max_score": 17,
                      "generate": False}
scenarios["single"] = {"file": "/home/jonathon/Documents/Uni/COMP6801/CyberAttackSimulator/network_attack_simulator/configs/single_site.yaml",  # noqa
                       "episodes": 1000000,
                       "steps": 500,
                       "timeout": 120,
                       "max_score": 17,
                       "generate": False}
scenarios["standard"] = {"file": "/home/jonathon/Documents/Uni/COMP6801/CyberAttackSimulator/network_attack_simulator/configs/standard.yaml",   # noqa
                         "episodes": 1000000,
                         "steps": 500,
                         "timeout": 120,
                         "max_score": 17,
                         "generate": False}


# Experiment agents
agents = OrderedDict()
agents["td_egreedy"] = {
    "tiny": {"type": "egreedy", "alpha": 0.1, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "small": {"type": "egreedy", "alpha": 0.1, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "medium": {"type": "egreedy", "alpha": 0.1, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "large": {"type": "egreedy", "alpha": 0.1, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "huge": {"type": "egreedy", "alpha": 0.1, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "default": {"type": "egreedy", "alpha": 0.1, "gamma": 0.99, "epsilon_decay_lambda": 0.0001}
    }
agents["td_ucb"] = {
    "tiny": {"type": "UCB", "alpha": 0.1, "gamma": 0.99, "c": 0.5},
    "small": {"type": "UCB", "alpha": 0.1, "gamma": 0.99, "c": 0.5},
    "medium": {"type": "UCB", "alpha": 0.1, "gamma": 0.99, "c": 0.5},
    "large": {"type": "UCB", "alpha": 0.1, "gamma": 0.99, "c": 0.5},
    "huge": {"type": "UCB", "alpha": 0.1, "gamma": 0.99, "c": 0.5},
    "default": {"type": "UCB", "alpha": 0.1, "gamma": 0.99, "c": 0.5},
    }
agents["dqn"] = {
    "tiny": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "small": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "medium": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "large": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "huge": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.0001},
    "default": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.0001}
     }
agents["random"] = {}


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


def get_scenario_env(scenario_name, seed=1):
    if not is_valid_scenario(scenario_name, verbose=True):
        return None
    scenario_params = scenarios[scenario_name]
    if scenario_params["generate"]:
        M = scenario_params["machines"]
        S = scenario_params["services"]
        rve = scenario_params["restrictiveness"]
        env = Cyber.from_params(M, S,
                                r_sensitive=R_SENS,  r_user=R_USR,
                                exploit_cost=COST_EXP, scan_cost=COST_SCAN,
                                restrictiveness=rve, exploit_probs=EXPLOIT_PROB, seed=seed)
    else:
        env = Cyber.from_file(scenario_params["file"], scan_cost=COST_SCAN, seed=seed)
    return env, scenario_params


def get_scenario_max(scenario_name):
    if is_valid_scenario(scenario_name, verbose=True):
        return scenarios[scenario_name]["max_score"]
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
    if not scenario_name == "default" and not is_valid_scenario(scenario_name, verbose=True):
        return None

    if agent_name == "random":
        return RandomAgent()

    if scenario_name not in agents[agent_name].keys():
        scenario_name = "default"

    agent_params = agents[agent_name][scenario_name]
    if agent_name == "dqn":
        # only import when necessary
        from network_attack_simulator.agents.dqn import DQNAgent
        state_size = env.get_state_size()
        num_actions = env.get_num_actions()
        print("State size=", state_size)
        print("Num actions=", num_actions)
        return DQNAgent(state_size, num_actions, **agent_params)
    return QLearningAgent(**agent_params)


def get_agent_label(agent_name):
    if not is_valid_agent(agent_name, verbose=True):
        return None

    if agent_name == "dqn":
        return "Deep Q-learning"
    elif agent_name == "td_egreedy":
        return "Tabular Q-learning e-greedy"
    elif agent_name == "td_ucb":
        return "Tabular Q-learning UCB"
    elif agent_name == "random":
        return "Random"
    return None
