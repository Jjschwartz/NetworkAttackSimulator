"""
Run Experimental scenarios
"""
import sys
from collections import OrderedDict
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv as Cyber
from cyber_attack_simulator.agents.q_learning import QLearningAgent
from cyber_attack_simulator.agents.dqn import DQNAgent

# To control progress message printing for individual episodes within runs
VERBOSE = True

# Experiment scenarios and agents to run
scenarios_list = ["tiny"]
agent_list = ["td_egreedy", "dqn"]


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
                      "episodes": 6000,
                      "steps": 600,
                      "timeout": 600}
scenarios["medium"] = {"machines": 13,
                       "services": 5,
                       "restrictiveness": 3,
                       "episodes": 16000,
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
    "tiny": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001},
    "small": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001},
    "medium": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001},
    "large": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001},
    "huge": {"type": "egreedy", "alpha": 0.1, "gamma": 0.9, "epsilon_decay_lambda": 0.001}
    }
agents["td_ucb"] = {
    "tiny": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0},
    "small": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0},
    "medium": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0},
    "large": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0},
    "huge": {"type": "UCB", "alpha": 0.1, "gamma": 0.9, "c": 1.0}
    }
agents["dqn"] = {
    "tiny": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001},
    "small": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001},
    "medium": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001},
    "large": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001},
    "huge": {"hidden_units": 256, "gamma": 0.99, "epsilon_decay_lambda": 0.001}
     }


# Experiment constants
RUNS = 3      # number of training runs to do
EVAL_WINDOW = 100      # number of episodes to evaluate policy over

# Environment constants
UNIFORM = True
EXPLOIT_PROB = "mixed"
R_SENS = R_USR = 10
COST_EXP = COST_SCAN = 1


def is_path_found(ep_tsteps, max_steps):
    """ Returns percentage of last EVAL_WINDOW episodes where the problem was solved as a
    probability """
    solved_count = 0
    for t in ep_tsteps[-EVAL_WINDOW:]:
        if t < max_steps:
            solved_count += 1
    return solved_count / EVAL_WINDOW


def get_expected_rewards(ep_rewards):
    """ Returns the average reward over the last EVAL_WINDOW episodes """
    return sum(ep_rewards[-EVAL_WINDOW:]) / EVAL_WINDOW


def run_experiment(scenario, agent_name, result_file):

    scenario_params = scenarios[scenario]
    M = scenario_params["machines"]
    S = scenario_params["services"]
    rve = scenario_params["restrictiveness"]
    num_episodes = scenario_params["episodes"]
    max_steps = scenario_params["steps"]
    timeout = scenario_params["timeout"]

    print("Running experiment: Scenario={}, agent={}, M={}, S={}, R={}"
          .format(scenario, agent_name, M, S, rve))

    for t in range(RUNS):
        env = Cyber.from_params(M, S,
                                r_sensitive=R_SENS,  r_user=R_USR,
                                exploit_cost=COST_EXP, scan_cost=COST_SCAN,
                                restrictiveness=rve, exploit_probs=EXPLOIT_PROB, seed=t)
        agent = get_agent(agent_name, scenario, env)
        ep_tsteps, ep_rews, ep_times = agent.train(env, num_episodes, max_steps, timeout, VERBOSE)

        path_found = is_path_found(ep_tsteps, max_steps)
        exp_reward = get_expected_rewards(ep_rews)
        training_time = sum(ep_times)

        write_results(result_file, scenario, agent_name, t, ep_tsteps, ep_rews, ep_times)

        print("\tTraining run {0} - path_found={1} - exp_reward={2} - train_time={3:.2f}"
              .format(t, path_found, exp_reward, training_time))


def write_results(result_file, scenario, agent, run, timesteps, rewards, times):
    """ Write results to file """

    for e in range(len(timesteps)):
        # scenario,agent,run,episode,timesteps,rewards,time
        result_file.write("{0},{1},{2},{3},{4},{5},{6:2f}\n".format(scenario, agent, run, e,
                          timesteps[e], rewards[e], times[e]))


def get_agent(agent_name, scenario, env):
    """ Returns a new agent instance """
    agent_params = agents[agent_name][scenario]
    if agent_name == "dqn":
        state_size = env.get_state_size()
        num_actions = env.get_num_actions()
        return DQNAgent(state_size, num_actions, **agent_params)
    return QLearningAgent(**agent_params)


def main():

    if len(sys.argv) != 2:
        print("Usage: python experiment.py outfile")
        return 1

    print("Welcome to your friendly experiment runner")
    result_file = open(sys.argv[1], 'w')

    # write header line
    result_file.write("scenario,agent,run,episode,timesteps,rewards,time\n")

    for scenario in scenarios_list:
        for agent_name in agent_list:
            run_experiment(scenario, agent_name, result_file)

    result_file.close()


if __name__ == "__main__":
    main()
