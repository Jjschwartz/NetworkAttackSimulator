"""
Run Experimental scenarios

Gets results for multiple agents and scenarios
    - whether scenario was solved
    - reward vs episode
    - time vs episode
"""
import sys
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv as Cyber
from cyber_attack_simulator.experiments.experiment_util import get_scenario
from cyber_attack_simulator.experiments.experiment_util import get_agent

# To control progress message printing for individual episodes within runs
VERBOSE = True

# Experiment scenarios and agents to run
# scenarios_list = ["tiny"]
# scenarios_list = ["small"]
scenarios_list = ["medium"]
# agent_list = ["td_egreedy", "td_ucb", "dqn"]
# agent_list = ["td_egreedy", "td_ucb"]
agent_list = ["dqn"]


# Experiment constants
RUNS = 1      # number of training runs to do
EVAL_WINDOW = 100      # number of episodes to evaluate policy over

# Environment constants
UNIFORM = False
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

    scenario_params = get_scenario(scenario)
    M = scenario_params["machines"]
    S = scenario_params["services"]
    rve = scenario_params["restrictiveness"]
    num_episodes = scenario_params["episodes"]
    max_steps = scenario_params["steps"]
    timeout = scenario_params["timeout"]

    print("\nRunning experiment: Scenario={}, agent={}, M={}, S={}, R={}"
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

        print("\tTraining run {0} - path_found={1} - exp_reward={2} - train_time={3:.2f}\n"
              .format(t, path_found, exp_reward, training_time))


def write_results(result_file, scenario, agent, run, timesteps, rewards, times):
    """ Write results to file """

    for e in range(len(timesteps)):
        # scenario,agent,run,episode,timesteps,rewards,time
        result_file.write("{0},{1},{2},{3},{4},{5},{6:.8f}\n".format(scenario, agent, run, e,
                          timesteps[e], rewards[e], times[e]))


def main():

    if len(sys.argv) != 3:
        print("Usage: python experiment.py outfile append")
        print("Where append is [1/0] for whether to append to file or not")
        return 1

    print("Welcome to your friendly experiment runner")
    if sys.argv[2] == "1":
        print("Appending to", sys.argv[1])
        result_file = open(sys.argv[1], 'a+')
    else:
        print("Writing to new file", sys.argv[1])
        result_file = open(sys.argv[1], 'w')
        # write header line
        result_file.write("scenario,agent,run,episode,timesteps,rewards,time\n")

    for scenario in scenarios_list:
        for agent_name in agent_list:
            run_experiment(scenario, agent_name, result_file)

    result_file.close()


if __name__ == "__main__":
    main()
