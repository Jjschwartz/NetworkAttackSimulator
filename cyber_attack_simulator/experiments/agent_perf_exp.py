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
from cyber_attack_simulator.experiments.experiment_util import get_scenario_env
from cyber_attack_simulator.experiments.experiment_util import get_agent

# To control progress message printing for individual episodes within runs
VERBOSE = False

# Experiment agents to run
agent_list = []
agent_list.append("td_egreedy")
agent_list.append("td_ucb")
agent_list.append("dqn")
agent_list.append("random")

# Experiment constants
RUNS = 1      # number of training runs to do
EVAL_WINDOW = 100      # number of episodes to evaluate policy over
EVAL_RUNS = 10    # number of evaluation runs
EVAL_EPSILON = 0.05     # epsilon for evaluation e-greedy policy

# Environment constants
RVE = 3
UNIFORM = False
EXPLOIT_PROB = 0.7
R_SENS = R_USR = 10
COST_EXP = COST_SCAN = 1

# Experiment parameters
SCALING_RUNS = 7
MACHINE_MIN = 3
MACHINE_MAX = 43
MACHINE_INTERVAL = 5
SERVICE_MIN = 5
SERVICE_MAX = 5
SERVICE_INTERVAL = 1
MAX_EPISODES = 1000000
MAX_STEPS = 500
EVAL_RUNS = 10
TIMEOUT = 120
VERBOSE = False
START_SEED = 3


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


def run_experiment(scenario, agent_name, result_file, eval_file):

    scenario_params = get_scenario(scenario)
    num_episodes = scenario_params["episodes"]
    max_steps = scenario_params["steps"]
    timeout = scenario_params["timeout"]

    print("\nRunning experiment: Scenario={}, agent={}".format(scenario, agent_name))

    for t in range(RUNS):
        env, _ = get_scenario_env(scenario, t)
        agent = get_agent(agent_name, scenario, env)
        if agent_name != "random":
            ep_tsteps, ep_rews, ep_times = agent.train(env, num_episodes, max_steps, timeout,
                                                       VERBOSE)
            path_found = is_path_found(ep_tsteps, max_steps)
            exp_reward = get_expected_rewards(ep_rews)
            training_time = sum(ep_times)
            print("\tTraining run {0} - path_found={1} - exp_reward={2} - train_time={3:.2f}"
                  .format(t, path_found, exp_reward, training_time))
            write_results_eps(result_file, scenario, agent_name, t, ep_tsteps, ep_rews, ep_times)

        run_evaluation(agent, env, scenario, agent_name, max_steps, t, eval_file)


def run_scaling_experiment(agent_name, eval_file):

    print("\nRunning scaling analysis for agent: \n\t {0}".format(str(agent_name)))

    for m in range(MACHINE_MIN, MACHINE_MAX + 1, MACHINE_INTERVAL):
        for s in range(SERVICE_MIN, SERVICE_MAX + 1, SERVICE_INTERVAL):
            print("\n>> Machines={0} Services={1}".format(m, s))
            for t in range(SCALING_RUNS):
                print("\tRun {0} of {1} seed={2}".format(t+1, SCALING_RUNS, t+START_SEED))
                env = Cyber.from_params(m, s,
                                        r_sensitive=R_SENS,  r_user=R_USR,
                                        exploit_cost=COST_EXP, scan_cost=COST_SCAN,
                                        restrictiveness=RVE, exploit_probs=EXPLOIT_PROB,
                                        seed=t+START_SEED)
                agent = get_agent(agent_name, "default", env)
                if agent_name != "random":
                    ep_tsteps, ep_rews, ep_times = agent.train(env, MAX_EPISODES, MAX_STEPS,
                                                               TIMEOUT, VERBOSE)
                    path_found = is_path_found(ep_tsteps, MAX_STEPS)
                    exp_reward = get_expected_rewards(ep_rews)
                    training_time = sum(ep_times)
                    print("\tTraining run {} - path_found={} - exp_reward={} - train_time={:.2f}"
                          .format(t, path_found, exp_reward, training_time))

                run_evaluation(agent, env, (m, s), agent_name, MAX_STEPS, t, eval_file)


def run_evaluation(agent, env, scenario, agent_name, max_steps, run, eval_file):
    """
    Evaluate a trained agent
    """
    tsteps_sum = 0
    reward_sum = 0
    solved_sum = 0
    for erun in range(EVAL_RUNS):
        tsteps, reward, solved = agent.evaluate_agent(env, max_steps, EVAL_EPSILON)
        write_results_eval(eval_file, scenario, agent_name, run, erun, tsteps, reward, solved)
        tsteps_sum += tsteps
        reward_sum += reward
        solved_sum += int(solved)
    print("\t\tEvaluation results - mean timesteps={:.2f} - mean reward={:.2f} - solved={:.2f}"
          .format(tsteps_sum / EVAL_RUNS, reward_sum / EVAL_RUNS, solved_sum / EVAL_RUNS))


def write_results_eps(result_file, scenario, agent, run, timesteps, rewards, times):
    """ Write results to file """
    for e in range(len(timesteps)):
        # scenario,agent,run,episode,timesteps,rewards,time
        result_file.write("{0},{1},{2},{3},{4},{5},{6:.8f}\n".format(scenario, agent, run, e,
                          timesteps[e], rewards[e], times[e]))


def write_results_eval(eval_file, scenario, agent, run, evalrun, timesteps, reward, solved):
    """ Write results to file """
    # scenario,agent,run,evalrun,timesteps,rewards, solved
    eval_file.write("{0},{1},{2},{3},{4},{5},{6}\n".format(scenario, agent, run, evalrun, timesteps,
                    reward, solved))


def main():

    if len(sys.argv) != 4:
        print("Usage: python experiment.py scenario outfile append")
        print("Where append is [1/0] for whether to append to file or not")
        return 1

    print("Welcome to your friendly experiment runner")

    scenario = sys.argv[1]
    result_file_name = sys.argv[2] + "_eps.csv"
    eval_file_name = sys.argv[2] + "_eval.csv"
    if sys.argv[3] == "1":
        print("Appending to", sys.argv[2])
        result_file = open(result_file_name, 'a+')
        eval_file = open(eval_file_name, "a+")
    else:
        print("Writing to new file", sys.argv[2])
        result_file = open(result_file_name, 'w')
        eval_file = open(eval_file_name, "w")
        # write header line
        result_file.write("scenario,agent,run,episode,timesteps,rewards,time\n")
        eval_file.write("scenario,agent,run,evalrun,timesteps,reward,solved\n")

    for agent_name in agent_list:
        if scenario == "scaling":
            run_scaling_experiment(agent_name, eval_file)
        else:
            run_experiment(scenario, agent_name, result_file, eval_file)

    result_file.close()


if __name__ == "__main__":
    main()
