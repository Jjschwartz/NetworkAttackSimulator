"""
This module contains the functionallity for generating the input file for a POMDP solver from a
given environment configuration

"""
from cyber_attack_simulator.envs.environment import CyberAttackSimulatorEnv
from collections import OrderedDict
from copy import deepcopy

# state output formatting
COMPROMISED = "C"
NOT_COMPROMISED = "NotC"
FAILURE = "Fail"
SUCCESS = "Success"


def generate_pomdp_config(config_path, output_name="pomdp.pomdp", discount=0.95):
    """
    Generate a pompdp from a given config file
    """
    print("Generating POMDP")
    print(">> Loading environment")

    # config params
    num_machines = 3
    num_services = 2
    exploit_prob = 1.0
    uniform = False
    restrictiveness = 1

    print("\tnumber of machines =", num_machines)
    print("\tnumber of services =", num_services)
    print("\texploit success probability =", exploit_prob)
    print("\tuniform =", uniform)
    print("\tfirewall restrictiveness =", restrictiveness)

    env = CyberAttackSimulatorEnv.from_params(num_machines, num_services,
                                              exploit_probs=exploit_prob,
                                              uniform=uniform,
                                              restrictiveness=restrictiveness)

    # 1 open file for writing
    fout = open(output_name, "w")

    # 2 discount and value
    print(">> Writing discount and value")
    fout.write("\ndiscount: {0}\n".format(discount))
    fout.write("values: reward\n")

    # 3 state space
    print(">> Generating state space")
    state_space = generate_state_space(env)
    print("\tState space size = {0}".format(len(state_space)))
    print(">> Writing state space")
    write_state_space(state_space, fout)

    # 4 action space
    print(">> Loading action space")
    action_space = env.action_space
    print("\tAction space size = {0}".format(len(action_space)))
    print(">> Writing action space")
    write_action_space(action_space, fout)

    # 6 observation space
    print(">> Generating observation space")
    obs_space = generate_obs_space(env)
    print("\tObservation space size = {0}".format(len(obs_space)))
    print(">> Writing observation space")
    write_obs_space(obs_space, fout)

    # 8 initial belief distribution
    print(">> Generating initial belief")
    init_belief = generate_init_belief(state_space)
    print(">>Writing initial belief")
    write_init_belief(init_belief, fout)
    fout.write("\n")

    # 9 observation function
    print(">> Generating observation function")
    obs_function = generate_obs_function(env, state_space, action_space)
    print("\tObservation function size = {0}".format(len(obs_function)))
    print(">> Writing observation function")
    write_obs_functions(obs_function, fout)
    fout.write("\n")

    # 10 transition function
    print(">> Generating transition function")
    trans_function = generate_transition_function(env, state_space, action_space)
    print("\tTransition function size = {0}".format(len(trans_function)))
    print(">> Writing transition function")
    write_transition_functions(trans_function, fout)
    fout.write("\n")

    # 11 reward function
    print(">> Generating reward function")
    rew_function = generate_reward_function(env, state_space, action_space)
    print("\tReward function size = {0}".format(len(rew_function)))
    print(">> Writing reward function")
    write_reward_function(rew_function, fout)
    fout.write("\n")

    fout.close()


def generate_state_space(env):
    """
    Generate state space from environment
    """
    nS = env.num_services
    address_space = env.address_space

    # get all possible combinations of compromised and service configurations
    service_configs = generate_service_configs(nS)[:-1]
    machine_states = generate_machine_states(service_configs)
    network_states = generate_network_states(address_space, machine_states)

    state_space = []
    for s in network_states:
        state_space.append(POMDPState(s))
    return state_space


def generate_service_configs(n):
    """
    Recursively generate list of all possible configurations of n services, where:

    N.B First permutation in list is always the all True permutation and final
    permutation in list is always the all False permutation.

    perms[1] = [True, True, ..., True]
    perms[-1] = [False, False, ..., False]

    Arguments:
        int n : number of services

    Returns:
        list[list[bool]] perms : list of list of bools of all possible configurations of n services
    """
    # base cases
    if n <= 0:
        return []
    if n == 1:
        return [[True], [False]]

    perms = []
    for p in generate_service_configs(n - 1):
        perms.append([True] + p)
        perms.append([False] + p)
    return perms


def generate_machine_states(service_configs):
    """
    Generate list of all possible configurations of service and compromised status for a single
    machine.

    Config = (compromised, [s0, s1, ..., sn])

    where:
        compromised = True/False
        sn = True/False
    """
    machine_states = []
    for cfg in service_configs:
        machine_states.append((True, cfg))
        machine_states.append((False, cfg))
    return machine_states


def generate_network_states(address_space, machine_states):
    """
    Recursively generate all possible network states from set of possible machine states for each
    machine on the network

    Network state = [((0,0), (compromised, [s0, ..., sn])), ... ]
    """
    network_states = []
    m = address_space[0]
    if len(address_space) == 1:
        for s in machine_states:
            network_states.append([(m, s)])
        return network_states

    for subState in generate_network_states(address_space[1:], machine_states):
        for s in machine_states:
            network_states.append([(m, s)] + subState)
    return network_states


def write_state_space(state_space, fout):
    """
    Write state space to file for given environment
    """
    fout.write("states: ")
    for s in state_space:
        fout.write(str(s))
        fout.write(" ")
    fout.write("\n")


def write_action_space(action_space, fout):
    """
    Write action space to file
    """
    fout.write("actions: ")
    for a in action_space:
        fout.write(format_action(a))
        fout.write(" ")
    fout.write("\n")


def format_action(a):
    """
    Convert Action object into string representation for pompdp file

    e.g scan machine (0, 0)
        0.0scan

    e.g. exploit service 1 on machine (1, 0)
        1.0exp1
    """
    address = "a{0}{1}".format(a.target[0], a.target[1])
    if a.is_scan():
        return address + "scan"
    else:
        return "{0}exp{1}".format(address, a.service)


def generate_obs_space(env):
    """
    Generate the observation space for the environment
    """
    obs_space = []
    nS = env.num_services
    service_configs = generate_service_obs(nS)[:-1]
    # if action fails
    obs_space.append(FAILURE)
    for cfg in service_configs:
        # if action was successful exploit
        obs_space.append(SUCCESS + COMPROMISED + cfg)
        # if action was successful scan
        obs_space.append(SUCCESS + NOT_COMPROMISED + cfg)
    return obs_space


def generate_service_obs(n):
    """
    Recursively generate list of all possible configurations of n services.
    Same as generate_service_configs except replace True/False with "yes"/"no"
    """
    # base cases
    if n <= 0:
        return []
    if n == 1:
        return ["yes", "no"]

    perms = []
    for p in generate_service_obs(n - 1):
        perms.append("yes" + p)
        perms.append("no" + p)
    return perms


def write_obs_space(obs_space, fout):
    """
    Write observation space to file
    """
    fout.write("observations: ")
    for o in obs_space:
        fout.write(o)
        fout.write(" ")
    fout.write("\n")


def generate_init_belief(state_space):
    """
    Generate the initial belief distribution over states

    Initially, there is uniform belief for all states where no machine in network is compromised
    """
    num_valid_init_states = 0
    for s in state_space:
        if not s.has_compromised_machine():
            num_valid_init_states += 1

    init_prob = 1 / num_valid_init_states
    init_belief = []
    for s in state_space:
        if not s.has_compromised_machine():
            init_belief.append(init_prob)
        else:
            init_belief.append(0)
    return init_belief


def write_init_belief(init_belief, fout):
    """
    Write initial belief state to file
    """
    fout.write("start: ")
    for b in init_belief:
        fout.write(str(b))
        fout.write(" ")
    fout.write("\n")


def generate_obs_function(env, state_space, action_space):
    """
    Generate the observation probabilities for each action and state
    """
    obs_function = []
    for a in action_space:
        a_string = format_action(a)
        for s in state_space:
            s_string = str(s)
            results = observation_function(env, a, s)
            for obs, prob in results:
                obs_function.append((a_string, s_string, obs, prob))
    return obs_function


def observation_function(env, action, state):
    """
    Given an action and state returns the set of possible observations and observation
    probabilities
    """
    result = []
    if action_failed(env, action, state):
        result.append((FAILURE, 1.0))
    else:
        exploit_prob = action.prob
        # 1. success with given probability
        success_obs = SUCCESS
        success_obs += NOT_COMPROMISED if action.is_scan() else COMPROMISED
        success_obs += state.get_machine_service_obs(action.target)
        result.append((success_obs, exploit_prob))
        if exploit_prob < 1.0:
            result.append((FAILURE, 1 - exploit_prob))
    return result


def action_failed(env, action, state):
    """
    Return whether an action failed against a target machine for a given network state

    Failure occurs when:
        1) target is not reachable
        2) action is an exploit and target service is not present on target machine
    """
    # 1. if action.target is not reachable
    if not target_reachable(env, action.target, state):
        return True
    # 2. if firewall prevents action.service traffic
    if not action_traffic_permitted(env, state, action):
        return True
    # 3. if action is an exploit and target doesn't have service running
    if not action.is_scan() and not state.machine_has_service(action.target, action.service):
        return True
    return False


def target_reachable(env, target, state):
    """
    Return True if target is reachable in network.

    Target is reachable iff:
        1) it's on an exposed subnet
        2) it's on a compromised subnet
        3) it's on a subnet connected to a compromised subnet
    """
    network = env.network
    if network.subnet_exposed(target[0]):
        return True
    for m in env.address_space:
        if state.machine_compromised(m):
            # machine compromised
            if network.subnets_connected(m[0], target[0]):
                return True
    return False


def action_traffic_permitted(env, state, action):
    """
    Return True if action is permitted in terms of firewall traffic
    """
    if action.is_scan():
        return True
    network = env.network
    service = action.service
    dest = action.target[0]
    # add 0 since 0 = internet
    compromised_subnets = set([0])
    for m in env.address_space:
        if state.machine_compromised(m):
            compromised_subnets.add(m[0])
    for src in compromised_subnets:
        if network.traffic_permitted(src, dest, service):
            return True
    return False


def write_obs_functions(obs_function, fout):
    """
    Write observation function to file
    """
    for o in obs_function:
        fout.write("O:{0} : {1} : {2} {3:.3f}\n".format(o[0], o[1], o[2], o[3]))


def generate_transition_function(env, state_space, action_space):
    """
    Generate the next state and transition probabilities for each action and state
    """
    trans_function = []
    for a in action_space:
        a_string = format_action(a)
        for s in state_space:
            s = s
            results = transition_function(env, a, s)
            for new_s, prob in results:
                trans_function.append((a_string, str(s), str(new_s), prob))
    return trans_function


def transition_function(env, action, state):
    """
    Generate next state/s and probabilities for performing action in given state
    """
    result = []
    if state.machine_compromised(action.target):
        result.append((str(state), 1.0))
    elif action_failed(env, action, state) or action.is_scan():
        result.append((str(state), 1.0))
    else:
        successor = state.get_successor_state(action.target)
        exploit_prob = action.prob
        # 1. success with given probability
        result.append((str(successor), exploit_prob))
        if exploit_prob < 1.0:
            result.append((str(state), 1 - exploit_prob))
    return result


def write_transition_functions(trans_function, fout):
    """
    Write transition function to file
    """
    for t in trans_function:
        fout.write("T:{0} : {1} : {2} {3:.3f}\n".format(t[0], t[1], t[2], t[3]))


def generate_reward_function(env, state_space, action_space):
    """
    Generate the rewards for all action and state pairs
    """
    rew_function = []
    for a in action_space:
        a_string = format_action(a)
        for start_state in state_space:
            result = reward_function(env, a, start_state)
            for end_state, r in result:
                rew_function.append((a_string, str(start_state), end_state, r))
    return rew_function


def reward_function(env, action, state):
    """
    Return the reward for taking action in given state
    """
    action_cost = action.cost
    result = []
    if state.machine_compromised(action.target):
        result.append(("*", 0.0))
    elif action_failed(env, action, state) or action.is_scan():
        result.append(("*", -action_cost))
    else:
        successor = state.get_successor_state(action.target)
        target_value = env.network.get_machine_value(action.target)
        exploit_prob = action.prob
        # 1. success with given probability
        result.append((str(successor), target_value - action_cost))
        if exploit_prob < 1.0:
            result.append((str(state), - action_cost))
    return result


def write_reward_function(reward_function, fout):
    """
    Write reward function to file
    """
    for r in reward_function:
        fout.write("R:{0} : {1} : {2} : * {3}\n".format(r[0], r[1], r[2], r[3]))


class POMDPState(object):
    """
    A state of the environment network.

    Defined by dictionary:
        {(0, 0): (compromised, services), (0,1): ... }

    Where:
        - compromised = True/False
        - services = list of bools of present/absent
    """

    def __init__(self, network_state):
        self.network_state = OrderedDict(network_state)

    def has_compromised_machine(self):
        for s in self.network_state.values():
            if s[0]:
                return True
        return False

    def machine_compromised(self, m):
        return self.network_state[m][0]

    def machine_has_service(self, target, service):
        return self.network_state[target][1][service]

    def get_machine_service_obs(self, target):
        output = ""
        for s in self.network_state[target][1]:
            if s:
                output += "yes"
            else:
                output += "no"
        return output

    def get_successor_state(self, target):
        """
        Return state following a successful exploit against target
        """
        new_network_state = deepcopy(self.network_state)
        target_state = list(new_network_state[target])
        target_state[0] = True
        new_network_state[target] = tuple(target_state)
        return POMDPState(new_network_state)

    def is_goal(self, sensitive_machines):
        """
        Return whether this state is the goal state
        """
        for sensitive_m in sensitive_machines:
            if not self.machine_compromised(sensitive_m):
                # at least one sensitive machine not compromised
                return False
        return True

    def __str__(self):
        output = "s"
        for m, s in self.network_state.items():
            comp = s[0]
            services = s[1]
            output += "{0}{1}".format(m[0], m[1])
            if comp:
                output += COMPROMISED
            else:
                output += NOT_COMPROMISED
            for s in services:
                if s:
                    output += "yes"
                else:
                    output += "no"
        return output
