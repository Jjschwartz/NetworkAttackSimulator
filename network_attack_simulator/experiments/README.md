# Experiments for Cyber Attack Simulator

This module contains some scripts for running and visualizing experiments on the NetworkAttackSimulator (NAS) using different Reinforement Learning (RL) algorithms. Some details on experiments run for RL agents on the NAS.

## Measuring Performance

For each RL agent the aim of the experiments is to answer:
1. Can it solve the problem?
    1. i.e gain access to all the sensitive documents
2. how optimal is it's solution?
    1. i.e. does it solve the problem with minimal number of actions
1. How does the RL alforithm scale?

#### Some definitions:

**Solution** - the learned policy which dictates what action to take for each state.

**Solved** - a scenario is considered solved by a given solution, if following the solution leads to gaining access to all the sensitive documents in the network within the set limit of number of steps.

**Optimal** - a solution is considered optimal if it solves a scenario in roughly the minimum number of actions, where the minumum number of actions is dictated by the max depth of a sensitive machine.

Since there are two sensitive machines, one in the sensitive subnet with depth 1 and one in a leaf node of the user subnet the minimum possible number of moves is actually:

    minumum moves = max depth of sensitive machine + 1

However, the actual number of moves taken will typically be more than that due to non-deterministic actions, so the optimal solution will generally solve the sceneario with slightly more than the minumum moves.

For our experiments we use total episode reward as a proxy for optimallity, with the optimal reward being:

    optimal reward = total reward possible - (action cost * minimum moves)

Where, total reward possible is equal to the sum of the rewards gained from successfully compromising the two sensitive machines on the network.

#### Measuring how long it takes to generate a solution:

Since it is unreliable to try and pinpoint at which episode the RL agent produced it's best solution we will be using time required to train for a set number of episodes as a proxy for measuring how long it takes to generate a solution.

Measuring it in this way is useful since the time required for each episode is dependent on the number of timesteps required to solve it, if an agent finds a more optimal solution earlier it will finish training sooner since subsequent episodes will be much shorter. Additionally, using this measure also allows for comparisons between RL methods in terms of how long each takes for each epidode.

#### Performance metrics:

Following are the performance metrics recoreded for each experiment:

1. Reward per episode
2. Timesteps per episode
3. Total training time

## The Test Scenarios

Here I describe the test scenarios used for measuring and comparing the performance of the different RL Agents.

Each scenario is defined by a number of environment parameters, specifically:
- Number of machines
- Number of services
- Exploit probability
- Uniform (distribution of machine configurations)
- Restrictiveness

It's also possible to adjust other parameters such as action costs, rewards and parameters that control machine configuration distribution, however for these experiments these are not important since they do not affect the environment that much.

For these experiments I will set some parameters to the same value for all trials, specifically:
- exploit probability = "mixed"
- uniform = false

I chose these values since they better reflect real world exploit probabilities (i.e. based on CVNN scores of top 10 exploits of 2017) and machine configurations on a network (i.e. machine configuration distribution is correllated within a network as opposed to uniform).

The values for the other environment parameters for each scenario are chosen to give a good range of problem sizes. Specifically, the number of machines is chosen mostly to be the minimum number of machines for there to be another subnet in the network (in the case of small to medium network) or for max depth of rewarded machine to increment by 1 (for all others).


## The RL Agents

For this experiment we want to determine which RL approaches work best for this environment in terms of the outlined performance metrics. To this end the agents we will be comparing are:

1. TD Q-learning agent using epsilon greedy action selection
2. TD Q-learning agent using Upper-confidence bound action selection
2. Deep Q-network agent using epsilon-greedy action selection
