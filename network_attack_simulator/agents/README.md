# Automated Pentesting Reinforcement Learning agents

This module contains a number of implementation of different Reinforcement Learning (RL) algorithms that can be used with the NetworkAttackSimulator.

## Implemented agents

So far the implemented agents include:
- Deep Q-learning using experience replay and a seperate target network
- Tabular Q-learning with epsilon-greedy or UCB action selection
- A random agent that selects actions uniformly at random from the action space


## Using the agents

To use an simply import the agent class into your program, feed the NetworkAttackSimulator instance into its train method and then let the agent work it's magic. Once training is complete you can use the output results to look at the episodic learning curves or you can check out the learned policy by running an episode on the environment and use the environment to render it.

```
from network_attack_simulator.agents.q_learning import QLearningAgent

env = NetworkAttackSimulator.from_params(M, S)
agent = QLearningAgent(**agent_params)
result = agent.train(env, num_episodes, max_steps, **train_args)
gen_episode = agent.generate_episode(env, max_steps)
env.render_episode(gen_episode)
```

## Resources

1. Reinforcement Learning: An Introducton
  - The bible for all things RL
  - http://incompleteideas.net/book/bookdraft2017nov5.pdf
1. Human level control through deep RL
 - The original paper for using DQN with experience replay and seperate target network
 - https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/
1. Awesome tutorial on implementing a DQN
  - https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
