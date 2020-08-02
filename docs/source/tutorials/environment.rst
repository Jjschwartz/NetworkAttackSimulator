.. _`env_tute`:

Interacting with NASim Environment
==================================

Assuming you are comfortable loading an environment from a scenario (see :ref:`loading_tute` or :ref:`gym_load_tute`), then interacting with a NASim Environment is very easy and follows the same interface as `OpenAI gym <https://github.com/openai/gym>`_.


Starting the environment
------------------------

First thing is simply loading the environment::

  import nasim
  # load my environment in the desired way (make_benchmark, load, generate)
  env = nasim.make_benchmark("tiny")

  # or using gym
  import gym
  env. = gym.make("nasim:Tiny-PO-v0")


Here we are using the default environment parameters: ``fully_obs=False``, ``flat_actions=True``, and ``flat_obs=True``.

The number of actions can be retrieved from the environment ``action_space`` attribute as follows::

  # When flat_actions=True
  num_actions = env.action_space.n

  # When flat_actions=False
  nvec_actions = env.action_space.nvec


The shape of the observations can be retrieved from the environment ``observation_space`` attribute as follows::

  obs_shape = env.observation_space.shape



Getting the initial observation and resetting the environment
-------------------------------------------------------------

To reset the environment and get the initial observation, use the ``reset()`` function::

  o = env.reset()


Performing a single step
------------------------

A step in the environment can be taken using the ``step(action)`` function. Here ``action`` can take a few different forms depending on if using ``flat_actions=True`` or ``flat_actions=False``, for our example we can simply pass an integer with 0 <= action < N, which specifies the index of the action in the action space. The ``step`` function then returns a ``(Observation, float, bool, dict)`` tuple corresponding to observation, reward, done, auxiliary info, respectively::

  action = # integer in range [0, env.action_space.n]
  o, r, done, info = env.step(action)


if ``done=True`` then the goal has been reached, and the episode is over. It is then recommended to stop or reset the environment, otherwise theres no gaurantee of what will happen.


An example agent
----------------

Some example agents are provided in the ``nasim/agents`` directory. Here is a quick example of a hypothetical agent interacting with the environment::

  import nasim

  env = nasim.make_benchmark("tiny")

  agent = AnAgent(...)

  o = env.reset()
  total_reward = 0
  done = False
  while not done:
      a = agent.choose_action(o)
      o, r, done, _ = env.step(a)
      total_reward += r

  print("Done")
  print("Total reward =", total_reward)


It's as simple as that.
