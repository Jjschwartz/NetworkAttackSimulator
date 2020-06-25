.. _`env_tute`:

Interacting with NASim Environment
==================================

Assuming you understand scenarios and are comfortable loading an environment from a scenario (see :ref:`scenarios_tute` if not), then interacting with a NASim Environment is very easy and follows the same interface as `OpenAI gym <https://github.com/openai/gym>`_.


Starting the environment
------------------------

First things is loading the environment::

  import nasim
  # load my environment in the desired way (make_benchmark, load, generate)
  env = nasim.make_benchmark("tiny")

Here we are using the default environment parameters: ``fully_obs=False``, ``flat_actions=True``, and ``flat_obs=True``.

Using ``flat_actions=True`` means our action space is made up of N discrete actions, where N is based on the number of hosts in the network and the number of exploits and scans available. For our example there are 3 hosts, 1 exploit and 3 scans (OS, Service, and Subnet), for a total of 3 * (1 + 3) = 12 actions. When using ``flat_actions=True`` the number of actions can be retrieved from the environment ``action_space`` attribute as follows::

  num_actions = env.action_space.n

Using ``flat_obs=True`` means the observations returned will be a 1D vector. For explanation of the features of this vector see :ref:`observation`. The shape of the observations can be retrieved from the environment ``observation_space`` attribute as follows::

  obs_shape = env.observation_space.shape

The ``fully_obs=False`` means the environment is partially observed, so the agent starts with no knowledge of the location, configuration and value of every host on the network and recieves only observations of features of the directly related to the action performed at each step. Whether the environment is fully or partially observable has no effect on the size and shape of the action and observation spaces or how the agent interacts with the environment. It will have significant implications for the algorithms used to solve the environment, but that is beyond the scope of this tutorial.

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
