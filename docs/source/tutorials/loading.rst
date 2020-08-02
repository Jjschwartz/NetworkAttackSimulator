.. _`loading_tute`:

Starting a NASim Environment
============================

Interaction with NASim is done primarily via the :class:`~nasim.envs.environment.NASimEnv` class, which handles a simulated network environment as defined by the chosen scenario.

There are two ways to start a new environment: (i) via the nasim library directly, or (ii) using the `gym.make()` function of the Open AI gym library.

In this tutorial we will be covering the first method. For the second method check out :ref:`gym_load_tute`.


.. _`env_params`:

Environment Settings
--------------------

For initialization the NASimEnv class takes a scenario definition and three optional arguments.

The scenario defines the network properties and the pen-tester specific information (e.g. exploits available, etc). For this tutorial we are going to stick to how to start a new environment, details on scenarios is covered in :ref:`scenarios_tute`.

The three optional arguments control the environment modes:

- ``fully_obs`` : The observability mode of environment, if True then uses fully observable mode, otherwise is partially observable (default=False)
- ``flat_actions`` : If true then uses a flat action space, otherwise will uses a parameterised action space (default=True).
- ``flat_obs`` :  If true then uses a 1D observation space, otherwise uses a 2D observation space (default=True)


If using fully observable mode (``fully_obs=True``) then the entire state of the network and the attack is observed after each step. This is 'easy' mode and does not reflect the reality of pen-testing, but it is useful for getting started and sanity checking algorithms and environments. When using partially observable mode (``fully_obs=False``) the agent starts with no knowledge of the location, configuration and value of every host on the network and recieves only observations of features of the directly related to the action performed at each step. This is 'hard' mode and reflects the reality of pen-testing more accurately.

Whether the environment is fully or partially observable has no effect on the size and shape of the action and observation spaces or how the agent interacts with the environment. It will have significant implications for the algorithms used to solve the environment, but that is beyond the scope of this tutorial.

Using ``flat_actions=True`` means our action space is made up of N discrete actions, where N is based on the number of hosts in the network and the number of exploits and scans available. For our example there are 3 hosts, 1 exploit and 3 scans (OS, Service, and Subnet), for a total of 3 * (1 + 3) = 12 actions. If ``flat_actions=False`` then each action is a vector with each element of the vector specifying a parameter of the action. For more info see :ref:`actions`.

Using ``flat_obs=True`` means the observations returned will be a 1D vector. Otherwise if ``flat_obs=False`` observations will be a 2D matrix. For explanation of the features of this vector see :ref:`observation`.


.. _`loading_env`:

Loading an Environment from a Scenario
--------------------------------------

NASim Environments can be constructed from scenarios in three ways: making an existing scenario, loading from a .yaml file, and generating from parameters.

.. note:: Each of the methods described below also accept `fully_obs`, `flat_actions` and `flat_obs` boolen arguments.


.. _`make_existing`:

Making an existing scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the easiest method for loading a new environment and closely matches the `OpenAI gym <https://github.com/openai/gym>`_ way of doing things. Loading an existing scenario is as easy as::

  import nasim
  env = nasim.make_benchmark("tiny")

And you are done.

You can also pass in a a random seed using the `seed` argument, which will have an effect when using a generated scenario.

.. note::  This method only works with the benchmark scenarios that come with NASim (for the full list see the :ref:`benchmark_scenarios`).


Loading a scenario from a YAML file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to load an existing or custom scenario defined in a YAML file, this is also very straight forward::

  import nasim
  env = nasim.load("path/to/scenario.yaml")

And once again, you are done (given your file is in a valid format)!


Generating a scenario
^^^^^^^^^^^^^^^^^^^^^

The final method for loading a new environment is to generate it using the NASim scenario generator. There are quite a number of parameters that can be used to control the what scenario is generated (for a full list see the :ref:`scenario_generator` class), but the two key parameters are the number of hosts in the network and the number of services running (which also controls number of exploits, unless otherwise specified).

To generate a new environment with 5 hosts running a possible 3 services::

  import nasim
  env = nasim.generate(5, 3)

And your done! If you want to pass in some other parameters (say the number of possible operating systems) these can be passed in as keyword arguments::

  env = nasim.generate(5, 3, num_os=3)


Once again, for a full list of available parameters refer to the :ref:`scenario_generator` documentation.
