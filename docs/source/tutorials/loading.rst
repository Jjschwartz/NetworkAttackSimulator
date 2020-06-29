.. _`loading_tute`:

Starting a NASim Environment
==============================

Interaction with NASim is done primarily via the :class:`~nasim.env.environment.NASimEnv` class, which handles a simulated network environment as defined by the chosen scenario.

For initialization the NASimEnv class takes a scenario definition and three optional arguments:

- ``fully_obs`` : The observability mode of environment, if True then uses fully observable mode, otherwise is partially observable (default=False)
- ``flat_actions`` : If true then uses a flat action space, otherwise will uses a parameterised action space (default=True).
- ``flat_obs`` :  If true then uses a 1D observation space, otherwise uses a 2D observation space (default=True)

The scenario defines the network properties and the pen-tester specific information (e.g. exploits available, etc). For this tutorial we are going to stick to how to start a new environment, details on scenarios is covered in :ref:`scenarios_tute`.


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
