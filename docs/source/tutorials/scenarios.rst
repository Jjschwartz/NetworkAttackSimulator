.. _`scenarios_tute`:

Understanding Scenarios
=======================

A scenario in NASim defines all the necessary properties for creating a network environment. Scenarios can be generated using the scenario generator or can be constructed manually using a .YAML file.

Each scenario definition can be broken down into two components: the network configuration and the pen-tester .

The network configuration is defined by a the following properties:

- subnets: the number and size of the subnets in the network
- topology: how the different subnets in the network are connected
- host_configurations: what OS and services are running on each host in the network
- firewall: which communication is prevented between subnets

The pen-tester is defined by:

- exploits: the set of exploits available to the pen-tester
- sensitive_hosts: the target hosts on the network and their value
- scan_costs: the cost of performing each type of scan (service, OS, and subnet)

The details of how each property of a scenario definition is represented is detailed below. However, if you would prefer to just get started, there are also a number of predefined scenarios available (see :ref:`make_existing` and :ref:`benchmark_scenarios` below).

.. _`loading_env`:

Loading an Environment from a Scenario
--------------------------------------

NASim Environments can be constructed from scenarios in three ways: making an existing scenario, loading from a .yaml file, and generating from parameters.

.. note:: Each of the methods described below also accept a `fully_obs` boolen argument that dictates the observability mode of the new environment, by default it is `True`. To make the environment partially observable (and closer to real life) set this to `False`.


.. _`make_existing`:

Making an existing scenario
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the easiest method for loading a new environment and closely matches the `OpenAI gym <https://github.com/openai/gym>`_ way of doing things. Loading an existing scenario is as easy as::

  import nasim
  env = nasim.make_benchmark("tiny")

And you are done.

You can also pass in a a random seed using the `seed` argument, which will have an effect when using a generated scenario.

.. note::  This method only works with the benchmark scenarios that come with NASim (for the full list see the :ref:`benchmark_scenarios` section below).


Loading a scenario from a YAML file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to load an existing or custom scenario defined in a YAML file, this is also very straight forward::

  import nasim
  env = nasim.load("path/to/scenario.yaml")

And once again, you are done (given your file is in a valid format)!


Generating a scenario
^^^^^^^^^^^^^^^^^^^^^

The final method for loading a new environment is to generate it using the NASim scenario generator. There are quite a number of parameters that can be used to control the what scenario is generated (for a full list see the `nasim.scenarios.ScenarioGenerator` class), but the two key parameters are the number of hosts in the network and the number of services running (which also controls number of exploits, unless otherwise specified).

To generate a new environment with 5 hosts running a possible 3 services::

  import nasim
  env = nasim.generate(5, 3)

And your done! If you want to pass in some other parameters (say the number of possible operating systems) these can be passed in as keyword arguments::

  env = nasim.generate(5, 3, num_os=3)


Once again, for a full list of available parameters refer to the `nasim.scenarios.ScenarioGenerator` documentation.
