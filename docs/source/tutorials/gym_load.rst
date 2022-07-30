.. _`gym_load_tute`:

Starting NASim using OpenAI gym
===============================

On startup NASim also registers each benchmark scenario as an `Open AI Gym <https://github.com/openai/gym>`_ , allowing NASim benchmark environments to be loaded using ``gym.make()``.

:ref:`all_benchmark_scenarios` can be loaded using ``gym.make()``.

.. note:: Custom scenarios must be loaded using the nasim library directly, see :ref:`loading_tute`.


Environment Naming
------------------

Unlike when starting an environment using the ``nasim`` library directly, where environment modes are specified as arguments to the ``nasim.make_benchmark()`` function, when using ``gym.make()`` the scenario and mode are specified in a single name.

When using ``gym.make()`` each environment has the following mode and naming convention:

  ``ScenarioName[PO][2D][VA]-vX``

Where:

- ``ScenarioName`` is the name of the benchmark scenario in Camel Casing
- ``[PO]`` is optional and specifies the environment is in partially observable mode, if it is not included the environment is in fully observable mode.
- ``[2D]`` is optional and specifies the environment is to return 2D observations, if it is not included the environment returns 1D observations.
- ``[VA]`` is optional and specifies the environment is to accept Vector actions (parametrised actions), if it is not included the environment expects integer (flat) actions.
- ``vX`` is the environment version. Currently (as of version ``0.10.0``) all environments are on ``v0``

For example, the 'tiny' benchmark scenario in partially observable mode with flat action-space and flat observation space has the name:

  ``TinyPO-v0``

Or the 'small-gen' benchmark scenario in fully observable mode with parametrised action-space and flat observation-space has the name:

  ``SmallGenVA-v0``


Or the 'medium-single-site' benchmark scenario in partially observable mode with parametrised action-space and 2D observation-space has the name:

  ``MediumSingleSitePO2DVA-v0``


.. note:: See :ref:`env_params` for more explanation on the different modes.


Usage
-----

Now we understand the naming of environments, making a new environment using ``gym.make()`` is easy.

For example to create a new ``TinyPO-v0`` environment:

.. code:: python

   import gym
   env = gym.make("nasim:TinyPO-v0")


.. note:: With the latest updates in ``v0.10.0`` and new Open AI gym API, you may need to specify ``new_step_api=True`` when calling the ``gym.make`` function if your implementation expects the new API. This is due to how the default wrappers that are created when calling ``gym.make`` handle things.
