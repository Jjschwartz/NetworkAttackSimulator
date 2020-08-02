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

  ``ScenarioName-[PO-]vX``

Where:

- ``ScenarioName`` is the name of the benchmark scenario in Camel Casing
- ``[PO-]`` is optional and specifies the environment is in partially observable mode, if it is not included the environment is in fully observable mode.
- ``vX`` controls the action and observation space mode and where ``X`` can be ``0``, ``1``, ``2``, or ``3``:

  - ``v0``: flat actions and flat obs
  - ``v1``: flat actions and 2D obs
  - ``v2``: parameterised actions and flat obs
  - ``v3``: parameterised actions and 2D obs

For example, the 'tiny' benchmark scenario in partially observable mode with flat action-space and flat observation space has the name:

  ``Tiny-PO-v0``

Or the 'small-gen' benchmark scenario in fully observable mode with parametrised action-space and flat observation-space has the name:

  ``SmallGen-v2``


.. note:: See :ref:`env_params` for more explanation on the different modes.


Usage
-----

Now we understand the naming of environments, making a new environment using ``gym.make()`` is easy.

For example to create a new ``Tiny-PO-v0`` environment:

.. code:: python

   import gym
   env = gym.make("nasim:Tiny-PO-v0")
