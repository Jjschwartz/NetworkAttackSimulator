.. _agents_reference:

Agents Reference
================

This page provides a short summary of the agents that come with the NASim library.

Available Agents
----------------

The agent implementations that come with NASim include:

* **keyboard_agent.py**: An agent that is controlled by the user via terminal inputs.
* **random_agent.py**: A random agent that selects an action randomly from all available actions at each time step.
* **bruteforce_agent.py**: An agent that repeatedly cycles through all available actions in order.
* **ql_agent.py**: A Tabular, epsilod-greedy Q-Learning reinforcement learning agent.
* **ql_replay_agent.py**: A Tabular, epsilod-greedy Q-Learning reinforcement learning agent (same as above) that incorporates an experience replay.
* **dqn_agent.py**: A Deep Q-Network reinforcement learning agent using experience replay and a target Q-Network.


Running Agents
--------------

Each agent file defines a main function so can be run in python via the terminal, with the specific scenario and settings specified as command line arguments:


.. code-block:: bash

    cd nasim/agents
    # to run a different agent, simply replace .py file with desired file
    # to run a different scenario, simply replace 'tiny' with desired scenario
    python bruteforce_agent.py tiny

    # to get details on command line arguments available (e.g. hyperparameters for Q-Learning and DQN agents)
    python bruteforce_agent.py --help


A description and details of how to run each agent can be found at the top of each agent file.


Viewing Agent Policies
----------------------

For the DQN and Tabular Q-Learning agents you can optionally also view the final policies learned by the agents after training has finished:

.. code-block:: bash

    # simply include the --render_eval flag with the DQN and Q-Learning agents
    python ql_agent.py tiny --render_eval


This will show a single episode of the agent, displaying the actions the agent performs along with the observations and rewards the agent recieves.
