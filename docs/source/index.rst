Welcome to Network Attack Simulator's documentation!
====================================================

Network Attack Simulator (NASim) is a lightweight, high-level network attack simulator written in python. It is designed to be used for rapid testing of autonomous pen-testing agents using reinforcement learning and planning. It is a simulator by definition so does not replicate all details of attacking a real system but it instead aims to capture some of the more salient features of network pen-testing such as the large and changing sizes of the state and action spaces, partial observability and varied network topology.

The environment is modelled after the `OpenAI gym <https://github.com/openai/gym>`_ interface.


What's new
----------

Version 0.8.0
*************

- Added option of specifying a 'value' for each host when defining a custom network using the .YAML format (thanks @Joe-zsc for the suggestion).
- Added the 'small-honeypot' scenario to included scenarios.


Version 0.7.5
*************

- Added 'undefined error' to observation to fix issue with initial and later observations being indistinguishable.


Version 0.7.4
*************

- Fixed issues with incorrect observation of host 'value' and 'discovery_value'. Now, when in partially observable mode, the agent will correctly only observe these values on the step that they are recieved
- Some other minor code formatting fixes


Version 0.7.3
*************

- Fixed issue with scenario YAML files not being included with PyPi package
- Added final policy visualisation option to DQN and Q-Learning agents


Version 0.7.2
*************

- Fixed bug with 're-registering' Gym environments when reloading modules
- Added example implementations of Tabular Q-Learning: `agents/ql_agent.py` and `agents/ql_replay.py`
- Added `Agents` section to docs, along with other minor doc updates


Version 0.7.1
*************

- Added some scripts for running random benchmarks and describing benchmark scenarios
- Added some more docs (including for creating custom scenarios) and updated other docs


Version 0.7
***********

- Implemented host based firewalls
- Added priviledge escalation
- Added a demo script, including a pre-trained agent for the 'tiny' scenario
- Fix to upper bound calculation (factored in reward for discovering a host)


Version 0.6
***********

- Implemented compatibility with gym.make()
- Updated docs for loading and interactive with NASimEnv
- Added extra functions to nasim.scenarios to make it easier to load scenarios seperately to a NASimEnv
- Fixed bug to do with class attributes and creating different scenarios in same python session
- Fixed up bruteforce agent and tests


Version 0.5
***********

- First official release on PyPi
- Cleaned up dependencies, setup.py, etc and some small fixes
- First stable version


The Docs
--------

.. toctree::
   :maxdepth: 2

   tutorials/index
   reference/index
   explanations/index
   community/index


How should I cite NASim?
------------------------

Please cite NASim in your publications if you use it in your research. Here is an example BibTeX entry:

.. code-block:: bash

    @misc{schwartz2019nasim,
    title={NASim: Network Attack Simulator},
    author={Schwartz, Jonathon and Kurniawatti, Hanna},
    year={2019},
    howpublished={\url{https://networkattacksimulator.readthedocs.io/}},
    }



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/Jjschwartz/NetworkAttackSimulator
