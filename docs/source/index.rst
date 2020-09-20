Welcome to Network Attack Simulator's documentation!
====================================================

Network Attack Simulator (NASim) is a lightweight, high-level network attack simulator written in python. It is designed to be used for rapid testing of autonomous pen-testing agents using reinforcement learning and planning. It is a simulator by definition so does not replicate all details of attacking a real system but it instead aims to capture some of the more salient features of network pen-testing such as the large and changing sizes of the state and action spaces, partial observability and varied network topology.

The environment is modelled after the `OpenAI gym <https://github.com/openai/gym>`_ interface.


What's new
----------

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


Future extensions
*****************

- Add priviledge escalation
- Host based firewalls


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
