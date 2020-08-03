**Status**: Still under development, interface is stable but expect some more features and bug fixes

Network Attack Simulator
========================

|docs|

Network Attack Simulator (NASim) is a simulated computer network complete with vulnerabilities, scans and exploits designed to be used as a testing environment for AI agents and planning techniques applied to network penetration testing.


Installation
------------

The easiest way to install the latest version of NASim hosted on PyPi is via pip::

  $ pip install nasim


To get the latest bleeding edge version and install in development mode see the `Install docs <https://networkattacksimulator.readthedocs.io/en/latest/tutorials/installation.html>`_


Using with OpenAI gym
---------------------

NASim implements the `Open AI Gym <https://github.com/openai/gym>`_ environment interface and so can be used with any algorithm that is developed for that interface.

See `Starting NASim using OpenAI gym <https://networkattacksimulator.readthedocs.io/en/latest/tutorials/gym_load.html>`_.


Documentation
-------------

The documentation is available at: https://networkattacksimulator.readthedocs.io/

Authors
-------

**Jonathon Schwartz** - Jonathon.schwartz@anu.edu.au

License
-------

`MIT`_ © 2020, Jonathon Schwartz

.. _MIT: LICENSE


Future Features
---------------

- Priviledge Escalation
- Host based firewalls


What's new
----------

- 2020-08-02 (v 0.6.0) (MINOR release)

  + Implemented compatibility with gym.make()
  + Updated docs for loading and interactive with NASimEnv
  + Added extra functions to nasim.scenarios to make it easier to load scenarios seperately to a NASimEnv
  + Fixed bug to do with class attributes and creating different scenarios in same python session
  + Fixed up bruteforce agent and tests

- 2020-07-31 (v 0.5.0) (MINOR release)

  + First official release on PyPi
  + Cleaned up dependencies, setup.py, etc and some small fixes


.. |docs| image:: https://readthedocs.org/projects/networkattacksimulator/badge/?version=latest
    :target: https://networkattacksimulator.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    :scale: 100%
