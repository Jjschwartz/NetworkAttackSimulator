.. _sim_to_real_explanation:

Sim-to-Real Gap Considerations
==============================

NASim is a fairly simplified simulator of network penetration testing. It's main goal is to capture some of the key features of network pentesting in a easy-to-use and fast simulator so that it can be used for rapid testing and prototyping of algorithms before these algorithms are tested on more realistic environments. That is to say there is a bit of gap between the scenarios in NASim and the real world.

In this document we wanted to lay down some considerations to think about when trying to extend your algorithm beyond NASim. This is by no means an exhaustive list, but will hopefully give you something to think about for the next steps, and also give an explanation of some of the design decisions made in NASim.

.. note:: This document is a work in progress so if you have any thoughts, useful references, etc on the topic of applying autonomous penetration testing in the real-world please reach out via email or open an issue on github.

Handling Partial Observability
------------------------------

One of the big assumptions made by NASim is that the pentester agent has access to the network addresses of every host in the network, even in partially observable mode. This information is given to the agent in it's list of actions. In practice in the real-world, depending on the scenario, this assumption may be invalid, and part of the challenge for the pentester is to be able to discover new hosts as they navigate through the network.

The main reason NASim is implemented with the network addresses being known is so that the action space size could be fixed, making it simpler to use with typical Deep Reinforcement Learning algorithms (i.e. with neural nets with fixed size input and output layers).

One of the research challenges is to develop algorithms that can handle action spaces that change as the pentester discovers more network addresses, or perhaps more realistic would be that the pentester's action space is mult-dimensional and includes choosing an address and exploit/scan/etc separately. There actually is some support for this built into NASim with the nasim.envs.action.ParameterisedActionSpace action space (see :ref:`actions`), but even using that action space some information about the size of the network is given to the pentester.

At this stage there is no plans to update NASim to support the no-information action space. This is partially due to time, but also to keep NASim simple and stable and because there are a lot of even better and more realistic environments being developed now (e.g. `CybORG <https://github.com/cage-challenge/CybORG>`_.)

One avenue for handling changing action space is to use auto-regressive actions as was done by `AlphaStar <https://www.deepmind.com/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii>`_.
