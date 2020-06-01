Welcome to Network Attack Simulator's documentation!
====================================================

Network Attack Simulator (NASim) is a lightweight, high-level network attack simulator written in python. It is designed to be used for rapid testing of autonomous pen-testing agents using reinforcement learning and planning. It is a simulator by definition so does not replicate all details of attacking a real system but it instead aims to capture some of the more salient features of network pen-testing such as the large and changing sizes of the state and action spaces, partial observability and varied network topology.

The environment is modelled after the `OpenAI gym <https://github.com/openai/gym>`_ interface.


What's new
----------

Version 0.1 (Coming Soon)
*************************
- First stable version
- Documentation

Future extensions
*****************
- Implement as OpenAI gym environment
- Add priviledge escalation


User Guide
----------

.. toctree::
   :maxdepth: 2

   user/installation
   user/development
   user/license
   user/help
   user/acknowledgements
   user/contact


API reference
-------------

Coming soon..


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
