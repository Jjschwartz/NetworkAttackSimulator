**Status**: Still under development, interface is stable but expect some more features and bug fixes

Network Attack Simulator
========================

|docs|

Network Attack Simulator (NASim) is a simulated computer network complete with vulnerabilities, scans and exploits designed to be used as a testing environment for AI agents and planning techniques applied to network penetration testing.

**Note**: NASim is currently under active development so there will be small changes still occuring that may affect some functionallity. However, I'm hoping for a stable release soon.


Installation
------------

The easiest way to install the latest version of NASim is via pip::

  $ pip install nasim


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

- 2020-07-31 (v 0.5.0)
  + First official release on PyPi
  + Cleaned up dependencies, setup.py, etc and some small fixes


.. |docs| image:: https://readthedocs.org/projects/networkattacksimulator/badge/?version=latest
    :target: https://networkattacksimulator.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    :scale: 100%
