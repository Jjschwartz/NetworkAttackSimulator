.. _installation:

Installation
==============


Dependencies
--------------

This framework is tested to work under Python 3.7 or later.

The required dependencies:

* Python >= 3.7
* Gym >= 0.17
* NumPy >= 1.18
* PyYaml >= 5.3

For rendering:

* NetworkX >= 2.4
* prettytable >= 0.7.2
* Matplotlib >= 3.1.3

We recommend to use the bleeding-edge version and to install it by following the :ref:`dev-install`. If you want a simpler installation procedure and do not intend to modify yourself the learning algorithms etc., you can look at the :ref:`user-install`.

.. _user-install:

User install instructions
--------------------------

NASIm is available on PyPi for and can be installed with ``pip`` with the following command:

.. code-block:: bash

    pip install nasim


This will install the base level, which includes all dependencies needed to use NASim. You can also install the dependencies for building the docs, running tests, and running the DQN example agent seperately or all together, as follows:

.. code-block:: bash

    # install dependencies for building docs
    pip install nasim[docs]

    # install dependencies for running tests
    pip install nasim[test]

    # install dependencies for running dqn_agent
    pip install nasim[dqn]

    # install all dependencies
    pip install nasim[all]



.. _dev-install:

Developer install instructions
-------------------------------

As a developer, you can set you up with the bleeding-edge version of NASim with:

.. code-block:: bash

    git clone -b master https://github.com/Jjschwartz/NetworkAttackSimulator.git


You can install the framework as a package along with all dependencies with (you can remove the '[all]' if you just want base level install):

.. code-block:: bash

    pip install -e .[all]
