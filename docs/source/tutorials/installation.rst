.. _installation:

Installation
==============


Dependencies
--------------

This framework is tested to work under Python 3.6 or later.

The required dependencies:

* Python >= 3.6
* NumPy >= 1.18
* PyYaml >= 5.3

For rendering:

* NetworkX >= 2.4
* prettytable >= 0.7.2
* Matplotlib >= 3.1.3


We recommend to use the bleeding-edge version and to install it by following the :ref:`dev-install`. If you want a simpler installation procedure and do not intend to modify yourself the learning algorithms etc., you can look at the :ref:`user-install`.

.. _dev-install:

Developer install instructions
-------------------------------

As a developer, you can set you up with the bleeding-edge version of NASim with:

.. code-block:: bash

    git clone -b master https://github.com/Jjschwartz/NetworkAttackSimulator.git

Assuming you already have a python environment with ``pip``, you can automatically install all the dependencies with:

.. code-block:: bash

    pip install -r requirements.txt


And you can install the framework as a package with:

.. code-block:: bash

    python install -e .


.. _user-install:

User install instructions
--------------------------

(Installation via PyPI coming soon.)
