.. _distribution:

Distribution
============

This document contains some notes on distributing NASim via PyPi. This is mainly as a reminder for the steps to take when releasing an update.

.. note:: Unless specified otherwise, all bash commands are assumed to be executed from the root directory of the NASim package.


Before pushing to master
~~~~~~~~~~~~~~~~~~~~~~~~

1. Ensure all tests are passing by running:

.. code-block:: bash

   cd test
   pytest

2. Ensure updates are included in the *What's new* section of the *README.rst* and *docs/source/index.rst* files (this step can be ignored for very small changes)
3. Ensure any necessary updates have been included in the documentation.
4. Make sure the documentation can be build by running:

.. code-block:: bash

   cd docs
   make html

5. Ensure ``setup.py`` has been updated to reflect any version and/or dependency changes.


After changes have been pushed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If pushing a new version (MAJOR, MINOR, or MICRO), do the following:

1. Add a tag with the release number to the commit.
2. On github create a new release and link it to the tagged commit
3. Publish the new release to PyPi:

.. code-block:: bash

   # build distributions
   python setup.py sdist bdist_wheel
   # upload latest distribution builds to pypi
   # this will ask for PyPi username and password
   python -m twine upload dist/* --skip-existing
