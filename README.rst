PennyLane Strawberry Fields Plugin
##################################

.. image:: https://img.shields.io/github/actions/workflow/status/PennyLaneAI/pennylane-sf/tests.yml?branch=master&logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane-sf/actions?query=workflow%3ATests

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-sf/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-sf

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-sf/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-sf

.. image:: https://readthedocs.com/projects/xanaduai-pennylane-sf/badge/?version=latest&style=flat-square
    :alt: Read the Docs
    :target: https://docs.pennylane.ai/projects/strawberryfields

.. image:: https://img.shields.io/pypi/v/PennyLane-sf.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-sf

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-sf.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-sf

\

    **❗ This plugin will not be supported in newer versions of PennyLane. It is compatible with versions
    of PennyLane up to and including 0.29❗** Please use 
    `Strawberry Fields <https://strawberryfields.readthedocs.io>`__ instead.

.. header-start-inclusion-marker-do-not-remove

The PennyLane-SF plugin integrates the StrawberryFields photonic quantum computing framework with PennyLane's
quantum machine learning capabilities.

`PennyLane <https://pennylane.readthedocs.io>`__ is a machine learning library for optimization and
automatic differentiation of hybrid quantum-classical computations.

`Strawberry Fields <https://strawberryfields.readthedocs.io>`__ is a full-stack Python library
for designing, simulating, and optimizing photonic quantum circuits.


.. header-end-inclusion-marker-do-not-remove

The plugin documentation can be found here: `PennyLane-Strawberry Fields <https://pennylane-sf.readthedocs.io/en/latest/>`__.


Features
========

* Provides two devices to be used with PennyLane: ``strawberryfields.fock`` and ``strawberryfields.gaussian``. These provide access to the Strawberry Fields Fock and Gaussian backends respectively.

* Supports all core PennyLane operations and observables across the two devices.

* Combine Strawberry Fields optimized simulator suite with PennyLane's automatic differentiation and optimization.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

Installation of PennyLane-SF, as well as all required Python packages, can be installed via ``pip``:
::

   	$ python -m pip install pennylane-sf


Make sure you are using the Python 3 version of pip.

Alternatively, you can install PennyLane-SF from the source code by navigating to the top directory and running
::

	$ python setup.py install

Dependencies
~~~~~~~~~~~~

PennyLane-SF requires the following libraries be installed:

* `Python <http://python.org/>`__ >=3.8

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`__ >=0.19, <0.30
* `StrawberryFields <https://strawberryfields.readthedocs.io/>`__ >=0.22


If you currently do not have Python 3 installed,
we recommend `Anaconda for Python 3 <https://www.anaconda.com/download/>`__, a distributed
version of Python packaged for scientific computation.

Software tests
~~~~~~~~~~~~~~

To ensure that PennyLane-SF is working correctly after installation, the test suite can be
run by navigating to the source code folder and running
::

	$ make test


Documentation
~~~~~~~~~~~~~

To build the HTML documentation, go to the top-level directory and run
::

    $ make docs

The documentation can then be found in the ``doc/_build/html/`` directory.

.. installation-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the PennyLane-SF repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`__ containing your contribution.
All contributers to PennyLane-SF will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links
to cool projects or applications built on PennyLane and Strawberry Fields.


Authors
=======

Josh Izaac, Ville Bergholm, Maria Schuld, Nathan Killoran and Christian Gogolin

If you are doing research using PennyLane and StrawberryFields, please cite our papers:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`__

    Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook.
    *Strawberry Fields: A Software Platform for Photonic Quantum Computing.* 2018.
    `arXiv:1804.03159 <https://arxiv.org/abs/1804.03159>`__

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-sf
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-sf/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

We also have a `Strawberry Fields Slack channel <https://u.strawberryfields.ai/slack>`__ -
come join the discussion and chat with our Strawberry Fields team.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

PennyLane-SF is **free** and **open source**, released under the Apache License, Version 2.0.

.. license-end-inclusion-marker-do-not-remove
