PennyLane Strawberry Fields Plugin
##################################

.. image:: https://img.shields.io/travis/com/XanaduAI/pennylane-sf/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.com/XanaduAI/pennylane-sf

.. image:: https://img.shields.io/codecov/c/github/xanaduai/pennylane-sf/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/pennylane-sf

.. image:: https://img.shields.io/codacy/grade/33d12f7d2d0644968087e33966ed904e.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/pennylane-sf?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/pennylane-sf&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/pennylane-sf.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://pennylane-sf.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane-SF.svg?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-SF

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-SF.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-SF


This PennyLane plugin allows the Strawberry Fields simulators to be used as PennyLane devices.

`Strawberry Fields <https://strawberryfields.readthedocs.io>`_ is a full-stack Python library for designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides two devices to be used with PennyLane: ``strawberryfields.fock`` and ``strawberryfields.gaussian``. These provide access to the Strawberry Fields Fock and Gaussian backends respectively.

* Supports all core PennyLane operations and expectation values across the two devices.

* Combine Strawberry Fields optimized simulator suite with PennyLane's automatic differentiation and optimization.


Installation
============

PennyLane-SF requires both PennyLane and Strawberry Fields. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install pennylane-sf


Getting started
===============

Once the PennyLane-SF plugin is installed, the two provided Strawberry Fields devices can be accessed straight away in PennyLane.

You can instantiate these devices for PennyLane as follows:

.. code-block:: python

    import pennylane as qml
    dev_fock = qml.device('strawberryfields.fock', wires=2, cutoff_dim=10)
    dev_gaussian = qml.device('strawberryfields.gaussian', wires=2)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane. For more details, see the `plugin usage guide <https://pennylane-sf.readthedocs.io/en/latest/usage.html>`_ and refer to the PennyLane documentation.


Contributing
============

We welcome contributions - simply fork the PennyLane-SF repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.  All contributers to PennyLane-SF will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on PennyLane and Strawberry Fields.


Authors
=======

Josh Izaac, Ville Bergholm, Maria Schuld, Nathan Killoran and Christian Gogolin

If you are doing research using PennyLane and StrawberryFields, please cite our papers:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018.
    `arXiv:1811.04968 <https://arxiv.org/abs/1811.04968>`_

    Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook.
    *Strawberry Fields: A Software Platform for Photonic Quantum Computing.* 2018.
    `arXiv:1804.03159  <https://arxiv.org/abs/1804.03159>`_


Support
=======

- **Source Code:** https://github.com/XanaduAI/pennylane-sf
- **Issue Tracker:** https://github.com/XanaduAI/pennylane-sf/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `Strawberry Fields Slack channel <https://u.strawberryfields.ai/slack>`_ -
come join the discussion and chat with our Strawberry Fields team.


License
=======

PennyLane-SF is **free** and **open source**, released under the Apache License, Version 2.0.
