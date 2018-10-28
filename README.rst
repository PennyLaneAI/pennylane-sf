OpenQML Strawberry Fields Plugin
################################

.. image:: https://img.shields.io/travis/XanaduAI/strawberryfields/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.org/XanaduAI/strawberryfields

.. image:: https://img.shields.io/codecov/c/github/xanaduai/strawberryfields/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/strawberryfields

.. image:: https://img.shields.io/codacy/grade/bd14437d17494f16ada064d8026498dd.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/strawberryfields?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/strawberryfields&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/strawberryfields.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://strawberryfields.readthedocs.io

.. image:: https://img.shields.io/pypi/v/StrawberryFields.svg?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/StrawberryFields

.. image:: https://img.shields.io/pypi/pyversions/StrawberryFields.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/StrawberryFields


This OpenQML plugin allows the Strawberry Fields simulators to be used as OpenQML devices.

`Strawberry Fields <https://strawberryfields.readthedocs.io>`_ is a full-stack Python library for designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

`OpenQML <https://openqml.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides two devices to be used with OpenQML: ``strawberryfields.fock`` and ``strawberryfields.gaussian``. These provide access to the Strawberry Fields Fock and Gaussian backends respectively.

* Supports all core OpenQML operations and expectation values across the two devices.

* Combine Strawberry Fields optimized simulator suite with OpenQML's automatic differentiation and optimization.


Installation
============

OpenQML-SF requires both OpenQML and Strawberry Fields. It can be installed via ``pip``:

.. code-block:: bash

    $ python -m pip install openqml-sf


Getting started
===============

Once the OpenQML-SF plugin is installed, the two provided Strawberry Fields devices can be accessed straight away in OpenQML.

You can instantiate these devices for OpenQML as follows:

.. code-block:: python

    import openqml as qm
    dev_fock = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)
    dev_gaussian = qm.device('strawberryfields.gaussian', wires=2)

These devices can then be used just like other devices for the definition and evaluation of QNodes within OpenQML. For more details, see :ref:`usage` and refer to the OpenQML documentation.


Contributing
============

We welcome contributions - simply fork the OpenQML-SF repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.  All contributers to OpenQML-SF will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on OpenQML and Strawberry Fields.


Authors
=======

Josh Izaac, Ville Bergholm, Christian Gogolin, Maria Schuld, and Nathan Killoran.

If you are doing research using OpenQML, please cite our whitepaper:

.. todo:: insert OpenQML whitepaper citation.

If you are doing research using Strawberry Fields, please cite `our whitepaper <https://arxiv.org/abs/1804.03159>`_:

  Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. *arXiv*, 2018. arXiv:1804.03159


Support
=======

- **Source Code:** https://github.com/XanaduAI/openqml-sf
- **Issue Tracker:** https://github.com/XanaduAI/openqml-sf/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `Strawberry Fields Slack channel <https://u.strawberryfields.ai/slack>`_ -
come join the discussion and chat with our Strawberry Fields team.


License
=======

OpenQML-SF is **free** and **open source**, released under the Apache License, Version 2.0.
