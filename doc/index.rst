OpenQML Strawberry Fields Plugin
################################

:Release: |release|
:Date: |today|


This OpenQML plugin allows the Strawberry Fields simulators to be used as OpenQML devices.

`Strawberry Fields <https://strawberryfields.readthedocs.io>`_ is a full-stack Python library for designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

`OpenQML <https://openqml.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides two devices to be used with OpenQML: ``strawberryfields.fock`` and ``strawberryfields.gaussian``. These provide access to the Strawberry Fields Fock and Gaussian backends respectively.


* Supports all core OpenQML operations and expectation values across the two devices.


* Combine Strawberry Fields' advanced quantum simulator suite with OpenQML's automatic differentiation and optimization.


To get started with the OpenQML Strawberry Fields plugin, follow the :ref:`installation steps <installation>`, then see the :ref:`usage <usage>` page.


Contents
========

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installing
   usage

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 1
   :caption: Code details

   code/simulator
   code/fock
   code/gaussian
   code/expectations

