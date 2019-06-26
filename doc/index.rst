PennyLane Strawberry Fields Plugin
##################################

:Release: |release|
:Date: |today|


This PennyLane plugin allows the Strawberry Fields simulators to be used as PennyLane devices.

`Strawberry Fields <https://strawberryfields.readthedocs.io>`_ is a full-stack Python library for designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization and automatic differentiation of hybrid quantum-classical computations.


Features
========

* Provides two devices to be used with PennyLane: ``strawberryfields.fock`` and ``strawberryfields.gaussian``. These provide access to the Strawberry Fields Fock and Gaussian backends respectively.


* Supports all core PennyLane operations and observables across the two devices.


* Combine Strawberry Fields' advanced quantum simulator suite with PennyLane's automatic differentiation and optimization.


To get started with the PennyLane Strawberry Fields plugin, follow the :ref:`installation steps <installation>`, then see the :ref:`usage <usage>` page.


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
   :maxdepth: 2
   :caption: Tutorials (external links)

   Photon redirection <https://pennylane.readthedocs.io/en/latest/tutorials/plugins_hybrid.html>
   Notebook downloads <https://pennylane.readthedocs.io/en/latest/tutorials/notebooks.html>

.. rst-class:: contents local topic

.. toctree::
   :maxdepth: 1
   :caption: Code details

   code/simulator
   code/fock
   code/gaussian
   code/expectations

