PennyLane-Strawberry Fields Plugin
##################################

:Release: |release|

.. warning::

    This plugin will not be supported in newer versions of Pennylane. It is compatible with versions
    of PennyLane up to and including 0.29. Please use 
    `Strawberry Fields <https://strawberryfields.readthedocs.io>`__ instead.

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Once the PennyLane-SF plugin is installed, the provided Strawberry Fields devices can be accessed
straight away in PennyLane, without the need to import any additional packages.

Devices
=======

PennyLane-SF provides various Strawberry Fields devices for PennyLane:

.. title-card::
    :name: 'strawberryfields.fock'
    :description: Full simulator that supports all continuous-variable operations.
    :link: devices/fock.html

.. title-card::
    :name: 'strawberryfields.gaussian'
    :description: Optimized simulator that supports only Gaussian operations and photon number resolving measurements.
    :link: devices/gaussian.html

.. title-card::
    :name: 'strawberryfields.remote'
    :description: Remote backends including access to continuous-variable hardware.
    :link: devices/remote.html

.. raw:: html

    <div style='clear:both'></div>
    </br>

.. title-card::
    :name: 'strawberryfields.gbs'
    :description: Specialized simulator giving access to analytic gradients in Gaussian boson sampling.
    :link: devices/gbs.html

.. title-card::
    :name: 'strawberryfields.tf'
    :description: TensorFlow simulator that supports backpropagation and all continuous-variable operations.
    :link: devices/tf.html

.. raw:: html

    <div style='clear:both'></div>
    </br>

.. note::

    The Strawberry Fields plugin only supports :ref:`continuous-variable (CV) operations <intro_ref_ops_cv>`,
    such as :class:`~.pennylane.Squeezing`, or :class:`~.pennylane.NumberOperator`.


Tutorials
=========


Check out these demos to see the PennyLane-SF plugin in action:

.. raw:: html

    <div class="row">

.. title-card::
    :name: Plugins and Hybrid computation
    :description: <img src="https://pennylane.ai/qml/_images/photon_redirection.png" width="100%"/>
    :link: https://pennylane.ai/qml/demos/tutorial_plugins_hybrid.html

.. title-card::
    :name: Function fitting with a photonic quantum neural network
    :description: <img src="https://pennylane.ai/qml/_images/qnn_output_28_0.png" width="100%"/>
    :link: https://pennylane.ai/qml/demos/quantum_neural_net.html

.. title-card::
    :name: Quantum advantage with Gaussian Boson Sampling
    :description: <img src="https://pennylane.ai/qml/_images/tutorial_gbs_thumbnail.png" width="100%"/>
    :link: https://pennylane.ai/qml/demos/tutorial_gbs.html

.. raw:: html

    </div></div><div style='clear:both'> <br/>


You can also use any of the *continuous-variable* based
`demos from the PennyLane documentation <https://pennylane.ai/qml/demonstrations.html>`_, for example
the tutorial on `Gaussian transformations <https://pennylane.ai/qml/demos/tutorial_gaussian_transformation.html>`_,
and simply replace ``'default.gaussian'`` with any of the available Strawberry Fields devices,
such as ``'strawberryfields.gaussian'``:

.. code-block:: python

    dev = qml.device('strawberryfields.gaussian', wires=XXX)

.. raw:: html

    <br/>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices/fock
   devices/gaussian
   devices/remote
   devices/gbs
   devices/tf

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
   ops
