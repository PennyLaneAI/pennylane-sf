PennyLane-Strawberry Fields Plugin
##################################

:Release: |release|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Once the PennyLane-SF plugin is installed, the provided Strawberry Fields devices can be accessed
straight away in PennyLane, without the need to import any additional packages.

Devices
=======

PennyLane-SF provides various Strawberry Fields devices for PennyLane:

.. devicegalleryitem::
    :name: 'strawberryfields.fock'
    :description: Full simulator that supports all continuous-variable operations.
    :link: devices/fock.html

.. devicegalleryitem::
    :name: 'strawberryfields.gaussian'
    :description: Optimized simulator that supports only Gaussian operations and photon number resolving measurements.
    :link: devices/gaussian.html

.. devicegalleryitem::
    :name: 'strawberryfields.remote'
    :description: Remote backends including access to continuous-variable hardware.
    :link: devices/remote.html

.. raw:: html

    <div style='clear:both'></div>
    </br>

.. devicegalleryitem::
    :name: 'strawberryfields.gbs'
    :description: Specialized simulator giving access to analytic gradients in Gaussian boson sampling.
    :link: devices/gbs.html

.. devicegalleryitem::
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

.. demogalleryitem::
    :name: Plugins and Hybrid computation
    :figure: https://pennylane.ai/qml/_images/photon_redirection.png
    :link:  https://pennylane.ai/qml/demos/tutorial_plugins_hybrid.html
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.

.. demogalleryitem::
    :name: Function fitting with a photonic quantum neural network
    :figure: https://pennylane.ai/qml/_images/qnn_output_28_0.png
    :link:  https://pennylane.ai/qml/demos/quantum_neural_net.html
    :tooltip: Fit one-dimensional noisy data with a quantum neural network.

.. demogalleryitem::
    :name: Quantum advantage with Gaussian Boson Sampling
    :figure: https://pennylane.ai/qml/_images/tutorial_gbs_thumbnail.png
    :link: https://pennylane.ai/qml/demos/tutorial_gbs.html
    :tooltip: Construct and simulate a Gaussian Boson Sampler.

.. raw:: html

    </div></div><div style='clear:both'> <br/>


You can also use any of the *continuous-variable* based
`demos from the PennyLane documentation <https://pennylane.ai/qml/demonstrations.html>`_, for example
the tutorial on `Gaussian transformations <https://pennylane.ai/qml/demos/tutorial_gaussian_transformation.html>`_,
and simply replace ``'default.gaussian'`` with any of the available Strawberry Fields devices,
such as ``'strawberryfields.gaussian'``:

.. code-block:: python

    dev = qml.device('strawberryfields.gaussian', wires=XXX)


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
