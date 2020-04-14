PennyLane-Strawberry Fields Plugin
##################################

:Release: |release|

.. image:: _static/puzzle_sf.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Once the PennyLane-SF plugin is installed, the two provided Strawberry Fields devices can be accessed
straight away in PennyLane, without the need to import any additional packages.

Devices
~~~~~~~

PennyLane-SF provides two Strawberry Fields devices for PennyLane:

.. devicegalleryitem::
    :name: 'strawberryfields.fock'
    :description: Supports all operations, including non-Gaussian gates and measurements.
    :link: devices/fock.html

.. devicegalleryitem::
    :name: 'strawberryfields.gaussian'
    :description: Optimized to support only Gaussian operations.
    :link: devices/gaussian.html

.. raw:: html

        <div style='clear:both'></div>
        </br>

.. note::

    The Strawberry Fields plugin only supports
    `continuous-variable (CV) operations <https://pennylane.readthedocs.io/en/stable/introduction/operations.html>`_,
    such as ``Squeezing``, or ``NumberOperator``.


Tutorials
~~~~~~~~~

To see the PennyLane-SF plugin in action, you can use any of the *continuous-variable* based `demos
from the PennyLane documentation <https://pennylane.ai/qml/demonstrations.html>`_, for example
the tutorial on `Gaussian transformations <https://pennylane.ai/qml/demos/tutorial_gaussian_transformation.html>`_,
and simply replace ``'default.gaussian'`` with any of the available Strawberry Fields devices,
such as ``'strawberryfields.gaussian'``:

.. code-block:: python

    dev = qml.device('strawberryfields.gaussian', wires=XXX)

To filter tutorials that explicitly use a StrawberryFields device,
use the "Strawberry Fields" filter on the right panel of the
`demos <https://pennylane.ai/qml/demonstrations.html>`_.


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

   devices/aer
   devices/basicaer
   devices/ibmq

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code