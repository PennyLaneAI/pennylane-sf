The Remote device
=================

Pennylane's Remote device gives access to remote backends from Strawberry Fields including
`hardware backends <https://strawberryfields.ai/photonics/hardware/index.html>`_.

The advantage of this representation is that *any* continuous-variable operation can be represented. However,
the **simulations are approximations**, whose accuracy increases with the cutoff dimension.

Accounts and Tokens
~~~~~~~~~~~~~~~~~~~

By default, the ``strawberryfields.ai`` device will attempt to use an already active or stored
Strawberry Fields account. If the device finds no account it will raise a warning:

.. code::

    'WARNING:strawberryfields.configuration:No Strawberry Fields configuration file found.'

You can use the ``strawberryfields.store_account("<my_token>")`` function to
permanently store an account.  Alternatively, you can use the `Strawberry
Fields command line interface for configuration
<https://strawberryfields.readthedocs.io/en/stable/code/sf_cli.html>`__.

.. warning:: Never publish code containing your token online.

Usage
~~~~~

You can instantiate the Remote device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('strawberryfields.ai', backend="X8", wires=8, shots=10, sf_token="XXX")

The device can then be used to create supported circuits to define and evaluate
QNodes within PennyLane. Refer to the `Strawberry Fields hardware page
<https://strawberryfields.readthedocs.io/en/stable/introduction/photonic_hardware.html>`__
for more details on circuit structures, backends to use and getting an
authentication token.

As an example, the following simple example defines a :code:`quantum_function`
circuit that first applies two-mode squeezing on the the vacuum state, followed
by beamsplitters, and then returns the photon number expectation. This function
is then converted into a QNode which is placed on the
:code:`strawberryfields.ai` device:

.. code-block:: python

    @qml.qnode(dev)
    def quantum_function(theta, x):
        qml.TwoModeSqueezing(1.0) | (q[0], q[4])
        qml.TwoModeSqueezing(1.0) | (q[1], q[5])
        qml.Beamsplitter(theta, phi,wires=[0,1])
        qml.Beamsplitter(theta, phi,wires=[4,5])
        return qml.expval(qml.NumberOperator(0))


Device options
~~~~~~~~~~~~~~

The Strawberry Fields Fock device accepts additional arguments beyond the PennyLane default device arguments.

``backend``
    The remote Strawberry Fields backend to use. Authentication is required for connection.

``sf_token``
    The SF API token used for remote access.

``shots=10``
    The number of circuit evaluations/random samples used to estimate
    expectation values and variances of observables.

Supported operations
~~~~~~~~~~~~~~~~~~~~~

The Strawberry Fields Fock device supports all continuous-variable (CV) operations and observables
provided by PennyLane, including both Gaussian and non-Gaussian operations.

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.Beamsplitter
    ~pennylane.CatState
    ~pennylane.CoherentState
    ~pennylane.ControlledAddition
    ~pennylane.ControlledPhase
    ~pennylane.CrossKerr
    ~pennylane.CubicPhase
    ~pennylane.DisplacedSqueezedState
    ~pennylane.Displacement
    ~pennylane.FockDensityMatrix
    ~pennylane.FockState
    ~pennylane.FockStateVector
    ~pennylane.GaussianState
    ~pennylane.Interferometer
    ~pennylane.Kerr
    ~pennylane.QuadraticPhase
    ~pennylane.Rotation
    ~pennylane.SqueezedState
    ~pennylane.Squeezing
    ~pennylane.ThermalState
    ~pennylane.TwoModeSqueezing

.. raw:: html

    </div>

**Supported observables:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.Identity
    ~pennylane.NumberOperator
    ~pennylane.TensorN

.. raw:: html

    </div>
