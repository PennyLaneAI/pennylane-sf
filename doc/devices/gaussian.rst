The Gaussian device
===================

The Gaussian device gives access to Strawberry Field's
`Gaussian simulator backend <https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.backends.GaussianBackend.html>`_.
This backend exploits the compact (and fully classically tractable) representation of
so-called *Gaussian* continuous-variable operations. However, the backend cannot simulate *non-Gaussian* gates,
such as a Cubic Phase or a Kerr gate.

The Gaussian device does not require a cutoff dimensions and simulations are exact up to numerical precision.

Usage
~~~~~

You can instantiate the Gaussian device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('strawberryfields.gaussian', wires=2)

The device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

For instance, the following simple example defines a :code:`quantum_function` circuit that first displaces
the vacuum state, applies a beamsplitter, and then returns the photon number expectation.
This function is converted into a QNode which is placed on the :code:`strawberryfields.gaussian` device:

.. code-block:: python

    @qml.qnode(dev)
    def quantum_function(x, theta):
        qml.Displacement(x, 0, wires=0)
        qml.Beamsplitter(theta, 0, wires=[0, 1])
        return qml.expval(qml.NumberOperator(0))

We can evaluate the QNode for arbitrary values of the circuit parameters:

>>> quantum_function(1., 0.543)
0.7330132578095255

We can also evaluate the derivative with respect to any parameter(s):

>>> dqfunc = qml.grad(quantum_function, argnum=0)
>>> dqfunc(1., 0.543)
1.4660265156190515

.. note::

    The ``qml.state``, ``qml.sample`` and ``qml.density_matrix`` measurements
    are not supported on the ``strawberryfields.gaussian`` device.

The continuous-variable QNodes available via Strawberry Fields can also be combined with qubit-based QNodes
and classical nodes to build up a hybrid computational model. Such hybrid models can be optimized using
the built-in optimizers provided by PennyLane.

Device options
~~~~~~~~~~~~~~

The Strawberry Fields Gaussian device accepts additional arguments beyond the PennyLane default device arguments.

``hbar=2``
	The convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`.
	Default value is :math:`\hbar=2`.

``cutoff_dim``
    the Fock basis truncation to be applied when computing quantities in the Fock basis (such as probabilities)

``shots=None``
	The number of circuit evaluations/random samples used to estimate expectation values of observables.
	The default value of ``None`` means that the exact expectation value is returned.

    If shots is a positive integer or a list of integers, the Gaussian device calculates the
    variance of the expectation value(s), and use the `Berry-Esseen theorem
    <https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem>`_ to estimate the sampled
    expectation value.

Supported operations
~~~~~~~~~~~~~~~~~~~~

The Strawberry Fields Gaussian device supports all *Gaussian* continuous-variable (CV) operations and
observables provided by PennyLane.

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.Beamsplitter
    ~pennylane.CoherentState
    ~pennylane.ControlledAddition
    ~pennylane.ControlledPhase
    ~pennylane.DisplacedSqueezedState
    ~pennylane.Displacement
    ~pennylane.GaussianState
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
    ~pennylane.X
    ~pennylane.P
    ~pennylane.QuadOperator
    ~pennylane.PolyXP
    ~pennylane.TensorN

.. raw:: html

    </div>
