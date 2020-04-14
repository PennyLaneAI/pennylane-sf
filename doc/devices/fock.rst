The Fock device
===============

Pennylane's Fock device gives access to
`Strawberry Field's Fock state simulator backend <https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.backends.FockBackend.html>`_.
This simulator represents quantum states in the Fock basis
:math:`\left| 0 \rangle, \left| 1 \rangle, \left| 2 \rangle, \dots, \left| \mathrm{D -1} \rangle`,
where :math:`D` is the user-given value for ``cutoff_dim`` that limits the dimension of the Hilbert space.

The advantage of this representation is that *any* continuous-variable operation can be represented. However,
the **simulations are approximations**, whose accuracy increases with the cutoff dimension.

.. warning::

    It is often useful to keep track of the normalization of a quantum state during optimization, to make sure
    the circuit does not "learn" to push its parameters into a regime where the simulation is vastly inaccurate.

.. note::

    For :math:`M` modes or wires and a cutoff dimension of :math:`D`, the Fock simulator needs to keep track of
    at least :math:`M^D` values. Hence, the simulation time grows much faster with the number of modes than in
    qubit-based simulators.

Usage
~~~~~

You can instantiate the Fock device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=10)

The device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

For instance, the following simple example defines a :code:`quantum_function` circuit that first displaces
the vacuum state, applies a beamsplitter, and then returns the photon number expectation.
This function is then converted into a QNode which is placed on the :code:`strawberryfields.fock` device:

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

The continuous-variable QNodes available via Strawberry Fields can also be combined with qubit-based QNodes
and classical nodes to build up a hybrid computational model. Such hybrid models can be optimized using
the built-in optimizers provided by PennyLane.

Device options
~~~~~~~~~~~~~~

The Strawberry Fields Fock device accepts additional arguments beyond the PennyLane default device arguments.

``cutoff_dim``
	the Fock basis truncation to be applied when executing quantum functions (``strawberryfields.fock`` only)

``hbar=2``
	The convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`.
	Default value is :math:`\hbar=2`.

``shots=0``
	The number of circuit evaluations/random samples used to estimate expectation values of observables.
	The default value of 0 means that the exact expectation value is returned.

	If shots is non-zero, the Fock device calculates the variance of the expectation value(s),
	and use the `Berry-Esseen theorem <https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem>`_ to
	estimate the sampled expectation value.

Supported operations
~~~~~~~~~~~~~~~~~~~~~

The Strawberry Fields Fock device supports all continuous-variable (CV) operations and observables
provided by PennyLane, including both Gaussian and non-Gaussian operations:

* **Supported operations:** ``Beamsplitter``, ``ControlledAddition``, ``ControlledPhase``,
  ``Displacement``, ``Kerr``, ``CrossKerr``, ``QuadraticPhase``, ``Rotation``, ``Squeezing``,
  ``TwoModeSqueezing``, ``CubicPhase``, ``CatState``, ``CoherentState``, ``FockDensityMatrix``,
  ``DisplacedSqueezedState``, ``FockState``, ``FockStateVector``, ``SqueezedState``, ``ThermalState``, ``GaussianState``

* **Supported observables:** ``Identity``, ``NumberOperator``, ``X``, ``P``, ``QuadOperator``, ``PolyXP``
