.. _usage:

Plugin usage
############

PennyLane-SF provides two Strawberry Fields simulator devices for PennyLane:

* :class:`strawberryfields.fock <~StrawberryFieldsFock>`: provides an PennyLane device for the Strawberry Fields Fock simulator

* :class:`strawberryfields.gaussian <~StrawberryFieldsGaussian>`: provides an PennyLane device for the Strawberry Fields Gaussian simulator


Using the devices
=================

Once Strawberry Fields and the PennyLane-SF plugin are installed, the two Strawberry Fields devices
can be accessed straight away in PennyLane.

You can instantiate these devices in PennyLane as follows:

>>> import pennylane as qml
>>> from pennylane import numpy as np
>>> dev_fock = qml.device('strawberryfields.fock', wires=2, cutoff_dim=10)
>>> dev_gaussian = qml.device('strawberryfields.gaussian', wires=2)

These devices can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

For instance, the following simple example defines a :code:`quantum_function` circuit that first displaces
the vacuum state, applies a beamsplitter, and then returns the photon number expectation.
This function is then converted into a QNode which is placed on the :code:`strawberryfields.fock` device:


>>> @qml.qnode(dev_fock)
>>> def quantum_function(x, theta):
>>> 	qml.Displacement(x, 0, wires=0)
>>> 	qml.Beamsplitter(theta, 0, wires=[0, 1])
>>> 	return qml.expval(qml.NumberOperator(0))

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
==============

The Strawberry Fields simulators accept additional arguments beyond the PennyLane default device arguments.

``cutoff_dim``
	the Fock basis truncation to be applied when executing quantum functions (``strawberryfields.fock`` only)

``hbar=2``
	The convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`.
	Default value is :math:`\hbar=2`.

``shots=0``
	The number of circuit evaluations/random samples used to estimate expectation values of observables.
	The default value of 0 means that the exact expectation value is returned.

	If shots is non-zero, the Strawberry Fields devices calculate the variance of the expectation value(s),
	and use the `Berry-Esseen theorem <https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem>`_ to
	estimate the sampled expectation value.


Supported operations
====================


:class:`strawberryfields.fock <~StrawberryFieldsFock>`
	The Strawberry Fields Fock device supports all continuous-variable (CV) operations and observables
	provided by PennyLane, including both Gaussian and non-Gaussian operations:

	* **Supported operations:** ``Beamsplitter``, ``ControlledAddition``, ``ControlledPhase``, ``Displacement``, ``Kerr``, ``CrossKerr``, ``QuadraticPhase``, ``Rotation``, ``Squeezing``, ``TwoModeSqueezing``, ``CubicPhase``, ``CatState``, ``CoherentState``, ``FockDensityMatrix``, ``DisplacedSqueezedState``, ``FockState``, ``FockStateVector``, ``SqueezedState``, ``ThermalState``, ``GaussianState``

	* **Supported observables:** ``Identity``, ``NumberOperator``, ``X``, ``P``, ``QuadOperator``, ``PolyXP``


:class:`strawberryfields.gaussian <~StrawberryFieldsGaussian>`
	The Strawberry Fields Gaussian device supports all *Gaussian* continuous-variable (CV) operations and
	observables provided by PennyLane:

	* **Supported operations:** ``Beamsplitter``, ``ControlledAddition``, ``ControlledPhase``, ``Displacement``, ``QuadraticPhase``, ``Rotation``, ``Squeezing``, ``TwoModeSqueezing``, ``CoherentState``, ``DisplacedSqueezedState``, ``SqueezedState``, ``ThermalState``, ``GaussianState``

	* **Supported observables:** ``Identity``, ``NumberOperator``, ``X``, ``P``, ``QuadOperator``, ``PolyXP``
