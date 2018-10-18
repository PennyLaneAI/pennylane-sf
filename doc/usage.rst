.. _usage:

Plugin usage
############

OpenQML-SF provides two Strawberry Fields simulator devices for OpenQML:

* :class:`strawberryfields.fock <~StrawberryFieldsFock>`: provides an OpenQML device for the Strawberry Fields Fock simulator.

* :class:`strawberryfields.gaussian <~StrawberryFieldsGaussian>`: provides an OpenQML device for the Strawberry Fields Gaussian simulator.


Using the devices
=================

Once the OpenQML-SF plugin is installed, the two provided Strawberry Fields devices can be accessed straight away in OpenQML.

You can instantiate these devices for OpenQML as follows:

>>> import openqml as qm
>>> from openqml import numpy as np
>>> dev_fock = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)
>>> dev_gaussian = qm.device('strawberryfields.gaussian', wires=2)

These devices can then be used just like other devices for the definition and evaluation of QNodes within OpenQML. For example, a simple example includes a quantum function that first displaces the vacuum state, applies a beamsplitter, and then returns the photon number expectation:


>>> @qm.qnode(dev_fock)
>>> def quantum_function(x, theta):
>>> 	qm.Displacement(x, 0, wires=0)
>>> 	qm.Beamsplitter(theta, 0, wires=[0, 1])
>>> 	return qm.expval.PhotonNumber(0)

Evaluating the QNode for a particular value of the circuit parameters:

>>> quantum_function(1., 0.543)
0.7330132578095255

We can also find the derivative with respect to the first parameter:

>>> dqfunc = qm.grad(quantum_function)
>>> dqfunc(1., 0.543)
1.4660265156190515

The continuous-variable Strawberry Fields based QNodes can also be combined with qubit-based QNodes and classical nodes to build up a hybrid computational model, and can then be used with the optimizers provided by OpenQML.

Device options
==============

The Strawberry Fields simulators accept additional arguments beyond the OpenQML default device arguments.

``cutoff_dim``
	the Fock basis truncation to be applied when executing quantum functions (``strawberryfields.fock`` only).

``hbar=2``
	the convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`. Default value is :math:`\hbar=2`.

``shots=0``
	the number of circuit evaluations/random samples used to estimate expectation values of expectations. The default value of 0 means that the exact expectation value is returned.

	If shots is non-zero, the Strawberry Fields devices calculate the variance of the expectation value(s), and use the `Berry-Esseen theorem <https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem>`_ to estimate the sampled expectation value.


Supported operations
====================


:class:`strawberryfields.fock <~StrawberryFieldsFock>`
	The Strawberry Fields Fock device supports all continuous-variable (CV) operations and expectations provided by OpenQML, including both Gaussian and non-Gaussian operations:

	* **Supported operations:** ``Beamsplitter``, ``ControlledAddition``, ``ControlledPhase``, ``Displacement``, ``Kerr``, ``CrossKerr``, ``QuadraticPhase``, ``Rotation``, ``Squeezing``, ``TwoModeSqueezing``, ``CubicPhase``, ``CatState``, ``CoherentState``, ``FockDensityMatrix``, ``DisplacedSqueezedState``, ``FockState``, ``FockStateVector``, ``SqueezedState``, ``ThermalState``, ``GaussianState``

	* **Supported expectations:** ``PhotonNumber``, ``X``, ``P``, ``Homodyne``, ``PolyXP``


:class:`strawberryfields.gaussian <~StrawberryFieldsGaussian>`
	The Strawberry Fields Gaussian device supports all *Gaussian* continuous-variable (CV) operations and expectations provided by OpenQML:

	* **Supported operations:** ``Beamsplitter``, ``ControlledAddition``, ``ControlledPhase``, ``Displacement``, ``QuadraticPhase``, ``Rotation``, ``Squeezing``, ``TwoModeSqueezing``, ``CoherentState``, ``DisplacedSqueezedState``, ``SqueezedState``, ``ThermalState``, ``GaussianState``

	* **Supported expectations:** ``PhotonNumber``, ``X``, ``P``, ``Homodyne``, ``PolyXP``
