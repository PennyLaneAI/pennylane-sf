The TF device
=============

The ``strawberryfields.tf`` device gives access to Strawberry Field's ``tf`` state simulator
backend. This simulator device has the following features:

* The simulator is written using TensorFlow, so supports classical backpropagation using
  PennyLane. Simply use ``interface="tf"`` when creating your QNode.

* Quantum states are represented in the Fock basis :math:`\left| 0 \right>, \left| 1 \right>, \left|
  2 \right>, \dots, \left| \mathrm{D -1} \right>`, where :math:`D` is the user-given value for
  ``cutoff_dim`` that limits the dimension of the Hilbert space.

  The advantage of this representation is that *any* continuous-variable operation can be
  represented. However, the **simulations are approximations**, whose accuracy, the simulation time,
  and required memory increases with the cutoff dimension.

.. warning::

    It is often useful to keep track of the normalization of a quantum state during optimization, to
    make sure the circuit does not "learn" to push its parameters into a regime where the simulation
    is vastly inaccurate.

Usage
~~~~~

You can instantiate the TF device in PennyLane as follows:

>>> import pennylane as qml
>>> import tensorflow as tf
>>> dev = qml.device('strawberryfields.tf', wires=2, cutoff_dim=10)

The device can then be used just like other devices for the definition and evaluation of QNodes
within PennyLane.

For instance, the following simple example defines a QNode that first displaces the vacuum state,
applies a beamsplitter, and then returns the marginal probability on the first wire. This function
is then converted into a QNode which is placed on the :code:`strawberryfields.tf` device:

.. code-block:: python

    @qml.qnode(dev, interface="tf")
    def circuit(x, theta):
        qml.Displacement(x, 0, wires=0)
        qml.Beamsplitter(theta, 0, wires=[0, 1])
        return qml.probs(wires=0)

We can evaluate the QNode for arbitrary values of the circuit parameters:

>>> x = tf.Variable(1.0)
>>> theta = tf.Variable(0.543)
>>> with tf.GradientTape() as tape:
...     res = circuit(x, theta)
>>> print(res)
tf.Tensor(
[[4.8045865e-01+0.j 3.5218298e-01+0.j 1.2907754e-01+0.j 3.1538557e-02+0.j
  5.7795495e-03+0.j 8.4729097e-04+0.j 1.0349592e-04+0.j 1.0811385e-05+0.j
  9.6350857e-07+0.j 6.1937492e-08+0.j]], shape=(1, 10), dtype=complex64)

We can also evaluate the derivative with respect to any parameter(s):

>>> jac = tape.jacobian(res, x)
>>> print(jac)
<tf.Tensor: shape=(1, 10), dtype=float32, numpy=
array([[-7.0436597e-01,  1.8805575e-01,  3.2707882e-01,  1.4299491e-01,
         3.7763387e-02,  7.2306832e-03,  1.0900890e-03,  1.3535164e-04,
         1.3895189e-05,  9.9099987e-07]], dtype=float32)>

The continuous-variable QNodes available via Strawberry Fields can also be combined with qubit-based
QNodes and classical nodes to build up a `hybrid computational model
<https://pennylane.ai/qml/demos/tutorial_plugins_hybrid.html>`_. Such hybrid models can be optimized
using the built-in optimizers provided by PennyLane.

PennyLane CV templates, such as :func:`~pennylane.templates.subroutines.Interferometer`
and :func:`~pennylane.templates.layers.CVNeuralNetLayers`, can also be used:

.. code-block:: python

    dev = qml.device("strawberryfields.tf", wires=3, cutoff_dim=5)

    @qml.qnode(dev, interface="tf")
    def circuit(weights):
        for i in range(3):
            qml.Squeezing(0.1, 0, wires=i)

        qml.templates.Interferometer(
            theta=weights[0],
            phi=weights[1],
            varphi=weights[2],
            wires=[0, 1, 2],
            mesh="rectangular",
        )
        return qml.probs(wires=0)

Once defined, we can now use this QNode within any TensorFlow computation:

>>> weights = qml.init.interferometer_all(n_wires=3)
>>> weights = [tf.convert_to_tensor(w) for w in weights]
>>> with tf.GradientTape() as tape:
...     tape.watch(weights)
...     res = circuit(weights)
>>> grad = tape.gradient(res, weights)
[<tf.Tensor: shape=(3,), dtype=float64, numpy=array([-4.93799348e-07,  5.99637985e-07,  8.90550478e-09])>,
 <tf.Tensor: shape=(3,), dtype=float64, numpy=array([-2.09796852e-07,  1.01452002e-08, -4.34359642e-08])>,
 <tf.Tensor: shape=(3,), dtype=float64, numpy=array([ 8.36735126e-10, -1.21872290e-10, -1.81160686e-09])>]

.. note::

    The ``strawberryfields.tf`` device does not support Autograph mode (``tf.function``).

Device options
~~~~~~~~~~~~~~

The Strawberry Fields TF device accepts additional arguments beyond the PennyLane default device arguments.


``cutoff_dim``
    the Fock basis truncation when applying quantum operations

``hbar=2``
    The convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`.
    Default value is :math:`\hbar=2`.

``analytic=True``
    Indicates if the device should calculate expectations and variances analytically.
    Note that backpropagation is not supported when ``analytic=False``; returned gradients
    and Jacobians will be ``None``.

``shots=1000``
    The number of shots used when returning samples. If ``analytic=False``, the number
    of circuit evaluations/random samples used to estimate expectation values of observables.

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
    ~pennylane.X
    ~pennylane.P
    ~pennylane.QuadOperator
    ~pennylane.PolyXP
    ~pennylane.TensorN

.. raw:: html

    </div>
