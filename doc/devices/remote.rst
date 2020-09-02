The Remote device
=================

Pennylane's Remote device gives access to remote backends from Strawberry Fields including
`hardware backends <https://strawberryfields.ai/photonics/hardware/index.html>`_.

Accounts and Tokens
~~~~~~~~~~~~~~~~~~~

By default, the ``strawberryfields.remote`` device will attempt to use an
already active or stored Strawberry Fields account. If the device finds no
account it will raise a warning:

.. code::

    'WARNING:strawberryfields.configuration:No Strawberry Fields configuration file found.'

It is recommended to use the ``strawberryfields.store_account()`` function to
permanently store an account:

.. code-block:: console

    import strawberryfields as sf
    sf.store_account("my_token")

Alternatively, you can use the `Strawberry
Fields command line interface for configuration
<https://strawberryfields.readthedocs.io/en/stable/code/sf_cli.html>`__. Please see
the `Strawberry Fields hardware details <https://strawberryfields.readthedocs.io/en/stable/introduction/photonic_hardware.html>`__
for more information.

.. warning:: Never publish code containing your token online.

Usage
~~~~~

You can instantiate the Remote device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('strawberryfields.remote', backend="X8", shots=10, sf_token="XXX")

The device can then be used to create supported circuits to define and evaluate
QNodes within PennyLane. Refer to the `Strawberry Fields hardware page
<https://strawberryfields.readthedocs.io/en/stable/introduction/photonic_hardware.html>`__
for more details on circuit structures, backends to use and getting an
authentication token.

As an example, the following simple example defines a :code:`quantum_function`
circuit that first applies two-mode squeezing on the the vacuum state, followed
by beamsplitters, and then returns the photon number expectation. This function
is then converted into a QNode which is placed on the
:code:`strawberryfields.remote` device:

.. code-block:: python

    @qml.qnode(dev)
    def quantum_function(theta, phi):
        qml.TwoModeSqueezing(1.0, 0.0, wires=[0,4])
        qml.TwoModeSqueezing(1.0, 0.0, wires=[1,5])
        qml.Beamsplitter(theta, phi, wires=[0,1])
        qml.Beamsplitter(theta, phi, wires=[4,5])
        return qml.expval(qml.NumberOperator(0))

The ``strawberryfields.remote`` device also supports returning Fock basis probabilities:

.. code-block:: python

    @qml.qnode(dev)
    def quantum_function(theta, x):
        qml.TwoModeSqueezing(1.0, 0.0, wires=[0,4])
        qml.TwoModeSqueezing(1.0, 0.0, wires=[1,5])
        qml.Beamsplitter(theta, phi, wires=[0,1])
        qml.Beamsplitter(theta, phi, wires=[4,5])
        return qml.probs(wires=[0, 1, 2, 4])

The probabilities will be returned as a 1-dimensional NumPy array with length :math:`D^N`, where
:math:`N` is the number of wires, and :math:`D` is the Fock basis truncation (one greater
than then number of photons detected).

in addition, Fock basis samples can returned from the device:

.. code-block:: python

    @qml.qnode(dev)
    def quantum_function(theta, x):
        qml.TwoModeSqueezing(1.0, 0.0, wires=[0,4])
        qml.TwoModeSqueezing(1.0, 0.0, wires=[1,5])
        qml.Beamsplitter(theta, phi, wires=[0,1])
        qml.Beamsplitter(theta, phi, wires=[4,5])
        return [qml.sample(qml.NumberOperator(i)) for i in [0, 1, 2, 4]]

This will return a NumPy array of shape ``(len(sampled_modes), shots)``.

Device options
~~~~~~~~~~~~~~

The Strawberry Fields Remote device accepts the following device arguments.

``backend``
    The remote Strawberry Fields backend to use. Authentication is required for connection.

``sf_token``
    The SF API token used for remote access.

``shots=10``
    The number of circuit evaluations/random samples used to estimate
    expectation values and variances of observables.

``hbar=2``
	The convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`.
	Default value is :math:`\hbar=2`.

``wires``
    Iterable that contains unique labels for the modes as numbers or strings
    (i.e., ``['m1', ..., 'm4', 'n1',...,'n4']``). The number of labels must
    match the number of modes accessible on the backend. If not provided, modes
    are addressed as consecutive integers ``[0, 1, ...]``, and their number is
    inferred from the backend.

Supported operations
~~~~~~~~~~~~~~~~~~~~~

The Strawberry Fields Remote device supports a subset of continuous-variable (CV)
operations and observables provided by PennyLane.

* Supported operations: The set of supported operations depends on the specific backend used.
  Please refer to the Strawberry Fields documentation for the chosen backend.

* Supported observables: This device only supports Fock-based measurements, including
  ``qml.probs()``, ``qml.NumberOperator``, and ``qml.TensorN``.
