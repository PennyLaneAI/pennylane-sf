The GBS device
==============

The GBS device gives access to a near-term model of photonic quantum computing called `Gaussian
boson sampling <https://strawberryfields.ai/photonics/concepts/gbs.html>`__. This model can
encode graphs and other quantities that can be represented by a symmetric matrix :math:`A`, with
a number of known `applications <https://strawberryfields.ai/photonics/applications.html>`__.
The output of a GBS device is a collection of samples listing the number of photons counted in
each mode, resulting in a probability distribution that captures information about the encoded
matrix.

The GBS device can then be trained by varying an initial matrix according to some trainable
parameters. When doing this, we can access the output GBS probability distribution as well as its
derivative.

Usage
~~~~~

You can instantiate the GBS device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('strawberryfields.gbs', wires=4, cutoff_dim=4)

Note that a ``cutoff_dim`` argument is used to truncate the number of photons in the output GBS
probability distribution.

The GBS device represents a fixed circuit composed of:

- Varying an input matrix according to some trainable parameters and embedding the result
  into GBS,
- Measuring the output probability distribution.

Hence, the ``quantum_function`` used to construct the QNode must follow a restricted pattern:

.. code-block:: python

    from pennylane_sf.ops import ParamGraphEmbed

    @qml.qnode(dev)
    def quantum_function(x):
        ParamGraphEmbed(x, A, n_mean, wires=range(4))
        return qml.probs(wires=range(4))

Here, the :class:`~.ParamGraphEmbed` operation was used to vary a symmetric matrix ``A`` according
to the trainable parameters ``x``. We must also fix an initial mean number of photons ``n_mean``
for the output samples.

Suppose we want to encode a graph with corresponding
`adjacency matrix <https://en.wikipedia.org/wiki/Adjacency_matrix>`__ ``A``. The size of the graph
is set by the number of wires, and hence using the example above we can encode a ``4`` node graph.
Suppose we fix the adjacency matrix and mean number of photons to

.. code-block:: python

    import numpy as np
    A = np.array(
    [[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    n_mean = 2.5

We must also fix a value for the trainable parameters ``x``. These parameters must be
non-negative and are typically chosen with an initial value close to one (which results in an
unchanged ``A``):

.. code-block:: python

    x = 0.9 * np.ones(4)

The GBS probability distribution can then be evaluated using:

.. code-block:: python

    quantum_function(x)

The derivative of the probability distribution can also be calculated using standard methods in
PennyLane. For example,

.. code-block:: python

    d_quantum_function = qml.jacobian(quantum_function)
    d_quantum_function(x)

The GBS probability distribution can also be post-processed and used as the input to other QNodes
or classical nodes.

Background
~~~~~~~~~~

The GBS device can be trained by varying an initial matrix :math:`A` according to some trainable
parameters :math:`\mathbf{w}`. One way to include trainable parameters is to transform :math:`A`
according to

.. math::

    A \rightarrow WAW,

where :math:`W` is a diagonal matrix with values given by :math:`\sqrt{\mathbf{w}}`. Using this
approach, a `recent paper <https://arxiv.org/abs/2004.04770>`__ has shown how to calculate the
derivative of the output GBS probability distribution :math:`P(\mathbf{n}, \mathbf{w})`:

.. math::

    \partial_{\mathbf{w}} P(\mathbf{n}, \mathbf{w}) = \frac{\mathbf{n} - \langle\mathbf{n}\rangle}{\mathbf{w}}P(\mathbf{n}, \mathbf{w}),,

where :math:`\mathbf{n}` is a sample given by counting the number of photons observed in each mode.

Device options
~~~~~~~~~~~~~~

The GBS device accepts additional arguments beyond the PennyLane default device arguments.

``cutoff_dim``
    the Fock basis truncation to be applied when computing probabilities in the Fock basis.

``shots=1000``
	The number of circuit evaluations/random samples used to estimate probabilities.
	Only used when ``analytic=False``, otherwise probabilities are exact.

``hbar=2``
	The convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`.
	Default value is :math:`\hbar=2`.

Supported operations
~~~~~~~~~~~~~~~~~~~~

The GBS device supports is a restricted model of quantum computing and supports only the
following operations and return types:

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane_sf.ops.ParamGraphEmbed

.. raw:: html

    </div>

**Supported return types:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.probs

.. raw:: html

    </div>
