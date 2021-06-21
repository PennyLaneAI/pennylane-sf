# Release 0.16.0

### Breaking changes

* Add support and tests for Python 3.9.
  [(#71)](https://github.com/PennyLaneAI/pennylane-sf/pull/70)
  
### Bug fixes

* Replaced left-over references to the deprecated `analytic` attribute with `shots`.
  [(#68)](https://github.com/PennyLaneAI/pennylane-sf/pull/68)

### Contributors

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, Romain Moyard

---

# Release 0.15.0

### Breaking changes

* For compatibility with PennyLane v0.15, the `analytic` keyword argument
  has been removed from all devices. Analytic expectation values can
  still be computed by setting `shots=None`.
  [(#65)](https://github.com/XanaduAI/pennylane-sf/pull/65)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac

---

# Release 0.14.0

### Improvements

* Updated the tests to work with the new core of PennyLane.
  [(#60)](https://github.com/PennyLaneAI/pennylane-sf/pull/60)

### Bug fixes

* Removed the device differentiation method from the TF device, and fixed it to
  work with the latest PL release.
  [(#62)](https://github.com/PennyLaneAI/pennylane-sf/pull/62)

* Adjusted the `StrawberryFieldsGBS.jacobian` method to work with the new
  core of PennyLane.
  [(#60)](https://github.com/PennyLaneAI/pennylane-sf/pull/60)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Josh Izaac.

# Release 0.12.0

### New features since last release

* A new device, `strawberryfields.tf`, provides support for using Strawberry Fields TF
  backend from within PennyLane.
  [(#50)](https://github.com/PennyLaneAI/pennylane-sf/pull/50)

  ```python
  dev = qml.device('strawberryfields.tf', wires=2, cutoff_dim=10)
  ```

  This device supports classical backpropagation when using the TensorFlow
  interface:

  ```python
    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def circuit(x, theta):
        qml.Displacement(x, 0, wires=0)
        qml.Beamsplitter(theta, 0, wires=[0, 1])
        return qml.probs(wires=0)
  ```

  Gradients will be computed using TensorFlow backpropagation:

  ```pycon
  >>> x = tf.Variable(1.0)
  >>> theta = tf.Variable(0.543)
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(x, theta)
  >>> jac = tape.jacobian(res, x)
  >>> print(jac)
  <tf.Tensor: shape=(1, 10), dtype=float32, numpy=
  array([[-7.0436597e-01,  1.8805575e-01,  3.2707882e-01,  1.4299491e-01,
           3.7763387e-02,  7.2306832e-03,  1.0900890e-03,  1.3535164e-04,
           1.3895189e-05,  9.9099987e-07]], dtype=float32)>
  ```

  For more details, please see the [TF device documentation](https://pennylane-sf.readthedocs.io/en/latest/devices/tf.html)

* A new device, `strawberryfields.gbs`, provides support for training of the Gaussian boson
  sampling (GBS) distribution.
  [(#47)](https://github.com/PennyLaneAI/pennylane-sf/pull/47)

  ```python
  dev = qml.device('strawberryfields.gbs', wires=4, cutoff_dim=4)
  ```

  This device allows the adjacency matrix ``A`` of a graph to be trained. The QNode must have a
  fixed structure:

  ```python
  from pennylane_sf.ops import ParamGraphEmbed
  import numpy as np

  A = np.array([
      [0.0, 1.0, 1.0, 1.0],
      [1.0, 0.0, 1.0, 0.0],
      [1.0, 1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0]])
  n_mean = 2.5

  @qml.qnode(dev)
  def quantum_function(x):
      ParamGraphEmbed(x, A, n_mean, wires=range(4))
      return qml.probs(wires=range(4))
  ```

  Here, ``n_mean`` is the initial mean number of photons in the output GBS samples. The GBS
  probability distribution for a choice of trainable parameters ``x`` can then be accessed:

  ```pycon
  >>> x = 0.9 * np.ones(4)
  >>> quantum_function(x)
  ```

  For more details, please see the [gbs device documentation](https://pennylane-sf.readthedocs.io/en/latest/devices/gbs.html)

### Improvements

* Adds the ability for the `StrawberryFieldsGBS` device to use the
  reparametrization trick in sampling mode.
  [(#55)](https://github.com/PennyLaneAI/pennylane-sf/pull/55)

### Bug fixes

* Applies minor fixes to `RemoteEngine`.
  [(#53)](https://github.com/PennyLaneAI/pennylane-sf/pull/53)

* Sets a fixed cutoff dimension for `RemoteEngine`.
  [(#54)](https://github.com/PennyLaneAI/pennylane-sf/pull/54)

* Adds unwrapping for operation parameters as indexing into NumPy arrays was
  added to PennyLane.
  [(#56)](https://github.com/PennyLaneAI/pennylane-sf/pull/56)

### Contributors

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Josh Izaac.

---

# Release 0.11.0

### New features since last release

* A new device, `strawberryfields.remote`, provides support for Xanadu's photonic
  hardware from within PennyLane.
  [(#41)](https://github.com/PennyLaneAI/pennylane-sf/pull/41)

  ```python
  dev = qml.device('strawberryfields.remote', backend="X8", shots=10, sf_token="XXX")
  ```

  Once created, the device can be bound to photonic QNode for evaluation
  and training:

  ```python
  @qml.qnode(dev)
  def quantum_function(theta, x):
      qml.TwoModeSqueezing(1.0, 0.0, wires=[0, 4])
      qml.TwoModeSqueezing(1.0, 0.0, wires=[1, 5])
      qml.Beamsplitter(theta, phi, wires=[0, 1])
      qml.Beamsplitter(theta, phi, wires=[4, 5])
      return qml.expval(qml.NumberOperator(0))
  ```

  Samples can also be returned from the hardware using

  ```python
  return [qml.sample(qml.NumberOperator(i)) for i in [0, 1, 2, 4]]
  ```

  For more details, please see the [remote device documentation](https://pennylane-sf.readthedocs.io/en/latest/devices/remote.html)

* The Strawberry Fields devices now support returning Fock state
  probabilities.
  [(#39)](https://github.com/PennyLaneAI/pennylane-sf/pull/39)

  ```python
  @qml.qnode(dev)
  def quantum_function(theta, x):
      qml.TwoModeSqueezing(1.0, 0.0, wires=[0, 1])
      return qml.probs(wires=0)
  ```

  If a subset of wires are requested, the marginal probabilities
  will be computed and returned. The returned probabilities will have
  the shape `[cutoff] * wires`.

  If not specified when instantiated, the cutoff for the Gaussian
  simulator is by default 10.

* Added the ability to compute the expectation value and variance of tensor number operators
  [(#37)](https://github.com/XanaduAI/pennylane-sf/pull/37)
  [(#42)](https://github.com/PennyLaneAI/pennylane-sf/pull/42)

* The Strawberry Fields devices now support custom wire labels.
  [(#48)](https://github.com/PennyLaneAI/pennylane-sf/pull/48)

  ```python
  dev = qml.device('strawberryfields.gaussian', wires=['alice', 1])

  @qml.qnode(dev)
  def circuit(x):
      qml.Displacement(x, 0, wires='alice')
      qml.Beamsplitter(wires=['alice', 1])
      return qml.probs(wires=[0, 1])
  ```

### Improvements

* PennyLane-SF has been updated to support the latest version of Strawberry Fields (v0.15)
  [(#44)](https://github.com/PennyLaneAI/pennylane-sf/pull/44)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Maria Schuld, Antal Száva

---

# Release 0.9.0

### Improvements

* Refactored the test suite.
  [#33](https://github.com/XanaduAI/pennylane-sf/pull/33)

### Documentation

* Major redesign of the documentation, making it easier to navigate.
  [#32](https://github.com/XanaduAI/pennylane-sf/pull/32)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Maria Schuld, Antal Száva

---

# Release 0.8.0

### Bug fixes

* Adds the `"model"` key to the `Device._capabilities` dictionary,
  to properly register the device as a CV device. Fixes
  [#28](https://github.com/XanaduAI/pennylane-sf/pull/28)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac

---

# Release 0.6.0

### Bug fixes

* Two missing gates have been added. Cubic phase gate has been added
  to the `strawberryfields.fock` device, and the Inteferometer has
  been added to both `strawberryfields.fock` and `strawberryfields.gaussian`.
  [#25](https://github.com/XanaduAI/pennylane-sf/pull/25)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac

---

# Release 0.5.0

* Renamed the observables `MeanPhoton` to `NumberOperator`, `Homodyne` to `QuadOperator` and
  `NumberState` to `FockStateProjector` to be compatible with the upcoming version of PennyLane (0.5.0).
  [#19](https://github.com/XanaduAI/pennylane-sf/pull/19)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac

---

# Release 0.1.0

Initial public release.

### Contributors
This release contains contributions from:

Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
