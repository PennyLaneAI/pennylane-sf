# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Strawberry Fields TF backend for PennyLane.
"""
from collections import OrderedDict
from collections.abc import Sequence  # pylint: disable=no-name-in-module
import uuid

import numpy as np
import tensorflow as tf

import strawberryfields as sf
from strawberryfields.backends.tfbackend.states import FockStateTF

# import state preparations
from strawberryfields.ops import (
    # Catstate,
    Coherent,
    DensityMatrix,
    DisplacedSqueezed,
    Fock,
    Ket,
    Squeezed,
    Thermal,
    Gaussian,
)

# import gates
from strawberryfields.ops import (
    BSgate,
    CKgate,
    CXgate,
    CZgate,
    Dgate,
    Kgate,
    Pgate,
    Rgate,
    S2gate,
    Sgate,
    Vgate,
    Interferometer,
)

from pennylane.wires import Wires

from .expectations import mean_photon, number_expectation, homodyne, poly_xp
from .simulator import StrawberryFieldsSimulator


def identity(state, device_wires, params):
    """Computes the expectation value of ``qml.Identity``
    observable in Strawberry Fields, corresponding to the trace.

    Args:
        state (strawberryfields.backends.states.BaseState): the quantum state
        device_wires (Wires): the measured modes
        params (Sequence): sequence of parameters (not used)

    Returns:
        float, float: trace and its variance
    """
    # pylint: disable=unused-argument
    N = state.num_modes
    D = state.cutoff_dim

    if N == len(device_wires):
        # trace of the entire system
        tr = state.trace()
        return tr, tr - tr ** 2

    # get the reduced density matrix
    N = len(device_wires)
    dm = state.reduced_dm(modes=device_wires.tolist())

    # construct the standard 2D density matrix, and take the trace
    new_ax = np.arange(2 * N).reshape([N, 2]).T.flatten()
    tr = tf.math.real(tf.linalg.trace(tf.reshape(tf.transpose(dm, new_ax), [D ** N, D ** N])))

    return tr, tr - tr ** 2


def fock_state(state, device_wires, params):
    """Computes the expectation value of the ``qml.FockStateProjector``
    observable in Strawberry Fields.

    Args:
        state (strawberryfields.backends.states.BaseState): the quantum state
        device_wires (Wires): the measured mode
        params (Sequence): sequence of parameters

    Returns:
        float, float: Fock state probability and its variance
    """
    # pylint: disable=unused-argument
    n = params[0]
    N = state.num_modes

    if N == len(device_wires):
        # expectation value of the entire system
        ex = state.fock_prob(n)
        return ex, ex - ex ** 2

    dm = state.reduced_dm(modes=device_wires.tolist())
    ex = tf.math.real(dm[tuple([n[i // 2] for i in range(len(n) * 2)])])

    var = ex - ex ** 2
    return ex, var


class StrawberryFieldsTF(StrawberryFieldsSimulator):
    r"""StrawberryFields TensorFlow device for PennyLane.

    For more details, see :doc:`/devices/tf`.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems accessible on the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
        cutoff_dim (int): Fock-space truncation dimension
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. If ``analytic=True``,
            this setting is ignored.
        hbar (float): the convention chosen in the canonical commutation
            relation :math:`[x, p] = i \hbar`
    """
    name = "Strawberry Fields TensorFlow PennyLane plugin"
    short_name = "strawberryfields.tf"

    _capabilities = {
        "model": "cv",
        "passthru_interface": "tf",
    }

    _operation_map = {
        # Cannot yet support catstates, since they still accept complex parameter
        # values in Strawberry Fields.
        # "CatState": Catstate,
        "CoherentState": Coherent,
        "FockDensityMatrix": DensityMatrix,
        "DisplacedSqueezedState": DisplacedSqueezed,
        "FockState": Fock,
        "FockStateVector": Ket,
        "SqueezedState": Squeezed,
        "ThermalState": Thermal,
        "GaussianState": Gaussian,
        "Beamsplitter": BSgate,
        "CrossKerr": CKgate,
        "ControlledAddition": CXgate,
        "ControlledPhase": CZgate,
        "Displacement": Dgate,
        "Kerr": Kgate,
        "QuadraticPhase": Pgate,
        "Rotation": Rgate,
        "TwoModeSqueezing": S2gate,
        "Squeezing": Sgate,
        "CubicPhase": Vgate,
        "Interferometer": Interferometer,
    }

    _observable_map = {
        "NumberOperator": mean_photon,
        "TensorN": number_expectation,
        "X": homodyne(0),
        "P": homodyne(np.pi / 2),
        "QuadOperator": homodyne(),
        "PolyXP": poly_xp,
        "FockStateProjector": fock_state,
        "Identity": identity,
    }

    matrix_gates = {
        "FockDensityMatrix",
        "GaussianState",
        "Interferometer",
        "FockStateVector",
    }

    _circuits = {}
    _asarray = staticmethod(tf.convert_to_tensor)

    def __init__(self, wires, *, cutoff_dim, analytic=True, shots=1000, hbar=2):
        super().__init__(wires, analytic=analytic, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim
        self.params = dict()

    def apply(self, operation, wires, par):
        """Apply a quantum operation.

        Args:
            operation (str): name of the operation
            wires (Wires): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """
        # convert PennyLane parameter conventions to
        # Strawberry Fields conventions

        # translate to consecutive wires used by device
        device_wires = self.map_wires(wires)

        if operation not in self.matrix_gates:
            # store parameters
            param_labels = [str(uuid.uuid4()) for _ in range(len(par))]

            for l, v in zip(param_labels, par):
                self.params[l] = v

            par = self.prog.params(*param_labels)

            if not isinstance(par, Sequence):
                par = (par,)

        op = self._operation_map[operation](*par)
        op | [self.q[i] for i in device_wires.labels]  # pylint: disable=pointless-statement

    def pre_measure(self):
        self.eng = sf.Engine("tf", backend_options={"cutoff_dim": self.cutoff})
        results = self.eng.run(self.prog, args=self.params)

        self.state = results.state
        self.samples = results.samples

    def reset(self):
        """Reset the device"""
        self.params = dict()
        super().reset()

    def probability(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        wires = wires or self.wires
        # convert to a wires object
        wires = Wires(wires)
        # translate to wires used by device
        device_wires = self.map_wires(wires)

        N = len(wires)
        cutoff = getattr(self, "cutoff", 10)

        if N == self.state.num_modes:
            # probabilities of the entire system
            probs = tf.reshape(self.state.all_fock_probs(cutoff=cutoff), -1)

        else:
            rdm = self.state.reduced_dm(modes=device_wires.tolist())
            new_state = FockStateTF(rdm, N, pure=False, cutoff_dim=cutoff)
            probs = tf.reshape(new_state.all_fock_probs(cutoff=cutoff), -1)

        ind = np.indices([cutoff] * N).reshape(N, -1).T
        probs = OrderedDict((tuple(k), v) for k, v in zip(ind, probs))
        return probs
