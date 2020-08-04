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
The Strawberry Fields remote device implements all the :class:`~pennylane.device.Device` methods,
and provides access to Xanadu's continuous-variable quantum hardware.
"""

from collections import OrderedDict

import numpy as np

import strawberryfields as sf

# import state preparations, gates and measurements
from strawberryfields import ops
from strawberryfields.utils.post_processing import (
    all_fock_probs_pnr,
    samples_expectation,
    samples_variance,
)

from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsRemote(StrawberryFieldsSimulator):
    r"""StrawberryFields remote device for PennyLane.

    A valid Strawberry Fields API token is needed for access. This token can be
    passed when creating the device. The default configuration options of
    Strawberry Fields are used to store the authentication token in a
    configuration file.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): number of circuit evaluations/random samples used to
            estimate expectation values of observables
        backend (str): name of the remote backend to be used
        hbar (float): the convention chosen in the canonical commutation
            relation :math:`[x, p] = i \hbar`
        sf_token (str): the SF API token used for remote access
    """
    name = "Strawberry Fields Hardware PennyLane plugin"
    short_name = "strawberryfields.remote"

    _operation_map = {
        "CatState": ops.Catstate,
        "CoherentState": ops.Coherent,
        "FockDensityMatrix": ops.DensityMatrix,
        "DisplacedSqueezedState": ops.DisplacedSqueezed,
        "FockState": ops.Fock,
        "FockStateVector": ops.Ket,
        "SqueezedState": ops.Squeezed,
        "ThermalState": ops.Thermal,
        "GaussianState": ops.Gaussian,
        "Beamsplitter": ops.BSgate,
        "CrossKerr": ops.CKgate,
        "ControlledAddition": ops.CXgate,
        "ControlledPhase": ops.CZgate,
        "Displacement": ops.Dgate,
        "Kerr": ops.Kgate,
        "QuadraticPhase": ops.Pgate,
        "Rotation": ops.Rgate,
        "TwoModeSqueezing": ops.S2gate,
        "Squeezing": ops.Sgate,
        "CubicPhase": ops.Vgate,
        "Interferometer": ops.Interferometer,
    }

    _observable_map = {
        "Identity": None,
        "NumberOperator": None,
        "TensorN": None,
    }

    def __init__(self, *, backend, shots=1000, hbar=2, sf_token=None):
        self.backend = backend
        eng = sf.RemoteEngine(self.backend)

        # Infer the number of modes from the device specs
        wires = eng.device_spec.modes
        super().__init__(wires, analytic=False, shots=shots, hbar=hbar)
        self.eng = eng

        if sf_token is not None:
            sf.store_account(sf_token)

    def reset(self):
        """Reset the device"""
        sf.hbar = self.hbar

        if self.q is not None:
            self.q = None

        if self.prog is not None:
            self.prog = None

        if self.samples is not None:
            self.samples = None

    def pre_measure(self):
        ops.MeasureFock() | self.q  # pylint: disable=pointless-statement, expression-not-assigned

        # RemoteEngine.run includes compilation that checks the validity of the
        # defined Program
        results = self.eng.run(self.prog, shots=self.shots)
        self.samples = results.samples

    def sample(
        self, observable, wires, par
    ):  # pylint: disable=unused-argument, missing-function-docstring
        if observable == "Identity":
            return np.ones(self.shots)
        wires = np.array(wires)
        selected_samples = self.samples[:, wires]
        return np.prod(selected_samples, axis=1)

    def expval(self, observable, wires, par):
        if observable == "Identity":
            return 1
        return samples_expectation(self.samples)

    def var(self, observable, wires, par):
        if observable == "Identity":
            return 0
        return samples_variance(self.samples)

    def probability(self, wires=None):  # pylint: disable=missing-function-docstring
        all_probs = all_fock_probs_pnr(self.samples)

        # Extract the cutoff value by checking the number of Fock states we
        # obtained probabilities for
        cutoff = all_probs.shape[0]

        if wires is None:

            all_probs = all_probs.flat
            N = self.num_wires
            ind = np.indices([cutoff] * N).reshape(N, -1).T
            all_probs = OrderedDict((tuple(k), v) for k, v in zip(ind, all_probs))
            return all_probs

        all_wires = np.arange(self.num_wires)
        wires_to_trace_out = np.setdiff1d(all_wires, wires)

        if wires_to_trace_out.size > 0:
            all_probs = np.sum(all_probs, axis=tuple(wires_to_trace_out))

        all_probs = all_probs.flat
        N = len(wires)

        ind = np.indices([cutoff] * N).reshape(N, -1).T
        all_probs = OrderedDict((tuple(k), v) for k, v in zip(ind, all_probs))
        return all_probs