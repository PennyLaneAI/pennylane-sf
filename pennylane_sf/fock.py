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
Strawberry Fields Fock device
=============================

**Module name:** :mod:`pennylane_sf.fock`

.. currentmodule:: pennylane_sf.fock

The Strawberry Fields Fock plugin implements all the :class:`~pennylane.device.Device` methods,
and provides a Fock-space simulation of a continuous-variable quantum circuit.

Classes
-------

.. autosummary::
   StrawberryFieldsFock

----
"""

import numpy as np

import strawberryfields as sf

# import state preparations
from strawberryfields.ops import (
    Catstate,
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

from .expectations import identity, mean_photon, number_expectation, homodyne, fock_state, poly_xp
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsFock(StrawberryFieldsSimulator):
    r"""StrawberryFields Fock device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems accessible on the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. If ``None``,
            the device calculates probability, expectation values, and variances analytically.
        cutoff_dim (int): Fock-space truncation dimension
        hbar (float): the convention chosen in the canonical commutation
            relation :math:`[x, p] = i \hbar`
    """

    name = "Strawberry Fields Fock PennyLane plugin"
    short_name = "strawberryfields.fock"

    _operation_map = {
        "CatState": Catstate,
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
        "InterferometerUnitary": Interferometer,
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

    _circuits = {}

    def __init__(self, wires, *, shots=None, cutoff_dim, hbar=2):
        super().__init__(wires, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim

    def pre_measure(self):
        self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff})
        results = self.eng.run(self.prog)

        self.state = results.state
        self.samples = results.samples
