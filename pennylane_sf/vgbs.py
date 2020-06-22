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
Strawberry Fields variational GBS device
========================================

**Module name:** :mod:`pennylane_sf.vgbs`

.. currentmodule:: pennylane_sf.vgbs

TODO

Classes
-------

.. autosummary::
   StrawberryFieldsVGBS


Code details
~~~~~~~~~~~~
"""
from thewalrus.quantum import find_scaling_adjacency_matrix as rescale
import pennylane as qml

import numpy as np

import strawberryfields as sf

# import gates
from strawberryfields.ops import Dgate, GraphEmbed

from .expectations import identity
from .simulator import StrawberryFieldsSimulator


class GraphTrain(qml.operation.CVOperation):
    """TODO"""

    def __init__(self, *params, **kwargs):
        par = list(params)
        par[2] = np.array(par[2])
        super().__init__(*par, **kwargs)

    num_params = 3
    num_wires = qml.operation.AllWires
    par_domain = 'A'

    grad_method = 'F'  # This would be better as A, but is incompatible with array inputs
    grad_recipe = None


class StrawberryFieldsVGBS(StrawberryFieldsSimulator):
    r"""TODO
    """
    name = 'Strawberry Fields variational GBS PennyLane plugin'
    short_name = 'strawberryfields.vgbs'

    _operation_map = {
        'Displacement': Dgate,
        "GraphTrain": GraphEmbed,
    }

    _observable_map = {
        'Identity': identity
    }

    _circuits = {}

    def __init__(self, wires, *, analytic=True, cutoff_dim, shots=1000, hbar=2):
        super().__init__(wires, analytic=analytic, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim

    def apply(self, operation, wires, par):
        """TODO
        """
        weights, A, n_mean = par
        A = A * rescale(A, n_mean)
        W = np.diag(np.sqrt(weights))
        WAW = W @ A @ W

        singular_values = np.linalg.svd(WAW, compute_uv=False)
        n_mean_WAW = np.sum(singular_values ** 2 / (1 - singular_values ** 2))

        op = self._operation_map[operation](WAW, mean_photon_per_mode=n_mean_WAW / len(A))
        op | [self.q[i] for i in wires] #pylint: disable=pointless-statement


    def pre_measure(self):
        self.eng = sf.Engine("gaussian")
        results = self.eng.run(self.prog)

        self.state = results.state
        self.samples = results.samples
