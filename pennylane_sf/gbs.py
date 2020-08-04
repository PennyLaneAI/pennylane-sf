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

**Module name:** :mod:`pennylane_sf.gbs`

.. currentmodule:: pennylane_sf.gbs

TODO

Classes
-------

.. autosummary::
   StrawberryFieldsGBS


Code details
~~~~~~~~~~~~
"""
from collections import OrderedDict
from thewalrus.quantum import find_scaling_adjacency_matrix as rescale
from thewalrus.quantum import photon_number_mean_vector

import numpy as np

import strawberryfields as sf
from strawberryfields.utils import all_fock_probs_pnr

# import gates
from strawberryfields.ops import Dgate, GraphEmbed, MeasureFock

from .expectations import identity
from .simulator import StrawberryFieldsSimulator

from pennylane_sf.ops import ParamGraphEmbed



class StrawberryFieldsGBS(StrawberryFieldsSimulator):
    r"""TODO

    The gradient is
        calculated on this device using the results of `Banchi et al.
        <https://arxiv.org/abs/2004.04770>`__.
    """
    name = "Strawberry Fields variational GBS PennyLane plugin"
    short_name = "strawberryfields.gbs"

    _operation_map = {
        "ParamGraphEmbed": GraphEmbed,
    }

    _observable_map = {
        "Identity": identity,
    }

    _circuits = {}

    _capabilities = {"model": "cv", "provides_jacobian": True}

    def __init__(self, wires, *, analytic=True, cutoff_dim, backend="gaussian", shots=1000,
                 hbar=2):
        # if not analytic and backend != "gaussian":
        #     raise ValueError("Only the Gaussian backend is supported in non-analytic mode.")

        super().__init__(wires, analytic=analytic, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim
        self.backend = backend
        self._params = None
        self._WAW = None

    def apply(self, operation, wires, par):
        """TODO
        """
        self._params, A, n_mean = par
        A *= rescale(A, n_mean)
        W = np.diag(np.sqrt(self._params))
        self._WAW = W @ A @ W

        singular_values = np.linalg.svd(self.WAW, compute_uv=False)
        n_mean_WAW = np.sum(singular_values ** 2 / (1 - singular_values ** 2))

        op = self._operation_map[operation](self.WAW, mean_photon_per_mode=n_mean_WAW / len(A))
        op | [self.q[i] for i in wires]  # pylint: disable=pointless-statement

        if not self.analytic:
            MeasureFock() | [self.q[i] for i in wires]

    def pre_measure(self):
        self.eng = sf.Engine(self.backend, backend_options={"cutoff_dim": self.cutoff})

        if self.analytic:
            results = self.eng.run(self.prog)
        else:
            # NOTE: currently, only the gaussian backend supports shots > 1
            results = self.eng.run(self.prog, shots=self.shots)

        self.state = results.state
        self.samples = results.samples

    def probability(self, wires=None):

        if self.analytic:
            # compute the fock probabilities analytically
            # from the state representation
            return super().probability(wires=wires)

        # compute the fock probabilities from samples
        N = len(wires) if wires is not None else len(self.num_wires)
        probs = all_fock_probs_pnr(self.samples)
        ind = np.indices([self.cutoff] * N).reshape(N, -1).T
        probs = OrderedDict((tuple(k), v) for k, v in zip(ind, probs))
        return probs

    def jacobian(self, operations, observables, variable_deps):
        self.reset()
        prob = np.squeeze(self.execute(operations, observables, parameters=variable_deps))

        jac = np.zeros([len(prob), self.num_wires])

        n = len(self._WAW)
        disp = np.zeros(2 * n)
        I = np.identity(2 * n)
        o_mat = np.block(
            [[np.zeros_like(self._WAW), np.conj(self._WAW)], [self._WAW, np.zeros_like(self._WAW)]]
        )
        cov = self.hbar * (np.linalg.inv(I - o_mat) - I / 2)
        mean_photons_by_mode = photon_number_mean_vector(disp, cov, hbar=self.hbar)

        for i, s in enumerate(np.ndindex(5, 5)):
            jac[i] = (s - mean_photons_by_mode) * prob[i] / self._params

        return jac
