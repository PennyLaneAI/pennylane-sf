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

The Strawberry Fields variational GBS device provides a way to encode variational parameters into
GBS so that the gradient with respect to the output probability distribution is accessible.

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
from strawberryfields.ops import GraphEmbed, MeasureFock

from .expectations import identity
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsGBS(StrawberryFieldsSimulator):
    r"""StrawberryFields variational GBS device for PennyLane.

    This device provides a method to embed variational parameters into GBS such that the analytic
    gradient of the probability distribution is accessible.

    Args:
        wires (int): the number of modes to initialize the device in
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
        cutoff_dim (int): Fock-space truncation dimension
        backend (str): name of the remote backend to be used
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. If ``analytic=True``,
            this setting is ignored.
        hbar (float): the convention chosen in the canonical commutation
            relation :math:`[x, p] = i \hbar`
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
        if not analytic and backend != "gaussian":
            raise ValueError("Only the Gaussian backend is supported in non-analytic mode.")

        super().__init__(wires, analytic=analytic, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim
        self.backend = backend
        self._params = None
        self._WAW = None

    @staticmethod
    def _calculate_WAW(params, A, n_mean):
        """Calculate the :math:`WAW` matrix.

        Rescales :math:`A` so that when encoded in GBS the mean photon number is equal to
        ``n_mean``.

        Args:
            params (array[float]): variable parameters
            A (array[float]): adjacency matrix
            n_mean (float): mean number of photons

        Returns:
            array[float]: the :math`WAW` matrix
        """
        A *= rescale(A, n_mean)
        W = np.diag(np.sqrt(params))
        return W @ A @ W

    @staticmethod
    def _calculate_n_mean(A):
        """Calculate the mean number of photons for an adjacency matrix encoded into GBS.

        Args:
            A (array[float]): adjacency matrix

        Returns:
            float: mean number of photons
        """
        singular_values = np.linalg.svd(A, compute_uv=False)
        return np.sum(singular_values ** 2 / (1 - singular_values ** 2))

    def apply(self, operation, wires, par):
        self._params, A, _ = par

        if len(self._params) != self.num_wires:
            raise ValueError("The number of variable parameters must be equal to the total number "
                             "of wires.")

        self._WAW = self._calculate_WAW(*par)
        n_mean_WAW = self._calculate_n_mean(self._WAW)

        op = self._operation_map[operation](self._WAW, mean_photon_per_mode=n_mean_WAW / len(A))
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
        wires = wires if wires else range(self.num_wires)

        if self.analytic:
            return super().probability(wires=wires)

        N = len(wires)
        samples = np.take(self.samples, wires, axis=1)

        probs = all_fock_probs_pnr(samples)
        ind = np.indices([self.cutoff] * N).reshape(N, -1).T

        probs = OrderedDict((tuple(i), probs[tuple(i)]) for i in ind)
        return probs

    def jacobian(self, operations, observables, variable_deps):
        """Calculate the Jacobian of the device.

        Args:
            operations (list[~pennylane.operation.Operation]): operations to be applied to the
                device
            observables (list[~pennylane.operation.Operation]): observables to be measured
            variable_deps (dict[int, ParameterDependency]): reference dictionary
                mapping free parameter values to the operations that
                depend on them

        Returns:
            array[float]: Jacobian matrix of size (``len(probs)``, ``num_wires``)
        """
        self.reset()
        prob = np.squeeze(self.execute(operations, observables, parameters=variable_deps))

        jac = np.zeros([len(prob), self.num_wires])

        n = len(self._WAW)
        disp = np.zeros(2 * n)
        cov = self._calculate_covariance(self._WAW, hbar=self.hbar)
        mean_photons_by_mode = photon_number_mean_vector(disp, cov, hbar=self.hbar)

        for i, s in enumerate(np.ndindex(*[self.cutoff] * self.num_wires)):
            jac[i] = (s - mean_photons_by_mode) * prob[i] / self._params

        return jac

    @staticmethod
    def _calculate_covariance(A, hbar):
        """Calculate the covariance matrix corresponding to an input adjacency matrix.

        Args:
            A (array[float]): adjacency matrix
            hbar (float): the convention chosen in the canonical commutation relation
                :math:`[x, p] = i \hbar`

        Returns:
            array[float]: covariance matrix
        """
        n = len(A)
        I = np.identity(2 * n)
        o_mat = np.block(
            [[np.zeros_like(A), np.conj(A)], [A, np.zeros_like(A)]]
        )
        cov = hbar * (np.linalg.inv(I - o_mat) - I / 2)
        return cov
