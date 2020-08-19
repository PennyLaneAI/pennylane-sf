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

import numpy as np
import pennylane as qml
import strawberryfields as sf
from pennylane.operation import Probability
from pennylane.wires import Wires
from strawberryfields.ops import GraphEmbed, MeasureFock
from strawberryfields.utils import all_fock_probs_pnr
from thewalrus.quantum import find_scaling_adjacency_matrix as rescale
from thewalrus.quantum import photon_number_mean_vector

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

    def __init__(self, wires, *, analytic=True, cutoff_dim, backend="gaussian", shots=1000, hbar=2):
        if not analytic and backend != "gaussian":
            raise ValueError("Only the Gaussian backend is supported in non-analytic mode.")

        super().__init__(wires, analytic=analytic, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim
        self.backend = backend
        self._params = None
        self._WAW = None

    @staticmethod
    def calculate_WAW(params, A, n_mean):
        """Calculates the :math:`WAW` matrix.

        Rescales :math:`A` so that when encoded in GBS the mean photon number is equal to
        ``n_mean``.

        Args:
            params (array[float]): variable parameters
            A (array[float]): adjacency matrix
            n_mean (float): mean number of photons

        Returns:
            array[float]: the :math:`WAW` matrix
        """
        A *= rescale(A, n_mean)
        W = np.diag(np.sqrt(params))
        return W @ A @ W

    @staticmethod
    def calculate_n_mean(A):
        """Calculates the mean number of photons for an adjacency matrix encoded into GBS.

        Note that for ``A`` to be directly encoded into GBS, its singular values must not exceed
        one.

        Args:
            A (array[float]): adjacency matrix

        Returns:
            float: mean number of photons
        """
        singular_values = np.linalg.svd(A, compute_uv=False)
        return np.sum(singular_values ** 2 / (1 - singular_values ** 2))

    # pylint: disable=missing-function-docstring
    def execute(self, queue, observables, parameters=None, **kwargs):
        parameters = parameters or {}
        if len(queue) > 1:
            raise ValueError(
                "The StrawberryFieldsGBS device accepts a single application of ParamGraphEmbed"
            )
        return super().execute(queue, observables, parameters, **kwargs)

    # pylint: disable=pointless-statement,expression-not-assigned
    def apply(self, operation, wires, par):
        self._params, A, _ = par

        if len(self._params) != self.num_wires:
            raise ValueError(
                "The number of variable parameters must be equal to the total number of wires."
            )

        self._WAW = self.calculate_WAW(*par)
        n_mean_WAW = self.calculate_n_mean(self._WAW)

        op = GraphEmbed(self._WAW, mean_photon_per_mode=n_mean_WAW / len(A))
        op | [self.q[wires.index(i)] for i in wires]

        if not self.analytic:
            MeasureFock() | [self.q[wires.index(i)] for i in wires]

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
        wires = wires or self.wires
        wires = Wires(wires)

        if self.analytic:
            return super().probability(wires=wires)

        N = len(wires)
        samples = np.take(self.samples, self.wires.indices(wires), axis=1)

        probs = all_fock_probs_pnr(samples)
        ind = np.indices([self.cutoff] * N).reshape(N, -1).T

        probs = OrderedDict((tuple(i), probs[tuple(i)]) for i in ind)
        return probs

    def jacobian(self, operations, observables, variable_deps):
        """Calculates the Jacobian of the device.

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

        requested_wires = observables[0].wires

        # Create dummy observable to measure probabilities over all wires
        obs_all_wires = qml.Identity(wires=self.wires)
        obs_all_wires.return_type = Probability
        prob = self.execute(operations, [obs_all_wires], parameters=variable_deps)[0]

        jac = self._jacobian_all_wires(prob)

        if requested_wires == self.wires:
            return jac

        # Unflatten into a [cutoff, cutoff, ..., cutoff, num_params] dimensional tensor
        jac = jac.reshape([self.cutoff] * self.num_wires + [self.num_wires])

        # Find indices to trace over
        all_indices = set(range(self.num_wires))
        requested_indices = set(self.wires.indices(requested_wires))
        trace_over_indices = all_indices - requested_indices

        traced_jac = np.sum(jac, axis=tuple(trace_over_indices))

        # Flatten into [cutoff ** num_requested_wires, num_params] dimensional tensor
        traced_jac = traced_jac.reshape(-1, self.num_wires)

        return traced_jac

    def _jacobian_all_wires(self, prob):
        """Calculates the jacobian of the probability distribution with respect to all wires.

        This function uses Eq. (28) of `this <https://arxiv.org/pdf/2004.04770.pdf>`__ paper.

        Args:
            prob (array[float]): the probability distribution as a flat array

        Returns:
            array[float]: the jacobian
        """
        jac = np.zeros([len(prob), self.num_wires])

        n = len(self._WAW)
        disp = np.zeros(2 * n)
        cov = self.calculate_covariance(self._WAW, hbar=self.hbar)
        mean_photons_by_mode = photon_number_mean_vector(disp, cov, hbar=self.hbar)

        for i, s in enumerate(np.ndindex(*[self.cutoff] * self.num_wires)):
            jac[i] = (s - mean_photons_by_mode) * prob[i] / self._params

        return jac

    @staticmethod
    def calculate_covariance(A, hbar):
        r"""Calculates the covariance matrix corresponding to an input adjacency matrix.

        Args:
            A (array[float]): adjacency matrix
            hbar (float): the convention chosen in the canonical commutation relation
                :math:`[x, p] = i \hbar`

        Returns:
            array[float]: covariance matrix
        """
        n = len(A)
        I = np.identity(2 * n)
        o_mat = np.block([[np.zeros_like(A), np.conj(A)], [A, np.zeros_like(A)]])
        cov = hbar * (np.linalg.inv(I - o_mat) - I / 2)
        return cov
