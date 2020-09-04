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
This module contains the variational GBS device.

The Strawberry Fields variational GBS device is a simulator that provides a way to encode
variational parameters into GBS so that the gradient with respect to the output probability
distribution is accessible.
"""
from collections import OrderedDict

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import GraphEmbed, MeasureFock
from strawberryfields.utils import all_fock_probs_pnr
from thewalrus.quantum import find_scaling_adjacency_matrix as rescale
from thewalrus.quantum import photon_number_mean_vector

import pennylane as qml
from pennylane.operation import Probability
from pennylane.wires import Wires

from .expectations import identity
from .simulator import StrawberryFieldsSimulator


# pylint: disable=too-many-instance-attributes
class StrawberryFieldsGBS(StrawberryFieldsSimulator):
    r"""StrawberryFields variational GBS device for PennyLane.

    This device provides a simulator that can embed variational parameters into GBS such that the
    analytic gradient of the probability distribution is accessible.

    For more details see :doc:`/devices/gbs`.

    Args:
        wires (int or Iterable[Number, str]]): the number of modes to initialize the device in
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
        cutoff_dim (int): Fock-space truncation dimension
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. If ``analytic=True``,
            this setting is ignored.
        samples (array): TODO pre-generated samples using the input adjacency matrix specified by
            :class:`ParamGraphEmbed`. Probabilities will be inferred from these samples if
            ``use_cache=True`` and ``analytic=False``, resulting in faster evaluation of the
            circuit and its derivative.
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

    def __init__(
        self, wires, *, analytic=True, cutoff_dim, shots=1000, samples=None
    ):
        super().__init__(wires, analytic=analytic, shots=shots)
        self.cutoff = cutoff_dim
        self.samples = samples
        self._use_cache = samples is not None
        self._params = None
        self._WAW = None
        self.Z_inv = None

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

        .. warning::

            Note that for ``A`` to be directly encoded into GBS, its singular values must
            be less than one.

        Args:
            A (array[float]): adjacency matrix

        Returns:
            float: mean number of photons
        """
        singular_values = np.linalg.svd(A, compute_uv=False)

        if not all(singular_values < 1):
            raise ValueError("Singular values of matrix A must be less than 1")

        return np.sum(singular_values ** 2 / (1 - singular_values ** 2))

    # pylint: disable=missing-function-docstring
    def execute(self, queue, observables, parameters=None, **kwargs):
        parameters = parameters or {}
        if len(queue) > 1:
            raise ValueError(
                "The StrawberryFieldsGBS device accepts only a single application of ParamGraphEmbed"
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
        self.eng = sf.Engine("gaussian", backend_options={"cutoff_dim": self.cutoff})

        if self.analytic:
            results = self.eng.run(self.prog)
        elif self._use_cache:
            self.reset = lambda: None
            return
        else:
            results = self.eng.run(self.prog, shots=self.shots)

        self.state = results.state
        self.samples = results.samples

    def _reparametrize_probability(self, p):
        """Takes an input probability distribution of :math:`A` and rescales it to the probability
        distribution of :math:`WAW`.

        Args:
            p (array): the probability distribution of :math:`A`

        Returns:
            array: the probability distribution of :math:`WAW`
        """
        Z_inv = self._calculate_z_inv(self._WAW)
        ind_all_wires = np.ndindex(*[self.cutoff] * self.num_wires)

        for s in ind_all_wires:
            res = np.prod(np.power(self._params, s))
            p[tuple(s)] = res * p[tuple(s)] * Z_inv / self.Z_inv

        return p

    def _marginal_over_wires(self, wires, array):
        """Calculates the marginal distribution over the specified wires by summing over the
        remaining.

        The flat input array is reshaped to have ``self.num_wires + 1`` axes.

        Args:
            wires (int or Iterable[Number, str]]): the wires to find marginals over
            array (array): the input array

        Returns:
            array: the traced-over array
        """
        trailing = int(array.size / self.cutoff ** self.num_wires)

        array = array.reshape([self.cutoff] * self.num_wires + [trailing])

        all_indices = set(range(self.num_wires))
        requested_indices = set(self.wires.indices(wires))

        trace_over_indices = all_indices - requested_indices  # Find indices to trace over
        return np.sum(array, axis=tuple(trace_over_indices)).ravel()

    def probability(self, wires=None):
        wires = wires or self.wires
        wires = Wires(wires)
        ind = np.ndindex(*[self.cutoff] * len(wires))

        if self.analytic:
            return super().probability(wires=wires)

        samples = np.take(self.samples, self.wires.indices(wires), axis=1)

        fock_probs = all_fock_probs_pnr(samples)
        cutoff = fock_probs.shape[0]
        diff = max(self.cutoff - cutoff, 0)
        probs = np.pad(fock_probs, [(0, diff)] * len(wires))

        if self._use_cache:
            probs = self._reparametrize_probability(probs)

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

        return self._marginal_over_wires(requested_wires, jac).reshape(-1, self.num_wires)

    def _jacobian_all_wires(self, prob):
        """Calculates the jacobian of the probability distribution with respect to all wires.

        This function uses Eq. (28) of `this <https://arxiv.org/pdf/2004.04770.pdf>`__ paper.

        Args:
            prob (array[float]): the probability distribution as a flat array

        Returns:
            array[float]: the jacobian
        """
        n = len(self._WAW)
        disp = np.zeros(2 * n)
        cov = self.calculate_covariance(self._WAW, hbar=self.hbar)
        mean_photons_by_mode = photon_number_mean_vector(disp, cov, hbar=self.hbar)

        basis_states = np.array(list(np.ndindex(*[self.cutoff] * self.num_wires)))
        jac = (basis_states - mean_photons_by_mode) * np.expand_dims(prob, 1) / self._params

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

    @staticmethod
    def _calculate_z_inv(A):
        """Calculates the 1/Z normalization coefficient from GBS.

        Args:
            A (array): adjacency matrix

        Returns:
            float: the coefficient
        """
        n = len(A)
        I = np.identity(2 * n)
        o_mat = np.block([[np.zeros_like(A), np.conj(A)], [A, np.zeros_like(A)]])
        return np.sqrt(np.linalg.det(I - o_mat))
