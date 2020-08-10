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
Base simulator class
====================

**Module name:** :mod:`pennylane_sf.simulator`

.. currentmodule:: pennylane_sf.simulator

A base class for constructing Strawberry Fields devices for PennyLane.
This class provides all the boilerplate for supporting PennyLane;
inheriting devices simply need to provide their engine run command
in :meth:`~.StrawberryFieldsSimulator.pre_measure`, as well as defining their ``_operation_map``
and ``_observable_map``, mapping PennyLane operations to their
Strawberry Fields counterparts.

Classes
-------

.. autosummary::
   StrawberryFieldsSimulator

Code details
~~~~~~~~~~~~
"""
import abc
from collections import OrderedDict

import numpy as np

from pennylane import Device
from pennylane.wires import Wires
import strawberryfields as sf
from strawberryfields.backends.states import BaseFockState, BaseGaussianState

from ._version import __version__


class StrawberryFieldsSimulator(Device):
    r"""Abstract StrawberryFields simulator device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. If ``analytic=True``,
            this setting is ignored.
        hbar (float): the convention chosen in the canonical commutation
            relation :math:`[x, p] = i \hbar`
    """
    name = "Strawberry Fields Simulator PennyLane plugin"
    pennylane_requires = ">=0.11.0"
    version = __version__
    author = "Josh Izaac"

    short_name = "strawberryfields"
    _operation_map = {}
    _observable_map = {}
    _capabilities = {"model": "cv"}

    def __init__(self, wires, *, analytic=True, shots=1000, hbar=2):
        super().__init__(wires, shots)
        self.hbar = hbar
        self.prog = None
        self.eng = None
        self.q = None
        self.state = None
        self.samples = None
        self.analytic = analytic

    def execution_context(self):
        """Initialize the engine"""
        self.reset()
        self.prog = sf.Program(self.num_wires)
        self.q = self.prog.register
        return self.prog

    def apply(self, operation, wires, par):
        """Apply a quantum operation.

        Args:
            operation (str): name of the operation
            wires (Wires): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """
        # convert PennyLane parameter conventions to
        # Strawberry Fields conventions
        if operation == "CatState":
            sf_par = (par[0] * np.exp(par[1] * 1j), par[2])
        else:
            sf_par = par

        # translate to consecutive wires used by device
        device_wires = self.map_wires(wires)

        op = self._operation_map[operation](*sf_par)
        op | [self.q[i] for i in device_wires.labels]  # pylint: disable=pointless-statement

    @abc.abstractmethod
    def pre_measure(self):
        """Run the engine"""
        raise NotImplementedError

    def expval(self, observable, wires, par):
        """Evaluate the expectation of an observable.

        Args:
            observable (str): name of the observable
            wires (Wires): subsystems the observable is evaluated on
            par (tuple): parameters for the observable

        Returns:
            float: expectation value
        """
        # translate to consecutive wires used by device
        device_wires = self.map_wires(wires)

        # The different "expectation" functions require different inputs,
        # which is at the moment solved by having dummy arguments.
        # This one-size-fits all "observable_map" logic should be revised.
        if observable == "PolyXP":
            # the poly_xp function currently requires the original wires of the observable
            ex, var = self._observable_map[observable](self.state, self.wires, wires, par)
        else:
            ex, var = self._observable_map[observable](self.state, device_wires, par)

        if not self.analytic:
            # estimate the expectation value
            # use central limit theorem, sample normal distribution once, only ok
            # if shots is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ex = np.random.normal(ex, np.sqrt(var / self.shots))

        return ex

    def var(self, observable, wires, par):
        """Evaluate the variance of an observable.

        Args:
            observable (str): name of the observable
            wires (Wires): subsystems the observable is evaluated on
            par (tuple): parameters for the observable

        Returns:
            float: variance value
        """
        # translate to consecutive wires used by device
        device_wires = self.map_wires(wires)

        # The different "expectation" functions require different inputs,
        # which is at the moment solved by having dummy arguments.
        # This one-size-fits all "observable_map" logic should be revised.
        if observable == "PolyXP":
            # the poly_xp function currently requires the original wires of the observable
            _, var = self._observable_map[observable](self.state, self.wires, wires, par)
        else:
            _, var = self._observable_map[observable](self.state, device_wires, par)

        return var

    def reset(self):
        """Reset the device"""
        sf.hbar = self.hbar

        if self.eng is not None:
            self.eng.reset()
            self.eng = None

        if self.state is not None:
            self.state = None

        if self.q is not None:
            self.q = None

        if self.prog is not None:
            self.prog = None

        if self.samples is not None:
            self.samples = None

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    @property
    def observables(self):
        """Get the supported set of observables.

        Returns:
            set[str]: the set of PennyLane observable names the device supports
        """
        return set(self._observable_map.keys())

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
            probs = self.state.all_fock_probs(cutoff=cutoff).flat

        else:
            if isinstance(self.state, BaseFockState):
                rdm = self.state.reduced_dm(modes=device_wires.tolist())
                new_state = BaseFockState(rdm, N, pure=False, cutoff_dim=cutoff)

            elif isinstance(self.state, BaseGaussianState):
                # Reduced Gaussian state
                mu, cov = self.state.reduced_gaussian(modes=device_wires.tolist())

                # scale so that hbar = 2
                mu /= np.sqrt(sf.hbar / 2)
                cov /= sf.hbar / 2

                # create reduced Gaussian state
                new_state = BaseGaussianState((mu, cov), N)

            probs = new_state.all_fock_probs(cutoff=cutoff).flat

        ind = np.indices([cutoff] * N).reshape(N, -1).T
        probs = OrderedDict((tuple(k), v) for k, v in zip(ind, probs))
        return probs
