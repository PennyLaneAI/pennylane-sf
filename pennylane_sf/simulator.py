# Copyright 2018 Xanadu Quantum Technologies Inc.

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

import numpy as np

from pennylane import Device
import strawberryfields as sf

from ._version import __version__


class StrawberryFieldsSimulator(Device):
    r"""Abstract StrawberryFields simulator device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
        hbar (float): the convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`
    """
    name = 'Strawberry Fields Simulator PennyLane plugin'
    pennylane_requires = '>=0.5.0'
    version = __version__
    author = 'Josh Izaac'

    short_name = 'strawberryfields'
    _operation_map = {}
    _observable_map = {}

    def __init__(self, wires, *, shots=0, hbar=2):
        super().__init__(wires, shots)
        self.hbar = hbar
        self.prog = None
        self.eng = None
        self.q = None
        self.state = None
        self.samples = None

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
            wires (Sequence[int]): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """
        # convert PennyLane parameter conventions to
        # Strawberry Fields conventions
        if operation == "DisplacedSqueezedState":
            sf_par = (par[0]*np.exp(par[1]*1j), par[2], par[3])
        elif operation == "CatState":
            sf_par = (par[0]*np.exp(par[1]*1j), par[2])
        else:
            sf_par = par

        op = self._operation_map[operation](*sf_par)
        op | [self.q[i] for i in wires] #pylint: disable=pointless-statement

    @abc.abstractmethod
    def pre_measure(self):
        """Run the engine"""
        raise NotImplementedError

    def expval(self, observable, wires, par):
        """Evaluate the expectation of an observable.

        Args:
            observable (str): name of the observable
            wires (Sequence[int]): subsystems the observable is evaluated on
            par (tuple): parameters for the observable

        Returns:
            float: expectation value
        """
        ex, var = self._observable_map[observable](self.state, wires, par)

        if self.shots != 0:
            # estimate the expectation value
            # use central limit theorem, sample normal distribution once, only ok
            # if shots is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ex = np.random.normal(ex, np.sqrt(var / self.shots))

        return ex

    def var(self, observable, wires, par):
        """Evaluate the variance of an observable.

        Args:
            observable (str): name of the observable
            wires (Sequence[int]): subsystems the observable is evaluated on
            par (tuple): parameters for the observable

        Returns:
            float: variance value
        """
        _, var = self._observable_map[observable](self.state, wires, par)
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
