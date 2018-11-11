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
in :meth:`~.StrawberryFieldsSimulator.pre_expval`, as well as defining their ``_operation_map``
and ``_expectation_map``, mapping PennyLane operations to their
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
            to estimate expectation values of expectations.
            For simulator devices, 0 means the exact EV is returned.
        hbar (float): the convention chosen in the canonical commutation relation :math:`[x, p] = i \hbar`
    """
    name = 'Strawberry Fields Simulator PennyLane plugin'
    api_version = '0.1.0'
    version = __version__
    author = 'Josh Izaac'

    short_name = 'strawberryfields'
    _operation_map = {}
    _expectation_map = {}

    def __init__(self, wires, *, shots=0, hbar=2):
        super().__init__(self.short_name, wires, shots)
        self.hbar = hbar
        self.eng = None
        self.q = None
        self.state = None

    def execution_context(self):
        """Initialize the engine"""
        self.reset()
        self.eng, self.q = sf.Engine(self.num_wires, hbar=self.hbar)
        return self.eng

    def apply(self, operation, wires, par):
        """Apply a quantum operation.

        Args:
            operation (str): name of the operation
            wires (Sequence[int]): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """
        op = self._operation_map[operation](*par)
        op | [self.q[i] for i in wires] #pylint: disable=pointless-statement

    @abc.abstractmethod
    def pre_expval(self):
        """Run the engine"""
        raise NotImplementedError

    def expval(self, expectation, wires, par):
        """Evaluate an expectation.

        Args:
            expectation (str): name of the expectation
            wires (Sequence[int]): subsystems the expectation is evaluated on
            par (tuple): parameters for the expectation
        Returns:
            float: expectation value
        """
        ex, var = self._expectation_map[expectation](self.state, wires, par)

        if self.shots != 0:
            # estimate the expectation value
            # use central limit theorem, sample normal distribution once, only ok
            # if shots is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ex = np.random.normal(ex, np.sqrt(var / self.shots))

        return ex

    def reset(self):
        """Reset the device"""
        if self.eng is not None:
            self.eng.reset()
            self.eng = None
        if self.state is not None:
            self.state = None
        if self.q is not None:
            self.q = None

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    @property
    def expectations(self):
        """Get the supported set of expectations.

        Returns:
            set[str]: the set of PennyLane expectation names the device supports
        """
        return set(self._expectation_map.keys())
