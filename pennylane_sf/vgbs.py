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

import numpy as np

import strawberryfields as sf

# import gates
from strawberryfields.ops import Dgate

from .expectations import identity
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsVGBS(StrawberryFieldsSimulator):
    r"""TODO
    """
    name = 'Strawberry Fields variational GBS PennyLane plugin'
    short_name = 'strawberryfields.vgbs'

    _operation_map = {
        'Displacement': Dgate,
    }

    _observable_map = {
        'Identity': identity
    }

    _circuits = {}

    def __init__(self, wires, *, analytic=True, cutoff_dim, shots=1000, hbar=2):
        super().__init__(wires, analytic=analytic, shots=shots, hbar=hbar)
        self.cutoff = cutoff_dim

    def pre_measure(self):
        self.eng = sf.Engine("gaussian")
        results = self.eng.run(self.prog)

        self.state = results.state
        self.samples = results.samples
