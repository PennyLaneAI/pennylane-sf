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
Strawberry Fields Gaussian device
=================================

**Module name:** :mod:`pennylane_sf.gaussian`

.. currentmodule:: pennylane_sf.gaussian

The Strawberry Fields Gaussian plugin implements all the :class:`~pennylane.device.Device` methods,
and provides a Gaussian simulation of a continuous-variable quantum circuit.

Classes
-------

.. autosummary::
   StrawberryFieldsGaussian


Code details
~~~~~~~~~~~~
"""

import numpy as np

import strawberryfields as sf

#import state preparations
from strawberryfields.ops import (Coherent, DisplacedSqueezed,
                                  Squeezed, Thermal, Gaussian)
# import gates
from strawberryfields.ops import (BSgate, CXgate, CZgate, Dgate,
                                  Pgate, Rgate, S2gate, Sgate)

from .expectations import (identity, mean_photon, homodyne, number_state, poly_xp)
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsGaussian(StrawberryFieldsSimulator):
    """StrawberryFields Gaussian device for PennyLane.
    """
    name = 'Strawberry Fields Gaussian PennyLane plugin'
    short_name = 'strawberryfields.gaussian'

    _operation_map = {
        'CoherentState': Coherent,
        'DisplacedSqueezedState': DisplacedSqueezed,
        'SqueezedState': Squeezed,
        'ThermalState': Thermal,
        'GaussianState': Gaussian,
        'Beamsplitter': BSgate,
        'ControlledAddition': CXgate,
        'ControlledPhase': CZgate,
        'Displacement': Dgate,
        'QuadraticPhase': Pgate,
        'Rotation': Rgate,
        'TwoModeSqueezing': S2gate,
        'Squeezing': Sgate
    }

    _observable_map = {
        'MeanPhoton': mean_photon,
        'X': homodyne(0),
        'P': homodyne(np.pi/2),
        'Homodyne': homodyne(),
        'PolyXP': poly_xp,
        'NumberState': number_state,
        'Identity': identity
    }

    _circuits = {}

    def pre_measure(self):
        self.eng = sf.Engine("gaussian")
        results = self.eng.run(self.prog)

        self.state = results.state
        self.samples = results.samples
