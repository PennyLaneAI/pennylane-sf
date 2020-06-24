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
from collections import OrderedDict

import strawberryfields as sf

#import state preparations
from strawberryfields.ops import (Coherent, DisplacedSqueezed,
                                  Squeezed, Thermal, Gaussian)
# import gates
from strawberryfields.ops import (BSgate, CXgate, CZgate, Dgate,
                                  Pgate, Rgate, S2gate, Sgate, Interferometer, MeasureFock)

from strawberryfields.utils.post_processing import samples_expectation, samples_variance, all_fock_probs_pnr

from .expectations import (identity, mean_photon, homodyne, fock_state, poly_xp)
from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsRemote(StrawberryFieldsSimulator):
    # TODO: docstring
    r"""StrawberryFields Gaussian device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        analytic (bool): indicates if the device should calculate expectations
            and variances analytically
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. If ``analytic=True``,
            this setting is ignored.
        hbar (float): the convention chosen in the canonical commutation
            relation :math:`[x, p] = i \hbar`
    """
    name = 'Strawberry Fields Hardware PennyLane plugin'
    short_name = 'strawberryfields.ai'

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
        'Squeezing': Sgate,
        'Interferometer': Interferometer
    }

    _observable_map = {
        'NumberOperator': mean_photon,
        'TensorN': None,
        'X': homodyne(0),
        'P': homodyne(np.pi/2),
        'QuadOperator': homodyne(),
        'PolyXP': poly_xp,
        'FockStateProjector': fock_state,
        'Identity': identity
    }

    def __init__(self, wires, *, chip="X8", shots=1000, hbar=2):
        super().__init__(wires, analytic=False, shots=shots, hbar=hbar)
        self.chip = chip

    def pre_measure(self):
        self.eng = sf.RemoteEngine(self.chip)

        self.all_measure_fock()

        # RemoteEngine.run also includes compilation that checks the validity
        # of the defined Program
        results = self.eng.run(self.prog, shots=self.shots)
        self.samples = results.samples

    def all_measure_fock(self):
        """TODO"""
        MeasureFock() | self.q #pylint: disable=pointless-statement

    def sample(self, observable, wires, par):
        return self.samples

    def expval(self, observable, wires, par):
        return samples_expectation(self.samples)

    def var(self, observable, wires, par):
        return samples_variance(self.samples)

    def probability(self, wires=None):
        all_probs = all_fock_probs_pnr(self.samples)

        if wires is None:
           return all_probs

        all_wires = np.arange(self.num_wires)
        wires_to_trace_out = np.setdiff1d(all_wires, wires)

        if wires_to_trace_out.size > 0:
            all_probs = np.sum(all_probs, axis=tuple(wires_to_trace_out))

        N = len(wires)

        # Extract the cutoff value by checking the number of Fock states we
        # obtained probabilities for
        cutoff = all_probs.shape[0]
        ind = np.indices([cutoff] * N).reshape(N, -1).T
        all_probs = OrderedDict((tuple(k), v) for k, v in zip(ind, all_probs))
        return all_probs
