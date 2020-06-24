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
Strawberry Fields Hardware device
=================================

**Module name:** :mod:`pennylane_sf.hw`

.. currentmodule:: pennylane_sf.hw

The Strawberry Fields Hardware plugin implements all the :class:`~pennylane.device.Device` methods,
and provides access to Xanadu's continuous-variable quantum hardware.

Classes
-------

.. autosummary::
   StrawberryFieldsRemote


Code details
~~~~~~~~~~~~
"""

import numpy as np
from collections import OrderedDict

import strawberryfields as sf

# import gates
from strawberryfields.ops import (BSgate, Rgate, S2gate, Interferometer,
        MeasureFock)

from strawberryfields.utils.post_processing import samples_expectation, samples_variance, all_fock_probs_pnr

from .simulator import StrawberryFieldsSimulator


class StrawberryFieldsRemote(StrawberryFieldsSimulator):
    r"""StrawberryFields remote device for PennyLane.

    A valid Strawberry Fields API token is needed for access. This token can be
    passed when creating the device. The default options of Strawberry Fields
    are used for configuration to store the authentication token in a
    configuration file.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. If ``analytic=True``,
            this setting is ignored.
        chip (str): name of the remote chip to be used
        hbar (float): the convention chosen in the canonical commutation
            relation :math:`[x, p] = i \hbar`
        sf_token (str): The SF API token.
    """
    name = 'Strawberry Fields Hardware PennyLane plugin'
    short_name = 'strawberryfields.ai'

    _operation_map = {
        'Beamsplitter': BSgate,
        'Rotation': Rgate,
        'TwoModeSqueezing': S2gate,
        'Interferometer': Interferometer
    }

    _observable_map = {
        'Identity': None,
        'TensorN': None,
    }

    def __init__(self, wires, *, chip="X8", shots=1000, hbar=2, sf_token=None):
        super().__init__(wires, analytic=False, shots=shots, hbar=hbar)
        self.chip = chip

        if sf_token is not None:
            sf.store_account(sf_token)


    def pre_measure(self):
        self.eng = sf.RemoteEngine(self.chip)

        self.all_measure_fock()

        # RemoteEngine.run also includes compilation that checks the validity
        # of the defined Program
        results = self.eng.run(self.prog, shots=self.shots)
        self.samples = results.samples

    def all_measure_fock(self):
        """Utility method for measurements in the Fock basis for all modes"""
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
