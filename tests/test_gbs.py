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
Unit tests for the GBS device.
"""
import numpy as np
import pennylane as qml
import pytest
from strawberryfields.ops import GraphEmbed, MeasureFock

from pennylane_sf import StrawberryFieldsGBS


class TestStrawberryFieldsGBS:
    """Integration tests for StrawberryFieldsGBS."""

    @pytest.mark.parametrize("backend", ["fock", "gaussian"])
    @pytest.mark.parametrize("cutoff", [3, 6])
    def test_load_device(self, backend, cutoff):
        """Test that the device loads correctly when analytic is True"""
        dev = qml.device("strawberryfields.gbs", wires=2, cutoff_dim=cutoff, backend=backend)
        assert dev.cutoff == cutoff
        assert dev.backend == backend

    def test_load_device_non_analytic_gaussian(self):
        """Test that the device loads correctly when analytic is False with a gaussian backend"""
        dev = qml.device("strawberryfields.gbs", wires=2, cutoff_dim=3, backend="gaussian",
                         analytic=False)
        assert dev.cutoff == 3
        assert dev.backend == "gaussian"

    def test_load_device_non_analytic_fock(self):
        """Test that the device raises a ValueError when loaded in non-analytic mode with a
        non-Gaussian backend"""
        with pytest.raises(ValueError, match="Only the Gaussian backend is supported"):
            qml.device("strawberryfields.gbs", wires=2, cutoff_dim=3, backend="fock",
                             analytic=False)

    def test_calculate_WAW(self):
        """Test that the _calculate_WAW method calculates correctly when the input adjacency matrix is
        already normalized to have a mean number of photons equal to 1."""
        const = 2
        A = 0.1767767 * np.ones((4, 4))
        params = const * np.ones(4)
        waw = StrawberryFieldsGBS._calculate_WAW(params, A, 1)
        assert np.allclose(waw, const * A)

    def test_calculate_n_mean(self):
        """Test that the _calculate_n_mean calculates correctly"""
        A = 0.1767767 * np.ones((4, 4))
        n_mean = StrawberryFieldsGBS._calculate_n_mean(A)
        assert np.allclose(n_mean, 1)

    def test_apply_analytic(self):
        """Test that the apply method constructs the correct program"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        op = "ParamGraphEmbed"
        wires = list(range(4))

        A = 0.1767767 * np.ones((4, 4))
        params = np.ones(4)
        n_mean = 1
        par = [params, A, n_mean]

        with dev.execution_context():
            dev.apply(op, wires, par)

        circ = dev.prog.circuit
        assert len(circ) == 1
        assert isinstance(circ[0].op, GraphEmbed)

    def test_apply_non_analytic(self):
        """Test that the apply method constructs the correct program when in non-analytic mode"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3, analytic=False)
        op = "ParamGraphEmbed"
        wires = list(range(4))

        A = 0.1767767 * np.ones((4, 4))
        params = np.ones(4)
        n_mean = 1
        par = [params, A, n_mean]

        with dev.execution_context():
            dev.apply(op, wires, par)

        circ = dev.prog.circuit
        assert len(circ) == 2
        assert isinstance(circ[0].op, GraphEmbed)
        assert isinstance(circ[1].op, MeasureFock)
