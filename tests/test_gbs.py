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
import pennylane as qml
import pytest


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
            dev = qml.device("strawberryfields.gbs", wires=2, cutoff_dim=3, backend="fock",
                             analytic=False)
