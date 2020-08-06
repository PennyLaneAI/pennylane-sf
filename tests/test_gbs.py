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
from strawberryfields.program import Program

from pennylane_sf import StrawberryFieldsGBS

target_cov = np.array([[ 2.2071e+00,  1.2071e+00,  1.2071e+00,  1.2071e+00, -2.6021e-18,
        -1.3878e-16,  6.9323e-17, -7.8505e-17],
       [ 1.2071e+00,  2.2071e+00,  1.2071e+00,  1.2071e+00, -2.7756e-17,
        -1.1102e-16,  5.5511e-17, -1.1102e-16],
       [ 1.2071e+00,  1.2071e+00,  2.2071e+00,  1.2071e+00, -5.3141e-17,
        -1.6653e-16,  1.0127e-17, -1.3770e-16],
       [ 1.2071e+00,  1.2071e+00,  1.2071e+00,  2.2071e+00, -7.8505e-17,
        -2.2204e-16, -1.5236e-17, -1.6306e-16],
       [-2.6021e-18, -2.7756e-17, -5.3141e-17, -7.8505e-17,  7.9289e-01,
        -2.0711e-01, -2.0711e-01, -2.0711e-01],
       [-1.3878e-16, -1.1102e-16, -1.6653e-16, -2.2204e-16, -2.0711e-01,
         7.9289e-01, -2.0711e-01, -2.0711e-01],
       [ 6.9323e-17,  5.5511e-17,  1.0127e-17, -1.5236e-17, -2.0711e-01,
        -2.0711e-01,  7.9289e-01, -2.0711e-01],
       [-7.8505e-17, -1.1102e-16, -1.3770e-16, -1.6306e-16, -2.0711e-01,
        -2.0711e-01, -2.0711e-01,  7.9289e-01]])


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

    def test_pre_measure_eng(self, tol):
        """Test that the pre_measure method operates as expected by initializing the engine
        correctly"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        prog = Program(4)
        op1 = GraphEmbed(0.1767767 * np.ones((4, 4)), mean_photon_per_mode=0.25)
        prog.append(op1, prog.register)
        dev.prog = prog
        dev.pre_measure()
        assert dev.eng.backend_name == "gaussian"
        assert dev.eng.backend_options == {'cutoff_dim': 3}

    def test_pre_measure_state_and_samples(self, tol):
        """Test that the pre_measure method operates as expected in analytic mode by generating the
        correct output state and not generating samples"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        prog = Program(4)
        op1 = GraphEmbed(0.1767767 * np.ones((4, 4)), mean_photon_per_mode=0.25)
        prog.append(op1, prog.register)
        dev.prog = prog
        dev.pre_measure()

        assert np.allclose(dev.state.displacement(), np.zeros(4))
        assert np.allclose(dev.state.cov(), target_cov, atol=tol)
        assert not dev.samples

    def test_pre_measure_state_and_samples_non_analytic(self, tol):
        """Test that the pre_measure method operates as expected in non-analytic mode by
        generating the correct output state and samples of the right shape"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3, analytic=False, shots=2)
        prog = Program(4)
        op1 = GraphEmbed(0.1767767 * np.ones((4, 4)), mean_photon_per_mode=0.25)
        op2 = MeasureFock()
        prog.append(op1, prog.register)
        prog.append(op2, prog.register)
        dev.prog = prog
        dev.pre_measure()

        assert np.allclose(dev.state.displacement(), np.zeros(4))
        assert np.allclose(dev.state.cov(), target_cov, atol=tol)
        assert dev.samples.shape == (2, 4)

