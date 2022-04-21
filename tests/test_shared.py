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
Tests for the shared functionalities of devices.
"""
import pytest

import strawberryfields as sf

import pennylane as qml
from pennylane import numpy as np


dev_list = [
    qml.device("strawberryfields.fock", wires=2, cutoff_dim=10),
    qml.device("strawberryfields.gaussian", wires=2),
]


@pytest.fixture(scope="function")
def disp_sq_circuit(dev):
    """Quantum node for a displaced squeezed circuit"""

    @qml.qnode(dev)
    def circuit(*pars):
        qml.Squeezing(pars[0], pars[1], wires=0)
        qml.Displacement(pars[2], pars[3], wires=0)
        qml.Squeezing(pars[4], pars[5], wires=1)
        qml.Displacement(pars[6], pars[7], wires=1)
        return qml.var(qml.TensorN(wires=[0, 1]))

    return circuit


######

# Parameters for a displaced squeezed circuit

# Parameters to operations on the first mode
# Squeezing
rs0, phis0 = 0.1, 0.1

# Displacement
rd0, phid0 = 0.316277, 0.32175055
alpha0 = rd0 * np.exp(1j * phid0)
first_pars = np.array([rs0, phis0, rd0, phid0])

# Parameters to operations on the second mode
# Squeezing
rs1, phis1 = 0.1, 0.15

# Displacement
rd1, phid1 = 0.14142136, 0.78539816
alpha1 = rd1 * np.exp(1j * phid1)
second_pars = np.array([rs1, phis1, rd1, phid1])


@pytest.fixture(scope="function")
def pars():
    """Parameters for the displaced squeezed circuit"""
    return np.concatenate([first_pars, second_pars])


@pytest.fixture(scope="function")
def reverted_pars():
    """Parameters for the displaced squeezed circuit such that parameters for
    operations acting on the second mode precede parameters to the operations
    acting on the first mode"""
    return np.concatenate([second_pars, first_pars])


class TestVarianceDisplacedSqueezed:
    """Test for the device variances of a displaced squeezed circuit"""

    @pytest.mark.parametrize("dev", dev_list)
    def test_tensor_number_displaced_squeezed(self, dev, disp_sq_circuit, pars, tol):
        """Test the variance of the TensorN observable for a squeezed displaced
        state"""

        # Checking the circuit variance and the analytic expression
        def squared_term(a, r, phi):
            """Analytic expression for <N^2>"""
            magnitude_squared = np.abs(a) ** 2
            squared_term = (
                -magnitude_squared
                + magnitude_squared ** 2
                + 2 * magnitude_squared * np.cosh(2 * r)
                - np.exp(-1j * phi) * a ** 2 * np.cosh(r) * np.sinh(r)
                - np.exp(1j * phi) * np.conj(a) ** 2 * np.cosh(r) * np.sinh(r)
                + np.sinh(r) ** 4
                + np.cosh(r) * np.sinh(r) * np.sinh(2 * r)
            )
            return squared_term

        var = disp_sq_circuit(*pars)

        n0 = np.sinh(rs0) ** 2 + np.abs(alpha0) ** 2
        n1 = np.sinh(rs1) ** 2 + np.abs(alpha1) ** 2
        expected = (
            squared_term(alpha0, rs0, phis0) * squared_term(alpha1, rs1, phis1) - n0 ** 2 * n1 ** 2
        )
        assert np.allclose(var, expected, atol=tol, rtol=0)
