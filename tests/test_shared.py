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


class TestVariance:
    """Test for the device variances"""

    @pytest.mark.parametrize("dev", dev_list)
    def test_tensor_number_displaced(self, dev, tol):
        """Test the variance of the TensorN observable for a displaced state"""

        @qml.qnode(dev)
        def circuit(a, phi):
            qml.Displacement(a, phi, wires=0)
            qml.Displacement(a, phi, wires=1)
            return qml.var(qml.TensorN(wires=[0, 1]))

        a = 0.4
        phi = -0.12

        expected = a ** 4 * (1 + 2 * a ** 2)

        var = circuit(a, phi)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # differentiate with respect to parameter a
        res = qml.jacobian(circuit, argnum=0)(a, phi).flat
        expected_gradient = 4 * (a ** 3 + 3 * a ** 5)
        assert np.allclose(res, expected_gradient, atol=tol, rtol=0)

        # differentiate with respect to parameter phi
        res = qml.jacobian(circuit, argnum=1)(a, phi).flat
        expected_gradient = 0
        assert np.allclose(res, expected_gradient, atol=tol, rtol=0)


@pytest.fixture(scope="function")
def disp_sq_circuit(dev):
    """Quantum node for a displaced squeezed circuit"""

    @qml.qnode(dev)
    def circuit(pars):
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

        var = disp_sq_circuit(pars)

        n0 = np.sinh(rs0) ** 2 + np.abs(alpha0) ** 2
        n1 = np.sinh(rs1) ** 2 + np.abs(alpha1) ** 2
        expected = (
            squared_term(alpha0, rs0, phis0) * squared_term(alpha1, rs1, phis1) - n0 ** 2 * n1 ** 2
        )
        assert np.allclose(var, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev", dev_list)
    def test_tensor_number_displaced_squeezed_pd_squeezing(
        self, dev, disp_sq_circuit, pars, reverted_pars, tol
    ):
        """Test the variance of the TensorN observable for a squeezed displaced
        state

        The analytic expression for the partial derivate wrt r of the second
        squeezing operation can be obtained by passing the parameters of
        operations acting on the second mode first (using reverted_pars).
        """

        def pd_sr(rs0, phis0, rd0, phid0, rs1, phis1, rd1, phid1):
            """Analytic expression for the partial derivative with respect to
            the r argument of the first squeezing operation (rs0)"""
            return (
                (
                    0.25
                    + rd0 ** 2 * (-0.25 - 2 * rd1 ** 2 + 2 * rd1 ** 4)
                    + (-(rd1 ** 2) + rd0 ** 2 * (-1 + 6 * rd1 ** 2)) * np.cosh(2 * rs1)
                    + (-0.25 + 1.25 * rd0 ** 2) * np.cosh(4 * rs1)
                )
                * np.sinh(2 * rs0)
                + (
                    -(rd1 ** 2)
                    + rd1 ** 4
                    + (-0.5 + 2.5 * rd1 ** 2) * np.cosh(2 * rs1)
                    + 0.5 * np.cosh(4 * rs1)
                )
                * np.sinh(4 * rs0)
                + rd1 ** 2
                * np.cos(2 * phid1 - phis1)
                * ((1 - 4 * rd0 ** 2) * np.sinh(2 * rs0) - 1.5 * np.sinh(4 * rs0))
                * np.sinh(2 * rs1)
                + rd0 ** 2
                * np.cos(2 * phid0 - phis0)
                * np.cosh(2 * rs0)
                * (
                    -0.25
                    + 2 * rd1 ** 2
                    - 2 * rd1 ** 4
                    + (1 - 4 * rd1 ** 2) * np.cosh(2 * rs1)
                    - 0.75 * np.cosh(4 * rs1)
                    + 2 * rd1 ** 2 * np.cos(2 * phid1 - phis1) * np.sinh(2 * rs1)
                )
            )

        # differentiate wrt r of the first squeezing operation (rs0)
        grad = qml.jacobian(disp_sq_circuit, argnum=0)(pars)
        expected_gradient = pd_sr(*pars)
        assert np.allclose(grad, expected_gradient, atol=tol, rtol=0)

        # differentiate wrt r of the second squeezing operation (rs1)
        grad = qml.jacobian(disp_sq_circuit, argnum=4)(pars)

        #
        expected_gradient = pd_sr(*reverted_pars)
        assert np.allclose(grad, expected_gradient, atol=tol, rtol=0)

    @pytest.mark.parametrize("dev", dev_list)
    def test_tensor_number_displaced_squeezed_pd_displacement(
        self, dev, disp_sq_circuit, pars, reverted_pars, tol
    ):
        """Test the variance of the TensorN observable for a squeezed displaced
        state

        The analytic expression for the partial derivate wrt r of the second
        displacement operation can be obtained by passing the parameters of
        operations acting on the second mode first (using reverted_pars).
        """

        def pd_dr(rs0, phis0, rd0, phid0, rs1, phis1, rd1, phid1):
            """Analytic expression for the partial derivative with respect to
            the r argument of the first displacement operation (rd0)"""
            return rd0 * (
                0.5
                - rd0 ** 2
                + (-2 + 4 * rd0 ** 2) * rd1 ** 2 * np.cosh(2 * rs1)
                + (-0.5 + rd0 ** 2) * np.cosh(4 * rs1)
                + (2 - 4 * rd0 ** 2) * rd1 ** 2 * np.cos(2 * phid1 - phis1) * np.sinh(2 * rs1)
                + np.cosh(2 * rs0)
                * (
                    -0.25
                    - 2 * rd1 ** 2
                    + 2 * rd1 ** 4
                    + (-1 + 6 * rd1 ** 2) * np.cosh(2 * rs1)
                    + 1.25 * np.cosh(4 * rs1)
                    - 4 * rd1 ** 2 * np.cos(2 * phid1 - phis1) * np.sinh(2 * rs1)
                )
                + np.cos(2 * phid0 - phis0)
                * np.sinh(2 * rs0)
                * (
                    -0.25
                    + 2 * rd1 ** 2
                    - 2 * rd1 ** 4
                    + (1 - 4 * rd1 ** 2) * np.cosh(2 * rs1)
                    - 0.75 * np.cosh(4 * rs1)
                    + 2 * rd1 ** 2 * np.cos(2 * phid1 - phis1) * np.sinh(2 * rs1)
                )
            )

        # differentiate with respect to r of the first displacement operation (rd0)
        grad = qml.jacobian(disp_sq_circuit, argnum=2)(pars)
        expected_gradient = pd_dr(*pars)
        assert np.allclose(grad, expected_gradient, atol=tol, rtol=0)

        # differentiate with respect to r of the second squeezing operation (rd1)
        grad = qml.jacobian(disp_sq_circuit, argnum=6)(pars)

        expected_gradient = pd_dr(*reverted_pars)
        assert np.allclose(grad, expected_gradient, atol=tol, rtol=0)
