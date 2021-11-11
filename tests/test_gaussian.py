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
Unit tests for the Gaussian plugin.
"""
import pytest

import strawberryfields as sf

import pennylane as qml
from pennylane import numpy as np
from pennylane.wires import Wires
from scipy.special import factorial as fac


psi = np.array(
    [
        0.08820314 + 0.14909648j,
        0.32826940 + 0.32956027j,
        0.26695166 + 0.19138087j,
        0.32419593 + 0.08460371j,
        0.02984712 + 0.30655538j,
        0.03815006 + 0.18297214j,
        0.17330397 + 0.2494433j,
        0.14293477 + 0.25095202j,
        0.21021125 + 0.30082734j,
        0.23443833 + 0.19584968j,
    ]
)

one_mode_single_real_parameter_gates = [
    ("ThermalState", qml.ThermalState),
    ("QuadraticPhase", qml.QuadraticPhase),
    ("Rotation", qml.Rotation),
]

two_modes_single_real_parameter_gates = [
    ("ControlledAddition", qml.ControlledAddition),
    ("ControlledPhase", qml.ControlledPhase),
]


unsupported_one_mode_single_real_parameter_gates = [
    ("Kerr", qml.Kerr),
    ("CubicPhase", qml.CubicPhase),
]


# compare to reference SF engine
def SF_gate_reference(sf_op, wires, *args):
    """SF reference circuit for gate tests"""
    eng = sf.Engine("gaussian")
    prog = sf.Program(2)
    with prog.context as q:
        sf.ops.S2gate(0.1) | q
        sf_op(*args) | [q[i] for i in wires]

    state = eng.run(prog).state
    return state.mean_photon(0)[0], state.mean_photon(1)[0]


# compare to reference SF engine
def SF_expectation_reference(sf_expectation, wires, num_wires, *args):
    """SF reference circuit for expectation tests"""
    eng = sf.Engine("gaussian")

    # Allows returning the variance of tensor number for 3 modes
    prog = sf.Program(num_wires)
    with prog.context as q:
        sf.ops.Dgate(0.1) | q[0]
        sf.ops.S2gate(0.1) | (q[0], q[1])

    state = eng.run(prog).state
    return sf_expectation(state, Wires(wires), args)[0]


class TestGaussian:
    """Test the Gaussian simulator."""

    def test_load_gaussian_device(self):
        """Test that the gaussian plugin loads correctly"""
        dev = qml.device("strawberryfields.gaussian", wires=2, cutoff_dim=15)
        assert dev.num_wires == 2
        assert dev.hbar == 2
        assert dev.shots is None
        assert dev.cutoff == 15
        assert dev.short_name == "strawberryfields.gaussian"

    def test_gaussian_args(self):
        """Test that the gaussian plugin requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'wires'"):
            dev = qml.device("strawberryfields.gaussian")

    def test_gaussian_circuit(self, tol):
        """Test that the gaussian plugin provides correct result for simple circuit"""
        dev = qml.device("strawberryfields.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.NumberOperator(0))

        assert np.allclose(circuit(1), 1, atol=tol, rtol=0)

    def test_nonzero_shots(self):
        """Test that the gaussian plugin provides correct results for high shot number"""
        shots = 10 ** 2
        dev = qml.device("strawberryfields.gaussian", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.NumberOperator(0))

        x = 1

        runs = []
        for _ in range(100):
            runs.append(circuit(x))

        expected_var = np.sqrt(1 / shots)
        assert np.allclose(np.mean(runs), x, atol=expected_var)


class TestGates:
    """Tests the supported gates compared to the result from Strawberry
    Fields"""

    @pytest.mark.parametrize("gate_name,pennylane_gate", one_mode_single_real_parameter_gates)
    def test_one_mode_single_real_parameter_gates(self, gate_name, pennylane_gate, tol):
        """Test that gates that take a single real parameter and acts on one mode provide the correct result"""
        a = 0.312

        operation = pennylane_gate

        wires = [0]

        dev = qml.device("strawberryfields.gaussian", wires=2)

        sf_operation = dev._operation_map[gate_name]

        assert dev.supports_operation(gate_name)

        @qml.qnode(dev)
        def circuit(*args):
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            operation(*args, wires=wires)
            return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

        res = circuit(a)
        sf_res = SF_gate_reference(sf_operation, wires, a)
        assert np.allclose(res, sf_res, atol=tol, rtol=0)

    @pytest.mark.parametrize("gate_name,pennylane_gate", two_modes_single_real_parameter_gates)
    def test_two_modes_single_real_parameter_gates(self, gate_name, pennylane_gate, tol):
        """Test that gates that take a single real parameter and acts on two
        modes provide the correct result"""
        a = 0.312

        operation = pennylane_gate

        wires = [0, 1]

        dev = qml.device("strawberryfields.gaussian", wires=2)

        sf_operation = dev._operation_map[gate_name]

        assert dev.supports_operation(gate_name)

        @qml.qnode(dev)
        def circuit(*args):
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            operation(*args, wires=wires)
            return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

        res = circuit(a)
        sf_res = SF_gate_reference(sf_operation, wires, a)
        assert np.allclose(res, sf_res, atol=tol, rtol=0)

    def test_gaussian_state(self, tol):
        """Test that the GaussianState gate works correctly"""
        V = np.array([[0.5, 0], [0, 2]])
        r = np.array([0, 0])

        wires = [0]

        gate_name = "GaussianState"
        operation = qml.GaussianState

        dev = qml.device("strawberryfields.gaussian", wires=2)

        sf_operation = dev._operation_map[gate_name]

        assert dev.supports_operation(gate_name)

        @qml.qnode(dev)
        def circuit(*args):
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            operation(*args, wires=wires)
            return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

        res = circuit(V, r)
        sf_res = SF_gate_reference(sf_operation, wires, V, r)
        assert np.allclose(res, sf_res, atol=tol, rtol=0)

    def test_interferometer(self, tol):
        """Test that the Interferometer gate works correctly"""
        U = np.array(
            [
                [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
                [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
            ]
        )

        wires = [0, 1]

        gate_name = "InterferometerUnitary"
        operation = qml.InterferometerUnitary

        dev = qml.device("strawberryfields.gaussian", wires=2)

        sf_operation = dev._operation_map[gate_name]

        assert dev.supports_operation(gate_name)

        @qml.qnode(dev)
        def circuit(*args):
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            operation(*args, wires=wires)
            return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

        res = circuit(U)
        sf_res = SF_gate_reference(sf_operation, wires, U)
        assert np.allclose(res, sf_res, atol=tol, rtol=0)

    def test_displaced_squeezed_state(self, tol):
        """Test that the DisplacedSqueezedState gate works correctly"""
        a = 0.312
        b = 0.123
        c = 0.532
        d = 0.124

        wires = [0]

        gate_name = "DisplacedSqueezedState"
        operation = qml.DisplacedSqueezedState

        dev = qml.device("strawberryfields.gaussian", wires=2)

        sf_operation = dev._operation_map[gate_name]

        assert dev.supports_operation(gate_name)

        @qml.qnode(dev)
        def circuit(*args):
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            operation(*args, wires=wires)
            return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

        res = circuit(a, b, c, d)
        sf_res = SF_gate_reference(sf_operation, wires, a, b, c, d)
        assert np.allclose(res, sf_res, atol=tol, rtol=0)


class TestUnsupported:
    """Test that an error is raised for gates and observables that are
    unsupported."""

    @pytest.mark.parametrize(
        "gate_name,pennylane_gate", unsupported_one_mode_single_real_parameter_gates
    )
    def test_unsupported_one_mode_single_real_parameter_gates(self, gate_name, pennylane_gate, tol):
        """Test those unsupported gates for the gaussian simulator that take a
        single real parameter and act on one mode """
        dev = qml.device("strawberryfields.gaussian", wires=2)
        a = 0.312

        operation = pennylane_gate
        wires = [0]

        @qml.qnode(dev)
        def circuit():
            operation(a, wires=wires)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(
            qml.DeviceError,
            match="Gate {} not supported " "on device strawberryfields.gaussian".format(gate_name),
        ):
            circuit()

    def test_cross_kerr_unsupported(self):
        """Test that the CrossKerr gate is unsupported for the gaussian
        simulator"""
        dev = qml.device("strawberryfields.gaussian", wires=2)
        a = 0.312

        gate_name = "CrossKerr"
        operation = qml.CrossKerr
        wires = [0, 1]

        @qml.qnode(dev)
        def circuit():
            operation(a, wires=wires)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(
            qml.DeviceError,
            match="Gate {} not supported " "on device strawberryfields.gaussian".format(gate_name),
        ):
            circuit()

    def test_fock_state_unsupported(self, tol):
        """Test that the FockState gate is unsupported for the gaussian
        simulator"""
        dev = qml.device("strawberryfields.gaussian", wires=2)
        arg = 1
        wires = [0]

        gate_name = "FockState"
        operation = qml.FockState

        @qml.qnode(dev)
        def circuit():
            operation(arg, wires=wires)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(
            qml.DeviceError,
            match="Gate {} not supported " "on device strawberryfields.gaussian".format(gate_name),
        ):
            circuit()

    def test_fock_state_vector_unsupported(self):
        """Test that the FockStateVector gate is unsupported for the gaussian
        simulator"""
        dev = qml.device("strawberryfields.gaussian", wires=2)

        gate_name = "FockStateVector"
        operation = qml.FockStateVector
        arg = psi
        wires = [0]

        @qml.qnode(dev)
        def circuit():
            operation(arg, wires=wires)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(
            qml.DeviceError,
            match="Gate {} not supported " "on device strawberryfields.gaussian".format(gate_name),
        ):
            circuit()

    def test_fock_density_matrix_unsupported(self, tol):
        """Test that the FockDensityMatrix gate is unsupported for the gaussian
        simulator"""
        dm = np.outer(psi, psi.conj())

        wires = [0]

        gate_name = "FockDensityMatrix"
        operation = qml.FockDensityMatrix

        dev = qml.device("strawberryfields.gaussian", wires=2)

        @qml.qnode(dev)
        def circuit(*args):
            operation(*args, wires=wires)
            return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

        with pytest.raises(
            qml.DeviceError,
            match="Gate {} not supported " "on device strawberryfields.gaussian".format(gate_name),
        ):
            circuit(dm)

    def test_cat_state_unsupported(self):
        """Test that the CatState gate is unsupported for the gaussian
        simulator"""
        dev = qml.device("strawberryfields.gaussian", wires=2)
        a = 0.312
        b = 0.123
        c = 0.532

        gate_name = "CatState"
        operation = qml.CatState
        wires = [0]

        @qml.qnode(dev)
        def circuit():
            operation(a, b, c, wires=wires)
            return qml.expval(qml.NumberOperator(0))

        with pytest.raises(
            qml.DeviceError,
            match="Gate {} not supported " "on device strawberryfields.gaussian".format(gate_name),
        ):
            circuit()


class TestExpectation:
    """Test that all supported expectations work as expected when compared to
    the Strawberry Fields results"""

    def test_number_operator(self, tol):
        """Test that the expectation value of the NumberOperator observable
        yields the correct result"""
        num_wires = 2
        dev = qml.device("strawberryfields.gaussian", wires=num_wires)

        gate_name = "NumberOperator"
        assert dev.supports_observable(gate_name)

        op = qml.NumberOperator
        sf_expectation = dev._observable_map[gate_name]
        wires = [0]

        @qml.qnode(dev)
        def circuit(*args):
            qml.Displacement(0.1, 0, wires=0)
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            return qml.expval(op(*args, wires=wires))

        assert np.allclose(
            circuit(), SF_expectation_reference(sf_expectation, wires, num_wires), atol=tol, rtol=0
        )

    @pytest.mark.parametrize("wires", [[0, 1], [0, 1, 2]])
    def test_tensor_number_operator(self, wires, tol):
        """Test that the expectation value of the TensorN observable
        yields the correct result"""
        num_wires = 3
        dev = qml.device("strawberryfields.gaussian", wires=num_wires)

        gate_name = "TensorN"
        assert dev.supports_observable(gate_name)

        op = qml.TensorN
        sf_expectation = dev._observable_map[gate_name]

        @qml.qnode(dev)
        def circuit():
            qml.Displacement(0.1, 0, wires=0)
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            return qml.expval(op(wires=wires))

        assert np.allclose(
            circuit(), SF_expectation_reference(sf_expectation, wires, num_wires), atol=tol, rtol=0
        )

    @pytest.mark.parametrize("gate_name,op", [("X", qml.X), ("P", qml.P)])
    def test_quadrature(self, gate_name, op, tol):
        """Test that the expectation of the X and P quadrature operators yield
        the correct result"""
        num_wires = 2
        dev = qml.device("strawberryfields.gaussian", wires=num_wires)

        assert dev.supports_observable(gate_name)

        sf_expectation = dev._observable_map[gate_name]
        wires = [0]

        @qml.qnode(dev)
        def circuit(*args):
            qml.Displacement(0.1, 0, wires=0)
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            return qml.expval(op(*args, wires=wires))

        assert np.allclose(
            circuit(), SF_expectation_reference(sf_expectation, wires, num_wires), atol=tol, rtol=0
        )

    def test_quad_operator(self, tol):
        """Test that the expectation for the generalized quadrature observable
        yields the correct result"""
        a = 0.312
        num_wires = 2
        dev = qml.device("strawberryfields.gaussian", wires=num_wires)

        op = qml.QuadOperator
        gate_name = "QuadOperator"
        assert dev.supports_observable(gate_name)

        sf_expectation = dev._observable_map[gate_name]
        wires = [0]

        @qml.qnode(dev)
        def circuit(*args):
            qml.Displacement(0.1, 0, wires=0)
            qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
            return qml.expval(op(*args, wires=wires))

        assert np.allclose(
            circuit(a),
            SF_expectation_reference(sf_expectation, wires, num_wires, a),
            atol=tol,
            rtol=0,
        )

    def test_polyxp(self, tol):
        """Test that PolyXP works as expected"""
        a = 0.54321
        nbar = 0.5234

        hbar = 2
        dev = qml.device("strawberryfields.gaussian", wires=2)
        Q = np.array([0, 1, 0])  # x expectation

        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.PolyXP(Q, 0))

        # test X expectation
        assert np.allclose(circuit(a), hbar * a, atol=tol, rtol=0)

        Q = np.diag([-0.5, 1 / (2 * hbar), 1 / (2 * hbar)])  # mean photon number

        @qml.qnode(dev)
        def circuit(x):
            qml.ThermalState(nbar, wires=0)
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.PolyXP(Q, 0))

        # test X expectation
        assert np.allclose(circuit(a), nbar + np.abs(a) ** 2, atol=tol, rtol=0)

    def test_fock_state_projector(self, tol):
        """Test that FockStateProjector works as expected"""
        a = 0.54321
        r = 0.123

        hbar = 2
        dev = qml.device("strawberryfields.gaussian", wires=2, hbar=hbar)

        # test correct number state expectation |<n|a>|^2
        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.FockStateProjector(np.array([2]), wires=0))

        expected = np.abs(np.exp(-np.abs(a) ** 2 / 2) * a ** 2 / np.sqrt(2)) ** 2
        assert np.allclose(circuit(a), expected, atol=tol, rtol=0)

        # test correct number state expectation |<n|S(r)>|^2
        @qml.qnode(dev)
        def circuit(x):
            qml.Squeezing(x, 0, wires=0)
            return qml.expval(qml.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

        expected = np.abs(np.sqrt(2) / (2) * (-np.tanh(r)) / np.sqrt(np.cosh(r))) ** 2
        assert np.allclose(circuit(r), expected, atol=tol, rtol=0)

    def test_trace(self, tol):
        """Test that Identity expectation works as expected"""
        r1 = 0.5
        r2 = 0.7

        hbar = 2
        dev = qml.device("strawberryfields.gaussian", wires=2, hbar=hbar)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.Squeezing(x, 0, wires=0)
            qml.Squeezing(y, 0, wires=1)
            return qml.expval(qml.Identity(wires=[0])), qml.expval(qml.Identity(wires=[1]))

        assert np.allclose(circuit(r1, r2), 1, atol=tol, rtol=0)


class TestVariance:
    """Test for the device variance"""

    def test_first_order_cv(self, tol):
        """Test variance of a first order CV expectation value"""
        dev = qml.device("strawberryfields.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.X(0))

        r = 0.543
        phi = -0.654

        var = circuit(r, phi)
        expected = np.exp(2 * r) * np.sin(phi) ** 2 + np.exp(-2 * r) * np.cos(phi) ** 2
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradA = circuit.qtape.jacobian(dev, method="analytic")
        gradF = circuit.qtape.jacobian(dev, method="numeric")
        expected = np.array(
            [
                2 * np.exp(2 * r) * np.sin(phi) ** 2 - 2 * np.exp(-2 * r) * np.cos(phi) ** 2,
                2 * np.sinh(2 * r) * np.sin(2 * phi),
            ]
        )
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        assert np.allclose(gradF, expected, atol=tol, rtol=0)

    def test_second_order_cv(self, tol):
        """Test variance of a second order CV expectation value"""
        dev = qml.device("strawberryfields.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        n = 0.12
        a = 0.765

        var = circuit(n, a)
        expected = n ** 2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradF = circuit.qtape.jacobian(dev, method="numeric")
        expected = np.array([2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)])
        assert np.allclose(gradF, expected, atol=tol, rtol=0)


class TestProbability:
    """Integration tests for returning probabilities"""

    def test_single_mode_probability(self, tol):
        """Test that a coherent state returns the correct probability"""
        dev = qml.device("strawberryfields.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(a, phi):
            qml.Displacement(a, phi, wires=0)
            return qml.probs(wires=0)

        a = 0.4
        phi = -0.12
        cutoff = 10

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_probs = np.abs(np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))) ** 2

        res = circuit(a, phi)
        assert np.allclose(res, ref_probs, atol=tol, rtol=0)

    def test_multi_mode_probability(self, tol):
        """Test that a product of coherent states returns the correct probability"""
        dev = qml.device("strawberryfields.gaussian", wires=2)

        @qml.qnode(dev)
        def circuit(a, phi):
            qml.Displacement(a, phi, wires=0)
            qml.Displacement(a, phi, wires=1)
            return qml.probs(wires=[0, 1])

        a = 0.4
        phi = -0.12
        cutoff = 10

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_probs = np.abs(np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))) ** 2
        ref_probs = np.kron(ref_probs, ref_probs)

        res = circuit(a, phi)
        assert np.allclose(res, ref_probs, atol=tol, rtol=0)

    def test_marginal_probability(self, tol):
        """Test that a coherent state marginal probability is correct"""
        dev = qml.device("strawberryfields.gaussian", wires=2)

        @qml.qnode(dev)
        def circuit(a, phi):
            qml.Displacement(a, phi, wires=1)
            return qml.probs(wires=1)

        a = 0.4
        phi = -0.12
        cutoff = 10

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_probs = np.abs(np.exp(-0.5 * np.abs(alpha) ** 2) * alpha ** n / np.sqrt(fac(n))) ** 2

        res = circuit(a, phi)
        assert np.allclose(res, ref_probs, atol=tol, rtol=0)

    def test_finite_diff_coherent(self, tol):
        """Test that the jacobian of the probability for a coherent states is
        approximated well with finite differences"""
        cutoff = 10

        dev = qml.device("strawberryfields.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(a, phi):
            qml.Displacement(a, phi, wires=0)
            return qml.probs(wires=[0])

        a = 0.4
        phi = -0.12

        n = np.arange(cutoff)

        # construct tape
        circuit.construct([a, phi], {})

        # differentiate with respect to parameter a
        circuit.qtape.trainable_params = {0}
        res_F = circuit.qtape.jacobian(dev, method="numeric").flatten()
        expected_gradient = 2 * np.exp(-(a ** 2)) * a ** (2 * n - 1) * (n - a ** 2) / fac(n)
        assert np.allclose(res_F, expected_gradient, atol=tol, rtol=0)

        # re-construct tape to reset trainable_params
        circuit.construct([a, phi], {})

        # differentiate with respect to parameter phi
        circuit.qtape.trainable_params = {1}
        res_F = circuit.qtape.jacobian(dev, method="numeric").flatten()
        expected_gradient = 0
        assert np.allclose(res_F, expected_gradient, atol=tol, rtol=0)

    def test_finite_diff_squeezed(self, tol):
        """Test that the jacobian of the probability for a squeezed states is
        approximated well with finite differences"""
        cutoff = 5
        # import pdb; pdb.set_trace()

        dev = qml.device("strawberryfields.gaussian", wires=1, cutoff_dim=cutoff)

        @qml.qnode(dev)
        def circuit(r, phi):
            qml.Squeezing(r, phi, wires=0)
            return qml.probs(wires=[0])

        r = 0.4
        phi = -0.12

        n = np.arange(cutoff)

        # construct tape
        circuit.construct([r, phi], {})

        # differentiate with respect to parameter r
        circuit.qtape.trainable_params = {0}
        res_F = circuit.qtape.jacobian(dev, method="numeric").flatten()
        assert res_F.shape == (cutoff,)

        expected_gradient = (
            np.abs(np.tanh(r)) ** n
            * (1 + 2 * n - np.cosh(2 * r))
            * fac(n)
            / (2 ** (n + 1) * np.cosh(r) ** 2 * np.sinh(r) * fac(n / 2) ** 2)
        )
        expected_gradient[n % 2 != 0] = 0
        assert np.allclose(res_F, expected_gradient, atol=tol, rtol=0)

        # re-construct tape to reset trainable_params
        circuit.construct([r, phi], {})

        # differentiate with respect to parameter phi
        circuit.qtape.trainable_params = {1}
        res_F = circuit.qtape.jacobian(dev, method="numeric").flatten()
        expected_gradient = 0
        assert np.allclose(res_F, expected_gradient, atol=tol, rtol=0)

    def test_finite_diff_coherent_two_wires(self, tol):
        """Test that the jacobian of the probability for a coherent states is approximated well with
        finite differences"""
        cutoff = 4

        dev = qml.device("strawberryfields.gaussian", wires=2, cutoff_dim=cutoff)

        @qml.qnode(dev, diff_method="finite-diff")
        def circuit(a, phi):
            qml.Displacement(a, phi, wires=0)
            qml.Displacement(a, phi, wires=1)
            return qml.probs(wires=[0, 1])

        a = np.array(0.4, requires_grad=True)
        phi = np.array(-0.12, requires_grad=False)

        c = np.arange(cutoff)
        d = np.arange(cutoff)
        n0, n1 = np.meshgrid(c, d)
        n0 = n0.flatten()
        n1 = n1.flatten()

        # differentiate with respect to parameter a
        res_F = qml.jacobian(circuit)(a, phi)
        expected_gradient = (
            2
            * (a ** (-1 + 2 * n0 + 2 * n1))
            * np.exp(-2 * a ** 2)
            * (-2 * a ** 2 + n0 + n1)
            / (fac(n0) * fac(n1))
        )
        assert np.allclose(res_F, expected_gradient, atol=tol, rtol=0)

        # differentiate with respect to parameter phi
        a = np.array(0.4, requires_grad=False)
        phi = np.array(-0.12, requires_grad=True)

        res_F = qml.jacobian(circuit)(a, phi)
        expected_gradient = 0
        assert np.allclose(res_F, expected_gradient, atol=tol, rtol=0)

    def test_analytic_diff_error(self, tol):
        """Test that the analytic gradients are not supported when returning
        Fock state probabilities."""
        dev = qml.device("strawberryfields.gaussian", wires=1)

        @qml.qnode(dev)
        def circuit(a, phi):
            qml.Displacement(a, phi, wires=0)
            return qml.probs(wires=[0])

        a = 0.4
        phi = -0.12

        # construct tape
        circuit.construct([a, phi], {})

        with pytest.raises(ValueError, match="The analytic gradient method cannot be used with"):
            _ = circuit.qtape.jacobian(dev, method="analytic").flatten()

    def test_tensorn_one_mode_is_mean_photon(self, tol):
        """Test variance of TensorN for a single mode, which resorts to
        calculations for the NumberOperator"""
        dev = qml.device("strawberryfields.gaussian", wires=1)

        op = qml.TensorN(wires=[0])

        # Check that instantiating TensorN on one mode returns the
        # NumberOperator
        assert isinstance(op, qml.NumberOperator)

        @qml.qnode(dev)
        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(op)

        n = 0.12
        a = 0.765

        var = circuit(n, a)
        expected = n ** 2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradF = circuit.qtape.jacobian(dev, method="numeric")
        expected = np.array([2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)])
        assert np.allclose(gradF, expected, atol=tol, rtol=0)
