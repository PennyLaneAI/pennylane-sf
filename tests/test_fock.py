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
Unit tests for the Fock plugin.
"""
import pytest
import unittest
import logging as log
log.getLogger()

import strawberryfields as sf

import pennylane as qml
from pennylane import numpy as np

from defaults import BaseTest


psi = np.array([ 0.08820314+0.14909648j,  0.32826940+0.32956027j,
        0.26695166+0.19138087j,  0.32419593+0.08460371j,
        0.02984712+0.30655538j,  0.03815006+0.18297214j,
        0.17330397+0.2494433j ,  0.14293477+0.25095202j,
        0.21021125+0.30082734j,  0.23443833+0.19584968j])


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == 'A':
        return [np.diag([x, 1]) for x in par]
    return par


class FockTests(BaseTest):
    """Test the Fock simulator."""

    def test_load_fock_device(self):
        """Test that the fock plugin loads correctly"""
        self.logTestName()

        dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=5)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.cutoff, 5)
        self.assertEqual(dev.hbar, 2)
        self.assertEqual(dev.shots, 1000)
        self.assertEqual(dev.short_name, 'strawberryfields.fock')

    def test_fock_args(self):
        """Test that the fock plugin requires correct arguments"""
        self.logTestName()

        with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'wires'"):
            dev = qml.device('strawberryfields.fock')
        with self.assertRaisesRegex(TypeError, "missing 1 required keyword-only argument: 'cutoff_dim'"):
            dev = qml.device('strawberryfields.fock', wires=1)

    def test_unsupported_gates(self):
        """Test error is raised with unsupported gates"""
        self.logTestName()

        dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=2)
        gates = set(dev._operation_map.keys())
        all_gates = qml.ops._cv__ops__

        for g in all_gates - gates:
            op = getattr(qml.ops, g)

            if op.num_wires is qml.operation.Wires.Any or qml.operation.Wires.All:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*args):
                args = prep_par(args, op)
                op(*args, wires=wires)

                if issubclass(op, qml.operation.CV):
                    return qml.expval(qml.NumberOperator(0))
                else:
                    return qml.expval(qml.PauliZ(0))

            with self.assertRaisesRegex(qml.DeviceError,
                "Gate {} not supported on device strawberryfields.fock".format(g)):
                args = np.random.random([op.num_params])
                circuit(*args)

    def test_unsupported_expectations(self):
        """Test error is raised with unsupported expectations"""
        self.logTestName()

        dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=2)
        obs = set(dev._observable_map.keys())
        all_obs = set(qml.ops._cv__obs__)

        for g in all_obs - obs:
            op = getattr(qml.ops, g)

            if op.num_wires is qml.operation.Wires.Any or qml.operation.Wires.All:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*args):
                args = prep_par(args, op)
                return qml.expval(op(*args, wires=wires))

            with self.assertRaisesRegex(qml.DeviceError,
                "Expectation {} not supported on device strawberryfields.fock".format(g)):
                args = np.random.random([op.num_params])
                circuit(*args)

    def test_fock_circuit(self):
        """Test that the fock plugin provides correct result for simple circuit"""
        self.logTestName()

        dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=10)

        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.NumberOperator(0))

        self.assertAlmostEqual(circuit(1), 1, delta=self.tol)

    def test_nonzero_shots(self):
        """Test that the fock plugin provides correct result for high shot number"""
        self.logTestName()

        shots = 10**2
        dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=10, shots=shots)

        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.NumberOperator(0))

        x = 1

        runs = []
        for _ in range(100):
            runs.append(circuit(x))

        expected_var = np.sqrt(1/shots)
        self.assertAlmostEqual(np.mean(runs), x, delta=expected_var)

    def test_supported_fock_gates(self):
        """Test that all supported gates work correctly"""
        self.logTestName()
        cutoff_dim = 10
        a = 0.312
        b = 0.123
        c = 0.532
        d = 0.124

        dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=cutoff_dim)

        gates = list(dev._operation_map.items())
        for g, sfop in gates:
            log.info('\tTesting gate {}...'.format(g))
            self.assertTrue(dev.supports_operation(g))

            op = getattr(qml.ops, g)
            if op.num_wires <= 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*args):
                qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
                op(*args, wires=wires)
                return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

            # compare to reference SF engine
            def SF_reference(*args):
                """SF reference circuit"""
                eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
                prog = sf.Program(2)
                with prog.context as q:
                    sf.ops.S2gate(0.1) | q
                    sfop(*args) | [q[i] for i in wires]

                state = eng.run(prog).state
                return state.mean_photon(0)[0], state.mean_photon(1)[0]

            if g == 'GaussianState':
                r = np.array([0, 0])
                V = np.array([[0.5, 0], [0, 2]])
                self.assertAllEqual(circuit(V, r), SF_reference(V, r))
            elif g == 'FockDensityMatrix':
                dm = np.outer(psi, psi.conj())
                self.assertAllEqual(circuit(dm), SF_reference(dm))
            elif g == 'FockStateVector':
                self.assertAllEqual(circuit(psi), SF_reference(psi))
            elif g == 'FockState':
                self.assertAllEqual(circuit(1), SF_reference(1))
            elif g == "DisplacedSqueezedState":
                self.assertAllEqual(circuit(a, b, c, d),
                                    SF_reference(a*np.exp(1j*b), c, d))
            elif g == "CatState":
                self.assertAllEqual(circuit(a, b, c),
                                    SF_reference(a*np.exp(1j*b), c))
            elif op.num_params == 1:
                self.assertAllEqual(circuit(a), SF_reference(a))
            elif op.num_params == 2:
                self.assertAllEqual(circuit(a, b), SF_reference(a, b))

    def test_supported_fock_expectations(self):
        """Test that all supported expectations work correctly"""
        self.logTestName()
        cutoff_dim = 10
        a = 0.312
        a_array = np.eye(3)

        dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=cutoff_dim)

        expectations = list(dev._observable_map.items())
        for g, sfop in expectations:
            log.info('\tTesting expectation {}...'.format(g))
            self.assertTrue(dev.supports_observable(g))

            op = getattr(qml.ops, g)
            if op.num_wires <= 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qml.qnode(dev)
            def circuit(*args):
                qml.Displacement(0.1, 0, wires=0)
                qml.TwoModeSqueezing(0.1, 0, wires=[0, 1])
                return qml.expval(op(*args, wires=wires))

            # compare to reference SF engine
            def SF_reference(*args):
                """SF reference circuit"""
                eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
                prog = sf.Program(2)
                with prog.context as q:
                    sf.ops.Xgate(0.2) | q[0]
                    sf.ops.S2gate(0.1) | q

                state = eng.run(prog).state
                return sfop(state, wires, args)[0]

            if op.num_params == 0:
                self.assertAllEqual(circuit(), SF_reference())
            elif op.num_params == 1:
                if g == 'FockStateProjector':
                    p = np.array([1])
                else:
                    p = a_array if op.par_domain == 'A' else a
                self.assertAllEqual(circuit(p), SF_reference(p))

    def test_polyxp(self):
        """Test that PolyXP works as expected"""
        self.logTestName()

        cutoff_dim = 12
        a = 0.14321
        nbar = 0.2234

        hbar = 2
        dev = qml.device('strawberryfields.fock', wires=1, hbar=hbar, cutoff_dim=cutoff_dim)
        Q = np.array([0, 1, 0]) # x expectation

        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.PolyXP(Q, 0))

        # test X expectation
        self.assertAlmostEqual(circuit(a), hbar*a)

        Q = np.diag([-0.5, 1/(2*hbar), 1/(2*hbar)]) # mean photon number

        @qml.qnode(dev)
        def circuit(x):
            qml.ThermalState(nbar, wires=0)
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.PolyXP(Q, 0))

        # test X expectation
        self.assertAlmostEqual(circuit(a), nbar+np.abs(a)**2)

    def test_fock_state(self):
        """Test that FockStateProjector works as expected"""
        self.logTestName()

        cutoff_dim = 12
        a = 0.54321
        r = 0.123

        hbar = 2
        dev = qml.device('strawberryfields.fock', wires=2, hbar=hbar, cutoff_dim=cutoff_dim)

        # test correct number state expectation |<n|a>|^2
        @qml.qnode(dev)
        def circuit(x):
            qml.Displacement(x, 0, wires=0)
            return qml.expval(qml.FockStateProjector(np.array([2]), wires=0))

        expected = np.abs(np.exp(-np.abs(a)**2/2)*a**2/np.sqrt(2))**2
        self.assertAlmostEqual(circuit(a), expected)

        # test correct number state expectation |<n|S(r)>|^2
        @qml.qnode(dev)
        def circuit(x):
            qml.Squeezing(x, 0, wires=0)
            return qml.expval(qml.FockStateProjector(np.array([2, 0]), wires=[0, 1]))

        expected = np.abs(np.sqrt(2)/(2)*(-np.tanh(r))/np.sqrt(np.cosh(r)))**2
        self.assertAlmostEqual(circuit(r), expected)

    def test_trace(self):
        """Test that Identity expectation works as expected"""
        self.logTestName()

        cutoff_dim = 5
        r1 = 0.5
        r2 = 0.7

        hbar = 2
        dev = qml.device('strawberryfields.fock', wires=2, hbar=hbar, cutoff_dim=cutoff_dim)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.Squeezing(x, 0, wires=0)
            qml.Squeezing(y, 0, wires=1)
            return qml.expval(qml.Identity(wires=[0, 1]))

        # reference SF circuit
        def SF_reference_trace(x, y):
            """SF reference circuit"""
            eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
            prog = sf.Program(2)
            with prog.context as q:
                sf.ops.Sgate(x) | q[0]
                sf.ops.Sgate(y) | q[1]

            state = eng.run(prog).state
            return state.trace()

        # test trace < 1 for high squeezing
        expected = SF_reference_trace(r1, r2)
        self.assertAlmostEqual(circuit(r1, r2), expected)


class TestVariance:
    """Test for the device variance"""

    def test_first_order_cv(self, tol):
        """Test variance of a first order CV expectation value"""
        dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=15)

        @qml.qnode(dev)
        def circuit(r, phi):
            qml.Squeezing(r, 0, wires=0)
            qml.Rotation(phi, wires=0)
            return qml.var(qml.X(0))

        r = 0.105
        phi = -0.654

        var = circuit(r, phi)
        expected = np.exp(2*r)*np.sin(phi)**2 + np.exp(-2*r)*np.cos(phi)**2
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradA = circuit.jacobian([r, phi], method='A')
        gradF = circuit.jacobian([r, phi], method='F')
        expected = np.array([
            2*np.exp(2*r)*np.sin(phi)**2 - 2*np.exp(-2*r)*np.cos(phi)**2,
            2*np.sinh(2*r)*np.sin(2*phi)
        ])
        assert np.allclose(gradA, expected, atol=tol, rtol=0)
        assert np.allclose(gradF, expected, atol=tol, rtol=0)

    def test_second_order_cv(self, tol):
        """Test variance of a second order CV expectation value"""
        dev = qml.device('strawberryfields.fock', wires=1, cutoff_dim=15)

        @qml.qnode(dev)
        def circuit(n, a):
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            return qml.var(qml.NumberOperator(0))

        n = 0.12
        a = 0.105

        var = circuit(n, a)
        expected = n ** 2 + n + np.abs(a) ** 2 * (1 + 2 * n)
        assert np.allclose(var, expected, atol=tol, rtol=0)

        # circuit jacobians
        gradF = circuit.jacobian([n, a], method='F')
        expected = np.array([2*a**2+2*n+1, 2*a*(2*n+1)])
        assert np.allclose(gradF, expected, atol=tol, rtol=0)
