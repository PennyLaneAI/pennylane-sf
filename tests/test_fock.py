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
import inspect
import unittest
import logging as log
log.getLogger()

import strawberryfields as sf

import openqml as qm
from openqml import numpy as np

from defaults import openqml_sf as qmsf, BaseTest


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

        dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=5)
        self.assertEqual(dev.num_wires, 2)
        self.assertEqual(dev.cutoff, 5)
        self.assertEqual(dev.hbar, 2)
        self.assertEqual(dev.shots, 0)
        self.assertEqual(dev.short_name, 'strawberryfields.fock')

    def test_fock_args(self):
        """Test that the fock plugin requires correct arguments"""
        self.logTestName()

        with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'wires'"):
            dev = qm.device('strawberryfields.fock')
        with self.assertRaisesRegex(TypeError, "missing 1 required keyword-only argument: 'cutoff_dim'"):
            dev = qm.device('strawberryfields.fock', wires=1)

    def test_unsupported_gates(self):
        """Test error is raised with unsupported gates"""
        self.logTestName()

        dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=2)
        gates = set(dev._operation_map.keys())
        all_gates = {m[0] for m in inspect.getmembers(qm.ops, inspect.isclass)}

        for g in all_gates - gates:
            op = getattr(qm.ops, g)

            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qm.qnode(dev)
            def circuit(*args):
                args = prep_par(args, op)
                op(*args, wires=wires)

                if issubclass(op, qm.operation.CV):
                    return qm.expval.PhotonNumber(0)
                else:
                    return qm.expval.PauliZ(0)

            with self.assertRaisesRegex(qm.DeviceError,
                "Gate {} not supported on device strawberryfields.fock".format(g)):
                args = np.random.random([op.num_params])
                circuit(*args)

    def test_unsupported_expectations(self):
        """Test error is raised with unsupported expectations"""
        self.logTestName()

        dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=2)
        obs = set(dev._expectation_map.keys())
        all_obs = {m[0] for m in inspect.getmembers(qm.expval, inspect.isclass)}

        for g in all_obs - obs:
            op = getattr(qm.expval, g)

            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qm.qnode(dev)
            def circuit(*args):
                args = prep_par(args, op)
                return op(*args, wires=wires)

            with self.assertRaisesRegex(qm.DeviceError,
                "Expectation {} not supported on device strawberryfields.fock".format(g)):
                args = np.random.random([op.num_params])
                circuit(*args)

    def test_fock_circuit(self):
        """Test that the fock plugin provides correct result for simple circuit"""
        self.logTestName()

        dev = qm.device('strawberryfields.fock', wires=1, cutoff_dim=10)

        @qm.qnode(dev)
        def circuit(x):
            qm.Displacement(x, 0, wires=0)
            return qm.expval.PhotonNumber(0)

        self.assertAlmostEqual(circuit(1), 1, delta=self.tol)

    def test_nonzero_shots(self):
        """Test that the fock plugin provides correct result for high shot number"""
        self.logTestName()

        shots = 10**2
        dev = qm.device('strawberryfields.fock', wires=1, cutoff_dim=10, shots=shots)

        @qm.qnode(dev)
        def circuit(x):
            qm.Displacement(x, 0, wires=0)
            return qm.expval.PhotonNumber(0)

        expected_var = np.sqrt(1/shots)
        self.assertAlmostEqual(circuit(1), 1, delta=expected_var)

    def test_supported_fock_gates(self):
        """Test that all supported gates work correctly"""
        self.logTestName()
        cutoff_dim = 10
        a = 0.312
        b = 0.123

        dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=cutoff_dim)

        gates = list(dev._operation_map.items())
        for g, sfop in gates:
            log.info('\tTesting gate {}...'.format(g))
            self.assertTrue(dev.supported(g))

            op = getattr(qm.ops, g)
            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qm.qnode(dev)
            def circuit(*args):
                qm.TwoModeSqueezing(0.1, 0, wires=[0, 1])
                op(*args, wires=wires)
                return qm.expval.PhotonNumber(0), qm.expval.PhotonNumber(1)

            # compare to reference SF engine
            def SF_reference(*args):
                """SF reference circuit"""
                eng, q = sf.Engine(2)
                with eng:
                    sf.ops.S2gate(0.1) | q
                    sfop(*args) | [q[i] for i in wires]

                state = eng.run('fock', cutoff_dim=cutoff_dim)
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

        dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=cutoff_dim)

        expectations = list(dev._expectation_map.items())
        for g, sfop in expectations:
            log.info('\tTesting expectation {}...'.format(g))
            self.assertTrue(dev.supported(g))

            op = getattr(qm.expval, g)
            if op.num_wires == 0:
                wires = [0]
            else:
                wires = list(range(op.num_wires))

            @qm.qnode(dev)
            def circuit(*args):
                qm.Displacement(0.1, 0, wires=0)
                qm.TwoModeSqueezing(0.1, 0, wires=[0, 1])
                return op(*args, wires=wires)

            # compare to reference SF engine
            def SF_reference(*args):
                """SF reference circuit"""
                eng, q = sf.Engine(2)
                with eng:
                    sf.ops.Xgate(0.2) | q[0]
                    sf.ops.S2gate(0.1) | q

                state = eng.run('fock', cutoff_dim=cutoff_dim)
                return sfop(state, wires, args)[0]

            if op.num_params == 0:
                self.assertAllEqual(circuit(), SF_reference())
            elif op.num_params == 1:
                p = a_array if op.par_domain == 'A' else a
                self.assertAllEqual(circuit(p), SF_reference(p))


if __name__ == '__main__':
    # run the tests in this file
    suite = unittest.TestSuite()
    for t in (FockTests,):
        ttt = unittest.TestLoader().loadTestsFromTestCase(t)
        suite.addTests(ttt)

    unittest.TextTestRunner().run(suite)
