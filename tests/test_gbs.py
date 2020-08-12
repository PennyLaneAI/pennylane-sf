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
from pennylane.qnodes.base import ParameterDependency
from pennylane.operation import Probability

from pennylane_sf import StrawberryFieldsGBS
from pennylane_sf.ops import ParamGraphEmbed
from pennylane_sf.simulator import StrawberryFieldsSimulator


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


samples = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 0],
    [2, 0, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 0],
    [0, 0, 2, 0],
    [0, 1, 2, 0],
    [2, 2, 2, 0],
])

probs_dict = {
    (0, 0, 0, 0): 0.3,
    (0, 0, 1, 1): 0.1,
    (0, 0, 2, 0): 0.2,
    (0, 1, 0, 0): 0.1,
    (0, 1, 2, 0): 0.1,
    (2, 0, 0, 0): 0.1,
    (2, 2, 2, 0): 0.1,
}

probs_dict_subset = {
    (0, 0): 0.4,
    (0, 1): 0.1,
    (0, 2): 0.3,
    (2, 0): 0.1,
    (2, 2): 0.1,
}


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

    def test_apply_wrong_dim(self):
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        op = "ParamGraphEmbed"
        wires = list(range(4))

        A = 0.1767767 * np.ones((4, 4))
        params = np.ones(3)
        n_mean = 1
        par = [params, A, n_mean]

        with pytest.raises(ValueError, match="The number of variable parameters must be"):
            dev.apply(op, wires, par)

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

    # def test_pre_measure_state_and_samples_non_analytic(self, tol):
    #     """Test that the pre_measure method operates as expected in non-analytic mode by
    #     generating the correct output state and samples of the right shape"""
    #     dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3, analytic=False, shots=2)
    #     prog = Program(4)
    #     op1 = GraphEmbed(0.1767767 * np.ones((4, 4)), mean_photon_per_mode=0.25)
    #     op2 = MeasureFock()
    #     prog.append(op1, prog.register)
    #     prog.append(op2, prog.register)
    #     dev.prog = prog
    #     dev.pre_measure()
    #
    #     assert np.allclose(dev.state.displacement(), np.zeros(4))
    #     assert np.allclose(dev.state.cov(), target_cov, atol=tol)
    #     assert dev.samples.shape == (2, 4)

    def test_probability_analytic(self, monkeypatch):
        """Test that the probability method in analytic mode simply calls the parent method in
        StrawberryFieldsSimulator. The test monkeypatches StrawberryFieldsSimulator.probability() to
        just return True."""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        with monkeypatch.context() as m:
            m.setattr(StrawberryFieldsSimulator, "probability", lambda *args, **kwargs: True)
            p = dev.probability()
        assert p

    def test_probability_non_analytic_all_wires(self):
        """Test that the probability method in non-analytic mode returns the expected dictionary
        mapping from samples to probabilities when a fixed set of pre-generated samples are
        input. The lexicographic order of the keys in the returned dictionary is also checked."""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3, analytic=False)

        dev.samples = samples
        dev_probs_dict = dev.probability()

        for sample, prob in probs_dict.items():
            assert dev_probs_dict[sample] == prob

        keys = iter(dev_probs_dict)
        assert next(keys) == (0, 0, 0, 0)
        assert next(keys) == (0, 0, 0, 1)
        assert next(keys) == (0, 0, 0, 2)
        assert next(keys) == (0, 0, 1, 0)

    def test_probability_non_analytic_subset_wires(self):
        """Test that the probability method in non-analytic mode returns the expected dictionary
        mapping from samples to probabilities when a fixed set of pre-generated samples are
        input and a subset of wires are requested. The lexicographic order of the keys in the
        returned dictionary is also checked."""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3, analytic=False)

        dev.samples = samples
        dev_probs_dict = dev.probability(wires=[0, 2])

        for sample, prob in probs_dict_subset.items():
            assert dev_probs_dict[sample] == prob

        keys = iter(dev_probs_dict)
        assert next(keys) == (0, 0)
        assert next(keys) == (0, 1)
        assert next(keys) == (0, 2)
        assert next(keys) == (1, 0)

    def test_calculate_covariance(self):
        """Test that the _calculate_covariance method returns the correct covariance matrix for a
        fixed example."""
        x = np.sqrt(0.5)
        A = np.array([[0, x], [x, 0]])
        cov = StrawberryFieldsGBS._calculate_covariance(A, 2)
        target = np.array(
            [
                [3.0, 0.0, 0.0, 2.82842712],
                [0.0, 3.0, 2.82842712, 0.0],
                [0.0, 2.82842712, 3.0, 0.0],
                [2.82842712, 0.0, 0.0, 3.0],
            ]
        )
        assert np.allclose(cov, target)

    def test_jacobian_all_wires(self):
        """Test that the _jacobian_all_wires method returns the correct jacobian for a fixed
        input probability distribution"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        dev._WAW = 0.1767767 * np.ones((4, 4))
        params = np.array([1, 2, 3, 4])
        n_mean = np.ones(4) / 4
        dev._params = params

        indices = np.indices([3, 3, 3, 3]).reshape(4, -1).T
        probs = np.arange(3 ** 4)
        jac_expected = np.zeros((3 ** 4, 4))

        for i, ind in enumerate(indices):
            jac_expected[i] = probs[i] * (ind - n_mean) / params

        jac = dev._jacobian_all_wires(probs)
        assert np.allclose(jac, jac_expected)

    def test_jacobian(self):
        """Test that the jacobian method returns correctly on a fixed example"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)

        A = 0.1767767 * np.ones((4, 4))
        params = np.ones(4)
        op = [ParamGraphEmbed(params, A, 1, wires=range(4))]

        ob = qml.Identity(wires=range(4))
        ob.return_type = Probability
        obs = [ob]

        variable_deps = {i: ParameterDependency(op, i) for i in range(4)}

        jac = dev.jacobian(op, obs, variable_deps)
        probs = dev.probability()

        jac_expected = np.zeros((81, 4))

        for i, (sample, prob) in enumerate(probs.items()):
            jac_expected[i] = (np.array(sample) - 0.25) * prob

        assert np.allclose(jac, jac_expected)

    def test_jacobian_subset_wires(self, monkeypatch):
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        A = 0.1767767 * np.ones((4, 4))
        dev._WAW = A
        params = np.array([1, 2, 3, 4])
        n_mean = np.ones(4) / 4
        dev._params = params

        indices = np.indices([3, 3, 3, 3]).reshape(4, -1).T
        probs = np.arange(3 ** 4)
        jac_expected = np.zeros((3 ** 4, 4))

        for i, ind in enumerate(indices):
            jac_expected[i] = probs[i] * (ind - n_mean) / params

        jac_expected = np.array(np.split(jac_expected, 9))
        jac_expected = np.sum(jac_expected, axis=0)

        op = [ParamGraphEmbed(params, A, 1, wires=range(4))]

        ob = qml.Identity(wires=range(4))
        ob.return_type = Probability
        obs = [ob]
        variable_deps = {i: ParameterDependency(op, i) for i in range(4)}

        # with monkeypatch.context() as m:
        #     m.setattr(StrawberryFieldsSimulator, "probability", lambda *args, **kwargs: probs)
        jac = dev.jacobian(op, obs, variable_deps)

        # print(indices)
        # indices = (0, 2)
        # other_indices = (1, 3)
        # jac_traced_expected = np.sum(jac_expected.reshape(3, 3, 3, 3, 4), axis=other_indices)
        # print(jac_traced_expected)
