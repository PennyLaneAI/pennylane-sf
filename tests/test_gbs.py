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
from pennylane import numpy as np
from pennylane.operation import Probability
from pennylane.qnodes.base import ParameterDependency
from pennylane.wires import Wires
from strawberryfields.ops import GraphEmbed, MeasureFock
from strawberryfields.program import Program

from pennylane_sf import StrawberryFieldsGBS
from pennylane_sf.ops import ParamGraphEmbed
from pennylane_sf.simulator import StrawberryFieldsSimulator

target_cov = np.array(
    [
        [
            2.2071e00,
            1.2071e00,
            1.2071e00,
            1.2071e00,
            -2.6021e-18,
            -1.3878e-16,
            6.9323e-17,
            -7.8505e-17,
        ],
        [
            1.2071e00,
            2.2071e00,
            1.2071e00,
            1.2071e00,
            -2.7756e-17,
            -1.1102e-16,
            5.5511e-17,
            -1.1102e-16,
        ],
        [
            1.2071e00,
            1.2071e00,
            2.2071e00,
            1.2071e00,
            -5.3141e-17,
            -1.6653e-16,
            1.0127e-17,
            -1.3770e-16,
        ],
        [
            1.2071e00,
            1.2071e00,
            1.2071e00,
            2.2071e00,
            -7.8505e-17,
            -2.2204e-16,
            -1.5236e-17,
            -1.6306e-16,
        ],
        [
            -2.6021e-18,
            -2.7756e-17,
            -5.3141e-17,
            -7.8505e-17,
            7.9289e-01,
            -2.0711e-01,
            -2.0711e-01,
            -2.0711e-01,
        ],
        [
            -1.3878e-16,
            -1.1102e-16,
            -1.6653e-16,
            -2.2204e-16,
            -2.0711e-01,
            7.9289e-01,
            -2.0711e-01,
            -2.0711e-01,
        ],
        [
            6.9323e-17,
            5.5511e-17,
            1.0127e-17,
            -1.5236e-17,
            -2.0711e-01,
            -2.0711e-01,
            7.9289e-01,
            -2.0711e-01,
        ],
        [
            -7.8505e-17,
            -1.1102e-16,
            -1.3770e-16,
            -1.6306e-16,
            -2.0711e-01,
            -2.0711e-01,
            -2.0711e-01,
            7.9289e-01,
        ],
    ]
)


samples = np.array(
    [
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
    ]
)

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

A = np.array(
    [[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
)

jac_exp = np.array(
    [
        [-1.766373e-01, -7.129473e-02, -6.295893e-02, -2.073651e-02],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-4.510550e-03, 4.569179e-02, 3.798593e-02, -5.295200e-04],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-1.151800e-04, 2.380030e-03, 1.981050e-03, -1.352000e-05],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [7.542846e-02, -1.517130e-03, -1.339750e-03, 1.935555e-02],
        [0, 0, 0, 0],
        [4.525708e-02, -9.102800e-04, 1.899296e-02, -2.647600e-04],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [3.771423e-02, 1.903825e-02, -6.698700e-04, -2.206300e-04],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.926120e-03, 9.723100e-04, 8.083300e-04, 4.942600e-04],
        [0, 0, 0, 0],
        [2.311340e-03, 1.166770e-03, 1.981050e-03, -1.352000e-05],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.926120e-03, 1.983360e-03, 8.083300e-04, -1.127000e-05],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [4.918000e-05, 5.065000e-05, 4.216000e-05, 1.262000e-05],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [3.290180e-03, -3.228000e-05, -2.851000e-05, 8.331500e-04],
        [0, 0, 0, 0],
        [3.948220e-03, -3.874000e-05, 8.083300e-04, 4.942600e-04],
        [0, 0, 0, 0],
        [1.184470e-03, -1.162000e-05, 4.952600e-04, -3.380000e-06],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [3.290180e-03, 8.102600e-04, -2.851000e-05, 4.118800e-04],
        [0, 0, 0, 0],
        [1.974110e-03, 4.861500e-04, 4.041700e-04, -5.630000e-06],
        [0, 0, 0, 0],
        [8.402000e-05, 2.069000e-05, 1.720000e-05, 2.128000e-05],
        [0, 0, 0, 0],
        [2.016400e-04, 4.966000e-05, 8.431000e-05, 2.524000e-05],
        [0, 0, 0, 0],
        [8.225500e-04, 4.132000e-04, -7.130000e-06, -2.350000e-06],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.680300e-04, 8.441000e-05, 3.440000e-05, 2.104000e-05],
        [0, 0, 0, 0],
        [2.016400e-04, 1.012900e-04, 8.431000e-05, -5.800000e-07],
        [0, 0, 0, 0],
        [2.150000e-06, 1.080000e-06, 9.000000e-07, 5.400000e-07],
    ]
)

jac_reduced = {
    (0, 2): np.array(
        [
            [-1.76637300e-01, -7.12947257e-02, -6.29589339e-02, -2.07365140e-02],
            [-4.51055487e-03, 4.56917936e-02, 3.79859282e-02, -5.29521141e-04],
            [-1.15180119e-04, 2.38003197e-03, 1.98104730e-03, -1.35216863e-05],
            [1.13142690e-01, 1.75211142e-02, -2.00962457e-03, 1.91349125e-02],
            [4.91093112e-02, 2.04538952e-03, 2.06096254e-02, 2.18228539e-04],
            [2.36052583e-03, 1.21741782e-03, 2.02320347e-03, -9.00489456e-07],
            [7.40290986e-03, 1.19117368e-03, -6.41464311e-05, 1.24268550e-03],
            [6.17437908e-03, 5.52514954e-04, 1.26409919e-03, 5.30933583e-04],
            [1.58989291e-03, 1.40405534e-04, 6.64783578e-04, 2.18297708e-05],
        ]
    ),
    (2, 3): np.array(
        [
            [-1.38100524e-01, -5.18432783e-02, -6.36359361e-02, -2.09594954e-02],
            [7.87186422e-02, -7.06875303e-04, -1.36825924e-03, 1.97674273e-02],
            [3.29018216e-03, -3.22841991e-05, -2.85095249e-05, 8.33152028e-04],
            [4.46467480e-02, 4.72510284e-02, 5.81913883e-02, -8.11183819e-04],
            [6.04237030e-03, 1.01797920e-03, 1.65106345e-03, 1.00954969e-03],
            [8.40170632e-05, 2.06904922e-05, 1.72010659e-05, 2.12751098e-05],
            [3.58226750e-03, 3.63647394e-03, 4.54166876e-03, -3.09992702e-05],
            [2.50825688e-04, 1.00303642e-04, 1.26468511e-04, 3.78635906e-05],
            [2.14543346e-06, 1.07774348e-06, 8.97072326e-07, 5.43274553e-07],
        ]
    ),
}


class TestStrawberryFieldsGBS:
    """Unit tests for StrawberryFieldsGBS."""

    @pytest.mark.parametrize("cutoff", [3, 6])
    def test_load_device(self, cutoff):
        """Test that the device loads correctly when analytic is True"""
        dev = qml.device("strawberryfields.gbs", wires=2, cutoff_dim=cutoff)
        assert dev.cutoff == cutoff
        assert dev.analytic

    def test_load_device_non_analytic(self):
        """Test that the device loads correctly when analytic is False"""
        dev = qml.device("strawberryfields.gbs", wires=2, cutoff_dim=3, analytic=False)
        assert dev.cutoff == 3
        assert not dev.analytic

    def test_calculate_WAW(self):
        """Test that the calculate_WAW method calculates correctly when the input adjacency matrix is
        already normalized to have a mean number of photons equal to 1."""
        const = 2
        A = 0.1767767 * np.ones((4, 4))
        params = const * np.ones(4)
        waw = StrawberryFieldsGBS.calculate_WAW(params, A, 1)
        assert np.allclose(waw, const * A)

    def test_calculate_n_mean(self):
        """Test that calculate_n_mean computes the mean photon number correctly"""
        A = 0.1767767 * np.ones((4, 4))
        n_mean = StrawberryFieldsGBS.calculate_n_mean(A)
        assert np.allclose(n_mean, 1)

    def test_calculate_n_mean_singular_values_large(self):
        """Test that calculate_n_mean raises a ValueError when not all of the singular values are
        less than one"""
        A = np.ones((4, 4))
        with pytest.raises(ValueError, match="Singular values of matrix A must be less than 1"):
            StrawberryFieldsGBS.calculate_n_mean(A)

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
        assert dev.eng.backend_options == {"cutoff_dim": 3}

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

    # def test_probability_analytic(self, monkeypatch):
    #     """Test that the probability method in analytic mode simply calls the parent method in
    #     StrawberryFieldsSimulator. The test monkeypatches StrawberryFieldsSimulator.probability() to
    #     just return True."""
    #     dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
    #     with monkeypatch.context() as m:
    #         m.setattr(StrawberryFieldsSimulator, "probability", lambda *args, **kwargs: True)
    #         p = dev.probability()
    #     assert p

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
        """Test that the calculate_covariance method returns the correct covariance matrix for a
        fixed example."""
        x = np.sqrt(0.5)
        A = np.array([[0, x], [x, 0]])
        cov = StrawberryFieldsGBS.calculate_covariance(A, 2)
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
        params = np.array([0.25, 0.5, 0.6, 1])
        op = [ParamGraphEmbed(params, A, 1, wires=range(4))]

        ob = qml.Identity(wires=range(4))
        ob.return_type = Probability
        obs = [ob]

        variable_deps = {i: ParameterDependency(op, i) for i in range(4)}

        jac = dev.jacobian(op, obs, variable_deps)
        assert np.allclose(jac, jac_exp)

    @pytest.mark.parametrize("wires", jac_reduced)
    def test_jacobian_wires_reduced(self, wires):
        """Test that the jacobian method returns correctly on a fixed example on a subset of
        wires"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        params = np.array([0.25, 0.5, 0.6, 1])
        op = [ParamGraphEmbed(params, A, 1, wires=range(4))]

        ob = qml.Identity(wires=wires)
        ob.return_type = Probability
        obs = [ob]

        variable_deps = {i: ParameterDependency(op, i) for i in range(4)}

        jac = dev.jacobian(op, obs, variable_deps)

        assert np.allclose(jac, jac_reduced[wires])


class TestIntegrationStrawberryFieldsGBS:
    """Integration tests for StrawberryFieldsGBS."""

    @pytest.mark.parametrize("wires", range(1, 5))
    @pytest.mark.parametrize("cutoff_dim", [2, 3])
    def test_shape(self, wires, cutoff_dim):
        """Test that the probabilities and jacobian are returned with the expected shape"""
        dev = qml.device("strawberryfields.gbs", wires=wires, cutoff_dim=cutoff_dim)
        a = np.ones((wires, wires))
        params = np.ones(wires)

        @qml.qnode(dev)
        def vgbs(params):
            ParamGraphEmbed(params, a, 1, wires=range(wires))
            return qml.probs(wires=range(wires))

        d_vgbs = qml.jacobian(vgbs, argnum=0)

        p = vgbs(params)
        dp = d_vgbs(params)

        assert p.shape == (cutoff_dim ** wires,)
        assert dp.shape == (cutoff_dim ** wires, wires)
        assert (p >= 0).all()
        assert (p <= 1).all()
        assert np.sum(p) <= 1

    @pytest.mark.parametrize("wires", range(2, 5))
    @pytest.mark.parametrize("cutoff_dim", [2, 3])
    def test_shape_reduced_wires(self, wires, cutoff_dim):
        """Test that the probabilities and jacobian are returned with the expected shape when
        probabilities are measured on a subset of wires"""
        dev = qml.device("strawberryfields.gbs", wires=wires, cutoff_dim=cutoff_dim)
        a = np.ones((wires, wires))
        params = np.ones(wires)

        @qml.qnode(dev)
        def vgbs(params):
            ParamGraphEmbed(params, a, 1, wires=range(wires))
            return qml.probs(wires=[0, 1])

        d_vgbs = qml.jacobian(vgbs, argnum=0)

        p = vgbs(params)
        dp = d_vgbs(params)

        assert p.shape == (cutoff_dim ** 2,)
        assert dp.shape == (cutoff_dim ** 2, wires)
        assert (p >= 0).all()
        assert (p <= 1).all()
        assert np.sum(p) <= 1

    @pytest.mark.parametrize("wires", [range(4), Wires(["a", 42, "bob", 3])])
    def test_example_jacobian(self, wires):
        """Test that the jacobian is correct on the fixed example"""
        dev = qml.device("strawberryfields.gbs", wires=wires, cutoff_dim=3)
        params = np.array([0.25, 0.5, 0.6, 1])

        @qml.qnode(dev)
        def vgbs(params):
            ParamGraphEmbed(params, A, 1, wires=wires)
            return qml.probs(wires=wires)

        d_vgbs = qml.jacobian(vgbs, argnum=0)
        dp = d_vgbs(params)

        assert np.allclose(dp, jac_exp)

    @pytest.mark.parametrize("wires", [range(4), Wires(["a", 42, "bob", 3])])
    @pytest.mark.parametrize("subset_wires", jac_reduced)
    def test_example_jacobian_reduced_wires(self, subset_wires, wires):
        """Test that the jacobian is correct on the fixed example with a subset of wires"""
        dev = qml.device("strawberryfields.gbs", wires=wires, cutoff_dim=3)
        params = np.array([0.25, 0.5, 0.6, 1])

        @qml.qnode(dev)
        def vgbs(params):
            ParamGraphEmbed(params, A, 1, wires=wires)
            return qml.probs(wires=[wires[subset_wires[0]], wires[subset_wires[1]]])

        d_vgbs = qml.jacobian(vgbs, argnum=0)
        dp = d_vgbs(params)

        assert np.allclose(dp, jac_reduced[subset_wires])

    def test_two_embed(self):
        """Test that the device raises an error if more than one ParamGraphEmbed is used"""
        dev = qml.device("strawberryfields.gbs", wires=4, cutoff_dim=3)
        params = np.array([0.25, 0.5, 0.6, 1])

        @qml.qnode(dev)
        def vgbs(params):
            ParamGraphEmbed(params, A, 1, wires=range(4))
            ParamGraphEmbed(params, A, 1, wires=range(4))
            return qml.probs(wires=range(4))

        with pytest.raises(ValueError, match="The StrawberryFieldsGBS device accepts only"):
            vgbs(params)
