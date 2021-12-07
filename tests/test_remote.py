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
Unit tests for the Fock plugin.
"""
import pytest

import strawberryfields as sf
from strawberryfields.devicespec import DeviceSpec
from strawberryfields.result import Result
import pennylane_sf

import pennylane as qml
from pennylane.wires import Wires
import numpy as np


MOCK_SAMPLES = np.array(
    [
        [3, 4, 2, 3, 4, 3, 1, 0],
        [4, 3, 3, 2, 0, 3, 1, 4],
        [2, 1, 3, 3, 3, 2, 2, 4],
        [4, 1, 4, 4, 2, 3, 3, 0],
        [4, 2, 3, 3, 3, 0, 0, 4],
        [1, 2, 4, 4, 2, 0, 0, 4],
        [2, 3, 1, 2, 1, 0, 4, 1],
        [1, 2, 0, 1, 2, 3, 3, 0],
        [1, 2, 4, 0, 0, 4, 2, 4],
        [1, 0, 1, 1, 1, 3, 1, 0],
    ]
)

full_probs = np.zeros([5] * 8)
for s in MOCK_SAMPLES:
    full_probs[tuple(s)] += 1
full_probs /= 10
full_probs = full_probs.ravel()

partial_probs = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.0,
        0.3,
        0.0,
        0.0,
        0.0,
        0.1,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.0,
        0.1,
        0.1,
        0.1,
        0.0,
    ]
)


MOCK_SAMPLES_PROD = np.array([0, 0, 864, 0, 0, 0, 0, 0, 0, 0])


mock_device_dict = {
    "layout": "",
    "modes": 8,
    "compiler": ["fock"],
    "gate_parameters": {},
}


class MockEngine:
    """Mock SF engine class"""

    def __init__(*args):
        pass

    def run(*args, **kwargs):
        return Result({"output": [MOCK_SAMPLES]})

    @property
    def device_spec(self):
        spec = mock_device_dict.copy()
        spec["target"] = "X8"
        return DeviceSpec(spec=spec)



class TestDevice:
    """General tests for the HardwareDevice."""

    def test_token(self, monkeypatch):
        """Tests that the SF store_account function is called with token."""
        test_token = "SomeToken"
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)

        class MockXccSettings:
            settings = {}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def save(self):
                MockXccSettings.settings.update(self.kwargs)

        monkeypatch.setattr("xcc.Settings", MockXccSettings)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=10, sf_token=test_token)

        assert MockXccSettings.settings[0] == test_token

    def test_reset(self, monkeypatch):
        """Tests the reset method of the remote device."""
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=10)

        @qml.qnode(dev)
        def quantum_function():
            return qml.sample(qml.TensorN(wires=list(range(8))))

        quantum_function()

        assert dev.q is not None
        assert dev.prog is not None
        assert dev.samples is not None

        dev.reset()

        assert dev.q is None
        assert dev.prog is None
        assert dev.samples is None

    def test_wires_argument(self, monkeypatch):
        """Tests the wires argument of the remote device."""
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)

        dev_no_wires = qml.device("strawberryfields.remote", backend="X8", shots=10)
        assert dev_no_wires.wires == Wires(range(8))

        with pytest.raises(ValueError, match="Device has a fixed number of"):
            qml.device("strawberryfields.remote", wires=8, backend="X8", shots=10)

        dev_iterable_wires = qml.device(
            "strawberryfields.remote", wires=range(8), backend="X8", shots=10
        )
        assert dev_iterable_wires.wires == Wires(range(8))

        with pytest.raises(ValueError, match="Device has a fixed number of"):
            qml.device("strawberryfields.remote", wires=range(9), backend="X8", shots=10)

    def test_analytic_error(self):
        """Test that instantiating the device with `shots=None` results in an error"""
        with pytest.raises(ValueError, match="does not support analytic"):
            dev = qml.device("strawberryfields.remote", wires=2, backend="X8", shots=None)


class TestSample:
    """Tests that samples are correctly returned from the hardware device."""

    def test_mocked_engine_run_samples(self, monkeypatch):
        """Tests that samples are determined by the RemoteEngine.run method
        from SF by using a mocked RemoteEngine."""
        modes = 8
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.sample(qml.TensorN(wires=list(range(modes))))

        a = quantum_function(1.0, 0)

        assert a.shape == (shots,)
        assert np.array_equal(a, MOCK_SAMPLES_PROD)

    def test_mocked_engine_run_samples_one_mode(self, monkeypatch):
        """Tests that samples are determined by the RemoteEngine.run method
        from SF by using a mocked RemoteEngine and specifying a single mode."""
        modes = [0]
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.sample(qml.TensorN(wires=modes))

        a = quantum_function(1.0, 0)

        expected = np.array([3, 4, 2, 4, 4, 1, 2, 1, 1, 1])
        assert a.shape == (shots,)
        assert np.array_equal(a, expected)

    def test_identity(self, monkeypatch):
        """Tests that sampling the identity returns an array of ones."""
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.sample(qml.Identity(wires=0))

        a = quantum_function(1.0, 0)

        expected = np.ones(shots)
        assert a.shape == (shots,)
        assert np.array_equal(a, expected)

    def test_fock_basis_samples(self, monkeypatch):
        """Test that fock basis samples are correctly returned"""
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        sampled_modes = np.array([0, 3, 5])

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return [qml.sample(qml.NumberOperator(i)) for i in sampled_modes]

        a = quantum_function(1.0, 0)

        assert a.shape == (len(sampled_modes), shots)
        assert np.array_equal(a, MOCK_SAMPLES.T[sampled_modes])


class TestExpval:
    """Test that expectation values are correctly returned from the hardware device."""

    def test_mocked_engine_run_expval(self, monkeypatch):
        """Tests that samples are processed by the samples_expectation SF
        function by using a mocked function instead which returns a pre-defined
        value."""
        modes = 8
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.expval(qml.TensorN(wires=list(range(modes))))

        expected_expval = 100

        monkeypatch.setattr(
            "pennylane_sf.remote.samples_expectation", lambda *args, **kwargs: expected_expval
        )
        a = quantum_function(1.0, 0)

        assert a == expected_expval

    def test_identity(self, monkeypatch):
        """Tests that the expectation value for the identity is zero."""
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.expval(qml.Identity(wires=0))

        a = quantum_function(1.0, 0)

        assert a == 1

    @pytest.mark.parametrize("mode, expectation", ((0, 2.3), (1, 2.0)))
    def test_expvals(self, monkeypatch, mode, expectation):
        """Test that the expectation value is evaluated correctly when applied to MOCK_SAMPLES"""
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)
        dev.samples = MOCK_SAMPLES
        w = Wires(dev.wires[mode])
        result = dev.expval(qml.NumberOperator, w, None)
        assert np.allclose(result, expectation)


class TestVariance:
    """Test that variances are correctly returned from the hardware device."""

    def test_mocked_engine_run_var(self, monkeypatch):
        """Tests that samples are processed by the samples_variance SF function
        by using a mocked function instead which returns a pre-defined value."""
        modes = 8
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.var(qml.TensorN(wires=list(range(modes))))

        expected_var = 100

        monkeypatch.setattr(
            "pennylane_sf.remote.samples_variance", lambda *args, **kwargs: expected_var
        )
        var = quantum_function(1.0, 0)

        assert var == expected_var

    def test_identity(self, monkeypatch):
        """Tests that the variance for the identity is zero."""
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.var(qml.Identity(wires=0))

        a = quantum_function(1.0, 0)
        assert a == 0

    @pytest.mark.parametrize("mode, var", ((0, 1.61), (1, 1.2)))
    def test_vars(self, monkeypatch, mode, var):
        """Test that the variance is evaluated correctly when applied to MOCK_SAMPLES"""
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)
        dev.samples = MOCK_SAMPLES
        w = Wires(dev.wires[mode])
        result = dev.var(qml.NumberOperator, w, None)
        assert np.allclose(result, var)


class TestProbs:
    """Test that probabilities are correctly returned from the hardware device."""

    @pytest.mark.parametrize("exp_prob, wires", [[full_probs, range(8)], [partial_probs, [0, 1]]])
    def test_probs(self, monkeypatch, tol, wires, exp_prob):
        """Tests that probabilities are correctly returned when the cutoff dimension of the samples
        matches the cutoff of the device."""
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots, cutoff_dim=5)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.probs(wires=wires)

        probs = quantum_function(1.0, 0)

        assert probs.shape == (dev.cutoff ** len(wires),)
        assert np.allclose(probs, exp_prob, atol=tol)

    def test_probs_subset_modes_high_cutoff(self, monkeypatch, tol):
        """Tests that probabilities are correctly returned when using a subset of modes and where
        the device cutoff is above the samples cutoff."""
        wires = [0, 1]
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots, cutoff_dim=6)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.probs(wires=wires)

        probs = quantum_function(1.0, 0)

        # with dev.cutoff = 6, probs will be a 36-dimensional flat array. Since
        # partial_probs is a 25-dimensional flat array, we first reshape to a 5x5 matrix and then
        # pad to get a 6x6 matrix, finally flattening
        exp_probs = np.reshape(partial_probs, (5, 5))
        exp_probs = np.pad(exp_probs, [(0, 1), (0, 1)]).ravel()

        assert probs.shape == (dev.cutoff ** len(wires),)
        assert np.allclose(probs, exp_probs, atol=tol)

    def test_probs_all_modes_high_cutoff(self, monkeypatch, tol):
        """Tests that probabilities are correctly returned when using all modes and where
        the device cutoff is above the samples cutoff."""
        wires = range(8)
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots, cutoff_dim=6)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.probs(wires=wires)

        probs = quantum_function(1.0, 0)

        exp_probs = np.reshape(full_probs, (5, 5, 5, 5, 5, 5, 5, 5))
        exp_probs = np.pad(exp_probs, [(0, 1)] * 8).ravel()

        assert probs.shape == (dev.cutoff ** len(wires),)
        assert np.allclose(probs, exp_probs, atol=tol)

    def test_probs_subset_modes_low_cutoff(self, monkeypatch, tol):
        """Tests that probabilities are correctly returned when using a subset of modes and where
        the device cutoff is below the samples cutoff."""
        wires = [0, 1]
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots, cutoff_dim=3)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.probs(wires=wires)

        with pytest.warns(UserWarning, match="Samples were generated where at least one mode"):
            probs = quantum_function(1.0, 0)

        # with dev.cutoff = 3, probs will be a 9-dimensional flat array. Since
        # partial_probs is a 25-dimensional flat array, we first reshape to a 5x5 matrix and then
        # take the first 3 elements on both axes, finally flattening
        exp_probs = partial_probs.reshape((5, 5))[:3, :3]

        assert probs.shape == (dev.cutoff ** len(wires),)
        assert np.allclose(probs, exp_probs.ravel(), atol=tol)

    def test_probs_all_modes_low_cutoff(self, monkeypatch, tol):
        """Tests that probabilities are correctly returned when using all modes and where
        the device cutoff is below the samples cutoff."""
        wires = range(8)
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots, cutoff_dim=3)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.probs(wires=wires)

        with pytest.warns(UserWarning, match="Samples were generated where at least one mode"):
            probs = quantum_function(1.0, 0)

        exp_probs = full_probs.reshape([5] * 8)[:3, :3, :3, :3, :3, :3, :3, :3]

        assert probs.shape == (dev.cutoff ** len(wires),)
        assert np.allclose(probs, exp_probs.ravel(), atol=tol)

    def test_mocked_probability(self, monkeypatch, tol):
        """Tests that pre-defined probabilities are correctly propagated
        through PennyLane when the StrawberryFieldsRemote.probability method is
        mocked out."""
        wires = [0, 1]
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.probs(wires=[0, 1])

        # Mocking the probability method of the device which returns a dictionary
        # When Device.execute gets called, the values() are extracted => keys
        # do not matter
        monkeypatch.setattr(
            "pennylane_sf.remote.StrawberryFieldsRemote.probability",
            lambda *args, **kwargs: {"somekey": partial_probs},
        )
        probs = quantum_function(1.0, 0)

        assert probs.shape == (dev.cutoff ** len(wires),)
        assert np.allclose(probs, partial_probs, atol=tol)

    def test_modes_none(self, monkeypatch):
        """Tests that probabilities are returned using SF without any further
        processing when no specific modes were specified."""
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = pennylane_sf.StrawberryFieldsRemote(backend="X8")

        mock_returned_probs = np.array(
            [
                [0.1, 0.0, 0.2, 0.1, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.1, 0.1, 0.1, 0.0],
            ]
        )

        shape = mock_returned_probs.shape[0]

        # Mock internal attributes used in the probability method
        # The pre-defined probabilities are fed in
        dev.num_wires = 2
        dev.samples = mock_returned_probs
        monkeypatch.setattr("pennylane_sf.remote.all_fock_probs_pnr", lambda arg: arg)
        probs_dict = dev.probability()

        probs = np.array(list(probs_dict.values()))

        # Check that the output shape is the shape of the flattened result
        # coming from all_fock_probs
        assert probs.shape == (shape ** 2,)
        assert np.array_equal(probs, mock_returned_probs.flatten())
