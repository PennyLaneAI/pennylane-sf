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
from strawberryfields.api import Result, DeviceSpec
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
        return Result(MOCK_SAMPLES)

    @property
    def device_spec(self):
        return DeviceSpec(target="X8", spec=mock_device_dict, connection=None)


class TestDevice:
    """General tests for the HardwareDevice."""

    def test_token(self, monkeypatch):
        """Tests that the SF store_account function is called with token."""
        test_token = "SomeToken"
        recorder = []
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        monkeypatch.setattr("strawberryfields.store_account", lambda arg: recorder.append(arg))
        dev = qml.device("strawberryfields.remote", backend="X8", shots=10, sf_token=test_token)

        assert recorder[0] == test_token

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

        dev_int_wires = qml.device("strawberryfields.remote", wires=8, backend="X8", shots=10)
        assert dev_int_wires.wires == Wires(range(8))

        with pytest.raises(ValueError, match="This hardware device has a fixed number"):
            qml.device("strawberryfields.remote", wires=7, backend="X8", shots=10)

        dev_iterable_wires = qml.device("strawberryfields.remote", wires=range(8), backend="X8", shots=10)
        assert dev_iterable_wires.wires == Wires(range(8))

        with pytest.raises(ValueError, match="This hardware device has a fixed number"):
            qml.device("strawberryfields.remote", wires=range(9), backend="X8", shots=10)


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
            "pennylane_sf.remote.samples_expectation", lambda *args: expected_expval
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

        monkeypatch.setattr("pennylane_sf.remote.samples_variance", lambda *args: expected_var)
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


class TestProbs:
    """Test that probabilities are correctly returned from the hardware device."""

    def test_mocked_engine_run_all_fock_probs(self, monkeypatch, tol):
        """Tests that probabilities are correctly summed when specifying a
        subset of the wires and using a mock SF RemoteEngine."""
        wires = [0, 1]
        shots = 10
        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev = qml.device("strawberryfields.remote", backend="X8", shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0, 1])
            qml.Beamsplitter(theta, phi, wires=[4, 5])
            return qml.probs(wires=wires)

        expected_probs = np.array(
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

        probs = quantum_function(1.0, 0)

        # all_fock_probs uses a cutoff equal to 1 + the maximum number of
        # photons detected (including no photons detected)
        cutoff = 1 + np.max(MOCK_SAMPLES)

        # Check that the shape of the expected probabilities indeed matches the cutoff
        # The combination of yields a shape of (cutoff, cutoff)
        assert probs.shape == cutoff ** len(wires)
        assert np.allclose(probs, expected_probs, atol=tol)

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

        expected_probs = np.array(
            [
                0.3,
                0.0,
                0.1,
                0.1,
                0.0,
                0.0,
                0.0,
                0.1,
                0.1,
                0.0,
                0.0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0,
                0.1,
                0.1,
                0.0,
            ]
        )

        # Mocking the probability method of the device which returns a dictionary
        # When Device.execute gets called, the values() are extracted => keys
        # do not matter
        monkeypatch.setattr(
            "pennylane_sf.remote.StrawberryFieldsRemote.probability",
            lambda *args, **kwargs: {"somekey": expected_probs},
        )
        probs = quantum_function(1.0, 0)

        # all_fock_probs uses a cutoff equal to 1 + the maximum number of
        # photons detected (including no photons detected)
        cutoff = 1 + np.max(MOCK_SAMPLES)

        # Check that the shape of the expected probabilities indeed matches the cutoff
        # The combination of yields a shape of (cutoff, cutoff)
        assert probs.shape == cutoff ** len(wires)
        assert np.allclose(probs, expected_probs, atol=tol)

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
