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
from strawberryfields.api import Result
import pennylane_sf

import pennylane as qml
import numpy as np

MOCK_SAMPLES = np.array([[3, 4, 2, 3, 4, 3, 1, 0],
       [4, 3, 3, 2, 0, 3, 1, 4],
       [2, 1, 3, 3, 3, 2, 2, 4],
       [4, 1, 4, 4, 2, 3, 3, 0],
       [4, 2, 3, 3, 3, 0, 0, 4],
       [0, 2, 4, 4, 2, 0, 0, 4],
       [0, 3, 1, 2, 1, 0, 4, 1],
       [0, 2, 0, 1, 2, 3, 3, 0],
       [1, 2, 4, 0, 0, 4, 2, 4],
       [0, 0, 1, 1, 1, 3, 1, 0]])

class MockEngine:
    """Mock SF engine class"""

    def __init__(*args):
        pass

    def run(*args, **kwargs):
        return Result(MOCK_SAMPLES)

class TestSample:
    """Test that samples are correctly returned from the hardware device"""

    def test_mocked_engine_run_samples(self, monkeypatch):
        """Tests that samples are determined by the RemoteEngine.run method
        from SF by using a mocked RemoteEngine"""
        modes = 8
        shots = 10
        dev = qml.device('strawberryfields.ai', chip="X8", wires=modes, shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi,wires=[0,1])
            qml.Beamsplitter(theta, phi,wires=[4,5])
            return qml.sample(qml.TensorN(wires=list(range(modes))))

        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        a = quantum_function(1., 0.543)

        assert a.shape == (shots, modes)
        assert np.array_equal(a, MOCK_SAMPLES)

class TestExpval:
    """Test that expectation values are correctly returned from the hardware device"""

    def test_mocked_engine_run_expval(self, monkeypatch):
        """Tests that samples are processed by the samples_expectation SF
        function by using a mocked function instead which returns a pre-defined
        value"""
        modes = 8
        shots = 10
        dev = qml.device('strawberryfields.ai', chip="X8", wires=modes, shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0,1])
            qml.Beamsplitter(theta, phi, wires=[4,5])
            return qml.expval(qml.TensorN(wires=list(range(modes))))

        expected_expval = 100 

        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        monkeypatch.setattr("pennylane_sf.hw.samples_expectation", lambda *args: expected_expval)
        a = quantum_function(1., 0.543)

        assert a == expected_expval

class TestVariance:
    """Test that variances are correctly returned from the hardware device"""

    def test_mocked_engine_run_var(self, monkeypatch):
        """Tests that samples are processed by the samples_variance SF function
        by using a mocked function instead which returns a pre-defined value"""
        modes = 8
        shots = 10
        dev = qml.device('strawberryfields.ai', chip="X8", wires=modes, shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0,1])
            qml.Beamsplitter(theta, phi, wires=[4,5])
            return qml.var(qml.TensorN(wires=list(range(modes))))

        expected_var = 100 

        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        monkeypatch.setattr("pennylane_sf.hw.samples_variance", lambda *args: expected_var)
        var = quantum_function(1., 0.543)

        assert var == expected_var


class TestProbs:
    """Test that probabilities are correctly returned from the hardware device"""

    def test_mocked_engine_run_all_fock_probs(self, monkeypatch):
        """Tests that probabilities are correctly summed when specifying a
        subset of the wires and using a mock SF RemoteEngine"""
        modes = 8
        shots = 10
        dev = qml.device('strawberryfields.ai', chip="X8", wires=modes, shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0,1])
            qml.Beamsplitter(theta, phi, wires=[4,5])
            return qml.probs(wires=[0, 1])

        expected_probs = np.array([[0.1, 0. , 0.2, 0.1, 0. ],
                                   [0. , 0. , 0.1, 0. , 0. ],
                                   [0. , 0.1, 0. , 0. , 0. ],
                                   [0. , 0. , 0. , 0. , 0.1],
                                   [0. , 0.1, 0.1, 0.1, 0. ]])

        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        probs = quantum_function(1., 0.543)

        assert np.array_equal(probs, expected_probs)

    def test_mocked_probability(self, monkeypatch):
        """Tests that pre-defined probabilities are correctly propagated
        through PennyLane when the StrawberryFieldsRemote.probability method is
        mocked out"""
        modes = 8
        shots = 10
        dev = qml.device('strawberryfields.ai', chip="X8", wires=modes, shots=shots)

        @qml.qnode(dev)
        def quantum_function(theta, phi):
            qml.Beamsplitter(theta, phi, wires=[0,1])
            qml.Beamsplitter(theta, phi, wires=[4,5])
            return qml.probs(wires=[0, 1])

        expected_probs = np.array([[0.3, 0. , 0.1, 0.1, 0. ],
                                  [0. , 0. , 0.1, 0.1, 0. ],
                                  [0. , 0, 0. , 0. , 0. ],
                                  [0. , 0. , 0. , 0. , 0.1],
                                  [0. , 0, 0.1, 0.1, 0. ]])

        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)

        # Mocking the probability method of the device which returns a dictionary
        # When Device.execute gets called, the values() are extracted => keys
        # do not matter
        monkeypatch.setattr("pennylane_sf.hw.StrawberryFieldsRemote.probability", lambda *args, **kwargs: {'somekey': expected_probs})
        probs = quantum_function(1., 0.543)

        assert np.array_equal(probs, expected_probs)
