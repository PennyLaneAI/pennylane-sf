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

    def __init__(*args):
        pass

    def run(*args, **kwargs):
        return Result(MOCK_SAMPLES)

class TestSample:
    """TODO"""

    def test_mocked_engine_run_samples(self, monkeypatch):
        """TODO"""
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
    """TODO"""

    def test_mocked_engine_run_expval(self, monkeypatch):
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
    """TODO"""

    def test_mocked_engine_run_var(self, monkeypatch):
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
    """TODO"""

    def test_mocked_engine_run_all_fock_probs(self, monkeypatch):
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
        monkeypatch.setattr("pennylane_sf.hw.samples_variance", lambda *args: expected_probs)
        probs = quantum_function(1., 0.543)

        assert np.array_equal(probs, expected_probs)
