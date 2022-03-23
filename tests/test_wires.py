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
"""Tests that a device handles arbitrary user-defined wire labels."""
# pylint: disable=no-self-use
import pennylane as qml
import pytest
from pennylane import numpy as np
from strawberryfields.device import Device
from strawberryfields.result import Result


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
    "target": "X8",
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
    def device(self):
        return Device(spec=mock_device_dict)


# ===== Factories for circuits using arbitrary wire labels and numbers


def make_simple_circuit_expval(device, wires):
    """Factory for a qnode returning expvals."""

    n_wires = len(wires)

    @qml.qnode(device)
    def circuit():
        qml.Displacement(0.5, 0.1, wires=wires[0 % n_wires])
        qml.Displacement(2.0, 0.2, wires=wires[1 % n_wires])
        if n_wires > 1:
            qml.Beamsplitter(0.5, 0.2, wires=[wires[0], wires[1]])
        return [qml.expval(qml.X(wires=w)) for w in wires]

    return circuit


def make_x8_circuit_expval(device, wires):
    """Factory for a qnode running the X8 remote device and returning expvals."""

    @qml.qnode(device)
    def circuit(theta, phi):
        qml.Beamsplitter(theta, phi, wires=[wires[0], wires[1]])
        qml.Beamsplitter(theta, phi, wires=[wires[4], wires[5]])
        return qml.expval(qml.TensorN(wires=wires))

    return circuit

# =====


class TestWiresIntegration:
    """Test that the simulator devices integrate with PennyLane's wire management."""

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    @pytest.mark.parametrize("circuit_factory", [make_simple_circuit_expval])
    def test_wires_fock(self, circuit_factory, wires1, wires2, tol):
        """Test that the expectation of the fock device is independent from the wire labels used."""
        dev1 = qml.device("strawberryfields.fock", wires1, cutoff_dim=5)
        dev2 = qml.device("strawberryfields.fock", wires2, cutoff_dim=5)

        circuit1 = circuit_factory(dev1, wires1)
        circuit2 = circuit_factory(dev2, wires2)

        assert np.allclose(circuit1(), circuit2(), tol)

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            (["a", "c", "d"], [2, 3, 0]),
            ([-1, -2, -3], ["q1", "ancilla", 2]),
            (["a", "c"], [3, 0]),
            ([-1, -2], ["ancilla", 2]),
            (["a"], ["nothing"]),
        ],
    )
    @pytest.mark.parametrize("circuit_factory", [make_simple_circuit_expval])
    def test_wires_gaussian(self, circuit_factory, wires1, wires2, tol):
        """Test that the expectation of the gaussian device is independent from the wire labels used."""
        dev1 = qml.device("strawberryfields.gaussian", wires1)
        dev2 = qml.device("strawberryfields.gaussian", wires2)

        circuit1 = circuit_factory(dev1, wires1)
        circuit2 = circuit_factory(dev2, wires2)

        assert np.allclose(circuit1(), circuit2(), tol)


class TestWiresIntegrationRemote:
    """Test that the remote devices integrate with PennyLane's wire management."""

    @pytest.mark.parametrize(
        "wires1, wires2",
        [
            (["a", "c", "d", "b", "f", "e", "h", "g"], [0, 1, 2, 3, 4, 5, 6, 7]),
            (["a", "c", "d", "b", "f", "e", "h", "g"], ["a", "b", "c", "d", "e", "f", "g", "h"]),
        ],
    )
    def test_wires_remote(self, wires1, wires2, tol, monkeypatch):
        """Test that the expectation of the remote device is independent from the wire labels used."""
        shots = 10
        expected_expval = 100

        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        dev1 = qml.device("strawberryfields.remote", wires=wires1, backend="X8", shots=shots)
        dev2 = qml.device("strawberryfields.remote", wires=wires2, backend="X8", shots=shots)

        circuit1 = make_x8_circuit_expval(dev1, wires1)
        circuit2 = make_x8_circuit_expval(dev2, wires2)

        monkeypatch.setattr(
            "pennylane_sf.remote.samples_expectation", lambda *args, **kwargs: expected_expval
        )

        assert np.allclose(circuit1(1.0, 0), circuit2(1.0, 0), tol)

    def test_subset_of_wires_in_probs(self, tol, monkeypatch):
        """Test that the right probability is returned when the probability of a subset of wires is requested."""
        shots = 7

        wires = ["a", "c", "d", "b", "f", "e", "h", "g"]

        monkeypatch.setattr("strawberryfields.RemoteEngine", MockEngine)
        device = qml.device("strawberryfields.remote", wires=wires, backend="X8", shots=shots)

        @qml.qnode(device)
        def circuit():
            return qml.probs(wires=["a"])

        expected = np.array([0., 0.4, 0.2, 0.1, 0.3])

        assert np.allclose(circuit(), expected)
