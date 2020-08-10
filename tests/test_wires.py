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


# =====


class TestWiresIntegration:
    """Test that the device integrates with PennyLane's wire management."""

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
