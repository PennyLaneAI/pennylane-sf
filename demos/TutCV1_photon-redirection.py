"""Photon redirection example.

In this demo we optimize a beam splitter
to redirect a photon from the first to the second mode.
"""

import openqml as qm
import numpy as np
from openqml.optimize import GradientDescentOptimizer

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)


@qm.qnode(dev)
def circuit(vars):
    """Qnode"""
    qm.FockState(1, [0])
    qm.Beamsplitter(vars[0], vars[1], [0, 1])

    return qm.expval.PhotonNumber(0)


def objective(vars):
    """Objective to minimize"""

    return circuit(vars)


gd = GradientDescentOptimizer(stepsize=0.1)

vars = np.array([0.01, 0.01])

for iteration in range(100):
    vars = gd.step(objective, vars)

    if iteration % 10 == 0:
        print('Cost after step {:3d}: {:0.7f} | Variables [{:0.7f}, {:0.7f}]'
              ''.format(iteration, objective(vars), vars[0], vars[1]))




