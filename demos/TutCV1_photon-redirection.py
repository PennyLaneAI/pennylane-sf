"""Photon redirection example.

In this demo we optimize a beam splitter
to redirect a photon from the first to the second mode.
"""

import openqml as qm
from openqml import numpy as np
from openqml.optimize import GradientDescentOptimizer

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)


@qm.qfunc(dev)
def circuit(weights):
    """Qnode"""
    qm.FockState(1, [0])
    qm.Beamsplitter(weights[0], weights[1], [0, 1])

    return qm.expectation.PhotonNumber(0)


def objective(weights):
    """Objective to minimize"""
    return circuit(weights)


vars_init = np.array([0.01, 0.0])

o = GradientDescentOptimizer(0.1)
variables = o.step(objective, vars_init)

print("Initial cost {:0.7f}: ".format(objective(variables)))
variables = vars_init
for iteration in range(100):
    variables = o.step(objective, variables)
    if iteration % 5 == 0:
        print('Cost after step {:3d}: {:0.7f}'
              ''.format(iteration, objective(variables)))




