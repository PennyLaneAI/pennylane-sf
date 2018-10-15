"""Two mode squeezing example.

In this demo we optimize an optical quantum circuit such that
the mean photon number at mode 1 is 1.
"""
import openqml as qm
from openqml import numpy as np
from openqml.optimize import GradientDescentOptimizer

dev = qm.device('strawberryfields.gaussian', wires=2)

@qm.qfunc(dev)
def circuit(alpha, r):
    """Two mode squeezing with PNR on mode 1

    Args:
        alpha (float): displacement parameter
        r (float): squeezing parameter
    """
    qm.Displacement(alpha, 0, wires=[0])
    qm.TwoModeSqueezing(r, 0, wires=[0, 1])
    return qm.expectation.PhotonNumber(wires=1)


def cost(weights):
    """Cost (error) function to be minimized.

    Args:
        weigts (float): weights
    """
    return np.abs(circuit(*weights)-1)


# initialize alpha and r
init_weights = np.array([1., 0.5])
o = GradientDescentOptimizer(0.001)

weights = init_weights
for it in range(50):
    weights = o.step(lambda w: cost(w), weights)
    print("Iter: {:5d} | Cost: {:0.7f} | Photon number {:0.4f}"
          .format(it+1, cost(weights), circuit(*weights)))



