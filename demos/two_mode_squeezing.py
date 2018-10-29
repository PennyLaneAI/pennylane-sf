"""Two mode squeezing example.

In this demo we optimize an optical quantum circuit such that the mean photon number at mode 1 is 1.
"""
import pennylane as qml
from pennylane import numpy as np

dev1 = qml.device('strawberryfields.gaussian', wires=2)

@qml.qfunc(dev1)
def circuit(alpha, r):
    """Two mode squeezing with PNR on mode 1

    Args:
        alpha (float): displacement parameter
        r (float): squeezing parameter
    """
    qml.Displacement(alpha, 0, wires=[0])
    qml.TwoModeSqueezing(r, 0, wires=[0, 1])
    return qml.expectation.Fock(wires=1)

def cost(weights):
    """Cost (error) function to be minimized.

    Args:
        weigts (float): weights
    """
    return np.abs(circuit(*weights)-1)

# initialize alpha and r with random value
init_weights = np.random.randn(2)
o = qml.Optimizer(cost, init_weights, optimizer='Nelder-Mead')

# train the circuit
c = o.train(max_steps=100)

# print the results
print('Initial [alpha, r] parameters:', init_weights)
print('Optimized [alpha, r] parameter:', o.weights)
print('Circuit output at optimized parameters:', circuit(*o.weights))
print('Circuit gradient at optimized parameters:', qml.grad(circuit, [o.weights]))
print('Cost gradient at optimized parameters:', qml.grad(cost, [o.weights, None]))
