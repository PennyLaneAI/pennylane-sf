"""Continuous-variable quantum neural network example.

In this demo we implement the photonic quantum neural net model
from Killoran et al. (arXiv:1806.06871) with the example
of function fitting.
"""

import openqml as qm
from openqml import numpy as np
from openqml.optimize import AdamOptimizer

dev = qm.device('strawberryfields.fock', wires=1, cutoff_dim=10)


def layer(w):
    """ Single layer of the quantum neural net."""

    # Bias
    qm.Displacement(w[0], w[1], [0])

    # Matrix multiplication of input layer
    qm.Rotation(w[2], [0])
    qm.Squeezing(w[3], w[4], [0])
    qm.Rotation(w[5], [0])

    # Nonlinear transformation
    qm.Kerr(w[6], [0])


@qm.qnode(dev)
def quantum_neural_net(weights, x=None):
    """The quantum neural net variational circuit."""

    # Encode input x into quantum state
    qm.Displacement(x, 0., [0])

    # execute "layers"
    for w in weights:
        layer(w)

    return qm.expval.X(0)


def square_loss(labels, predictions):
    """ Square loss function

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions
    Returns:
        float: square loss
    """
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l-p)**2
    loss = loss/len(labels)

    return loss


def cost(weights, features, labels):
    """Cost function to be minimized."""

    # Compute prediction for each input in data batch
    preds = [quantum_neural_net(weights, x=x) for x in features]

    loss = square_loss(labels, preds)

    return loss


# load function data
data = np.loadtxt("sine.txt")
X = data[:, 0]
Y = data[:, 1]

# initialize weights
num_layers = 4
vars_init = 0.05*np.random.randn(num_layers, 7)

# create optimizer
o = AdamOptimizer(0.005, beta1=0.9, beta2=0.999)

# train
vars = vars_init
for it in range(50):
    vars = o.step(lambda v: cost(v, X, Y), vars)
    print("Iter: {:5d} | Cost: {:0.7f} | Mean of abs vars: {:0.7f}"
          .format(it+1, cost(vars, X, Y), np.mean(np.abs(vars))))
