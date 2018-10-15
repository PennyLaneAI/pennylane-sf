"""Continuous-variable quantum neural network.

In this demo we implement the cv-qnn of Ref XXX with the
example of function fitting.
"""

import openqml as qm
from openqml import numpy as np
from openqml.optimize import AdamOptimizer

dev = qm.device('strawberryfields.fock', wires=1, cutoff_dim=10)


def layer(w):
    """ Single layer of the continuous-variable quantum neural net."""

    # Bias
    qm.Displacement(w[0], w[1], [0])

    # Matrix multiplication of input layer
    qm.Rotation(w[2], [0])
    qm.Squeezing(w[3], w[4], [0])
    qm.Rotation(w[5], [0])

    # Element-wise nonlinear transformation
    qm.Kerr(w[6], [0])


@qm.qfunc(dev)
def quantum_neural_net(weights, x):
    """The quantum neural net variational circuit."""

    # Encode input into quantum state
    qm.Displacement(x, 0., [0])

    # execute "layers"
    for i in range(6):  # TODO: back to multidim arrays
        layer(weights[i*7: i*7+7])

    return qm.expectation.X(0)


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


def regularizer(weights):
    w_flat = weights.flatten()
    return np.abs(np.sum(w_flat**2))


def cost(weights, features, labels):
    """Cost (error) function to be minimized."""
    # Compute prediction for each input in data batch
    predictions = [quantum_neural_net(weights, x) for x in features]
    loss = square_loss(labels, predictions)
    cost = loss #+ 0.0*regularizer(weights)

    return cost


# load function data
data = np.loadtxt("sine.txt")
X = data[:, 0]
Y = data[:, 1]

# initialize weights
num_layers = 6
weights0 = 0.5*np.random.randn(num_layers*7)
print("Initial cost: {:0.7f}".format(cost(weights0, X, Y)))

# create optimizer
o = AdamOptimizer(0.5)

# train
weights = weights0
for it in range(10):
    weights = o.step(lambda w: cost(w, X, Y), weights)
    print("Iter: {:5d} | Cost: {:0.7f}".format(it+1, cost(weights, X, Y)))
