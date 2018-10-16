"""Continuous-variable quantum kernel classifier example.

In this demo we implement the explicit quantum kernel classifier
from Schuld and Killoran (arXiv:1803.07128) with a 2-dimensional moons data set.
"""

import openqml as qm
from openqml import numpy as np
from openqml.optimize import GradientDescentOptimizer

dev = qm.device('strawberryfields.fock', wires=2, cutoff_dim=10)


def layer(w):
    """ Single layer of the quantum neural net."""

    qm.Beamsplitter(w[0], w[1], [0, 1])

    # linear gates in quadrature
    qm.Displacement(w[2], 0., [0])
    qm.Displacement(w[3], 0., [1])

    # quadratic gates in quadrature
    qm.QuadraticPhase(w[4], [0])
    qm.QuadraticPhase(w[5], [1])

    # cubic gates in quadrature
    qm.CubicPhase(w[6], [0])
    qm.CubicPhase(w[7], [1])


def featuremap(x):
    """Encode input x into a squeezed state."""

    qm.Squeezing(1.5, x[0], [0])
    qm.Squeezing(1.5, x[1], [0])


@qm.qnode(dev)
def qclassifier(weights, x=None):
    """The variational circuit of the quantum classifier."""

    # execute feature map
    featuremap(x)

    # execute linear classifier
    for w in weights:
        layer(w)

    return qm.expval.PhotonNumber(0)


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


def accuracy(labels, predictions):
    """ Share of equal labels and predictions

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions
    Returns:
        float: accuracy
    """

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l-p) < 1e-5:
            loss += 1
    loss = loss/len(labels)

    return loss


def cost(weights, X, Y):
    """Cost function to be minimized."""

    outpts = [qclassifier(weights, x=x) for x in X]

    loss = square_loss(Y, outpts)

    return loss


# load function data
data = np.loadtxt("moons.txt")
X = data[:, 0:2]
Y = data[:, -1]

# split into training and validation set
num_data = len(Y)
num_train = int(0.5*num_data)
index = np.random.permutation(range(num_data))
X_train = X[index[: num_train]]
Y_train = Y[index[: num_train]]
X_val = X[index[num_train: ]]
Y_val = Y[index[num_train: ]]


# initialize weights
num_layers = 4
vars_init = 0.05*np.random.randn(num_layers, 8)

# create optimizer
o = GradientDescentOptimizer(0.01)

# train
batch_size = 5
vars = vars_init

# select minibatch of training samples
batch_index = np.random.randint(0, num_train, (batch_size,))
X_train_batch = X_train[batch_index]
Y_train_batch = Y_train[batch_index]
for it in range(50):

    vars = o.step(lambda v: cost(v, X_train_batch, Y_train_batch), vars)

    pred_train = [qclassifier(vars, x=x_) for x_ in X_train_batch]


    print(pred_train)
    print(Y_train_batch)


    # Compute accuracy on train and validation set
    #pred_train = [qclassifier(vars, x=x_) for x_ in X_train]
    #pred_val = [qclassifier(vars, x=x_) for x_ in X_val]
    #acc_train = accuracy(Y_train, pred_train)
    #acc_val = accuracy(Y_val, pred_val)

    #print("Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
    #      "".format(it+1, cost(vars, X_train_batch, Y_train_batch), acc_train, acc_val))