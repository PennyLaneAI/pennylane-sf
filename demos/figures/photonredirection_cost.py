import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm


import pennylane as qml
from pennylane import numpy as np
from pennylane._optimize import GradientDescentOptimizer

dev = qml.device('strawberryfields.fock', wires=2, cutoff_dim=10)

@qml.qfunc(dev)
def func(x, y):

    qml.FockState(1, [0])
    qml.Beamsplitter(x, y, [0, 1])

    return qml.expectation.Fock(1)


fig = plt.figure(figsize = (5, 3))
ax = fig.gca(projection='3d')

# Landscape.
X = np.arange(-3.1, 3.1, 0.1)
Y = np.arange(-3.1, 3.1, 0.1)
length = len(X)
xx, yy = np.meshgrid(X, Y)
Z = np.array([[-func(x, y) for x in X] for y in Y]).reshape(length, length)

# Plot the surface.
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

# # Optimizer trajectories
# path = np.loadtxt("gd_path_0p5.txt")
# path_z = [func(*weights)+0.01 for weights in path]
# path_x = [p[0] for p in path]
# path_y = [p[1] for p in path]
# ax.plot(path_x, path_y, path_z, c='g', marker='.', label="graddesc lr=0.5")
#
# path = np.loadtxt("adag_path_0p5.txt")
# path_z = [func(*weights)+0.01 for weights in path]
# path_x = [p[0] for p in path]
# path_y = [p[1] for p in path]
# ax.plot(path_x, path_y, path_z, c='brown', marker='.', label="adagrad lr=0.5")

# Customize the z axis.
ax.set_zlim(-1.0, 0.0)
ax.set_xlabel("w1")
ax.set_ylabel("w2")

ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.legend()
plt.savefig("redirection_landscape.svg")
plt.show()