import numpy as np
import matplotlib.pyplot as plt
from phe import paillier


def plot_ellipsoid(P, alpha, label):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    # Generate points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.array([np.cos(theta), np.sin(theta)])

    # Scale and rotate points to form the ellipsoid
    # The factor sqrt(alpha) adjusts the size of the ellipsoid
    ellipsoid_points = np.sqrt(alpha) * eigenvectors @ np.diag(np.sqrt(1 / eigenvalues)) @ circle_points

    # Plotting
    plt.fill(ellipsoid_points[0, :], ellipsoid_points[1, :], label=label, alpha=0.7)


def plot_boundary(c, b, label, bounds, color):
    x = np.linspace(bounds[0], bounds[1], 2)
    y = ((b - c[0] * x) / c[1]).reshape(x.shape)
    if 0. in c:
        if c[0] == 0.:
            y = np.linspace(bounds[0], bounds[1], 2)
            x = [-b, -b]
        else:
            x = np.linspace(bounds[0], bounds[1], 2)
            y = [-b, -b]
    plt.plot(x, y, "--", color=color, label=label)


def admm_analysis(iter_max, agents, mediator, timing_log):
    x_ticks = np.arange(0, iter_max + 1)
    plt.figure(1)
    plt.plot(x_ticks, agents[0].x, label=r"$x_1$")
    plt.plot(x_ticks, agents[1].x, label=r"$x_2$")
    plt.plot(x_ticks, agents[2].x, label=r"$x_3$")
    for i, x_bar_i in enumerate(mediator.x_bar):
        if isinstance(x_bar_i, paillier.EncryptedNumber):
            mediator.x_bar[i] = mediator.pri_k.decrypt(x_bar_i)
    plt.plot(x_ticks, mediator.x_bar, "--", label=r"$\bar{x}$")
    plt.xlabel("iteration number")
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(0, iter_max), timing_log)
    plt.xlabel("iteration number")
    plt.ylabel("time [s]")
    plt.show()
