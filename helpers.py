import numpy as np
import matplotlib.pyplot as plt


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