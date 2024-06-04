import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

F = np.array([[0.84, 0.23],
              [-0.47, 0.12]])
G = np.array([[0.07, 0.3],
              [0.23, 0.1]])

gamma_1 = 8
gamma_2 = 10
R = np.diag([1 / gamma_1, 1 / gamma_2])


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
    plt.fill(ellipsoid_points[0, :], ellipsoid_points[1, :], label=label, alpha=0.7)  # Ellipsoid boundary


def plot_boundary(c, b, label, bounds, color):
    x = np.linspace(bounds[0], bounds[1], 2)
    y = ((b - c[0] * x) / c[1]).reshape(x.shape)
    plt.plot(x, y, "--", color=color, label=label)


def find_reachable_set(a):
    P = cp.Variable((2, 2), symmetric=True)
    obj = cp.Minimize(- cp.log_det(P))
    const = []
    slack = np.ones(P.shape) * 1e-5
    const += [P + slack >> 0]  # positive definite
    matrix = cp.bmat([[a*P - F.T @ P @ F, - F.T @ P @ G],
                        [- G.T @ P @ F, (1 - a) * R - G.T @ P @ G]])
    const += [matrix >> 0]  # positive semi definite

    problem = cp.Problem(obj, const)
    problem.solve()
    #print(f"The optimization problem is {problem.status}.")
    return P.value, problem.status


def find_safe_reachable_set(a, c, b):
    Rhat = cp.Variable((2, 2), symmetric=True)
    Y = cp.Variable((2, 2), symmetric=True)

    obj = cp.Minimize(cp.trace(Rhat))

    matrix = cp.bmat([[a * Y, np.zeros((2, 2)), Y @ F.T],
                      [np.zeros((2, 2)), (1 - a) * Rhat, G.T],
                      [F @ Y, G, Y]])
    const = []
    for i in range(len(b)):
        const += [c[:, i].T @ Y @ c[:, i] <= b[i]**2 / 2]
    const += [matrix >> 0]
    const += [Rhat == cp.diag(cp.diag(Rhat))]
    const += [Rhat - R >> 0]
    slack = np.ones(Y.shape) * 1e-5
    const += [Y + slack >> 0]

    problem = cp.Problem(obj, const)
    problem.solve()
    #print(f"The optimization problem is {problem.status}.")
    Y_inv = np.linalg.inv(Y.value)
    return Y_inv, Rhat.value, problem.status


def do_grid_search(constraints=False, **kwargs):
    P_area_min = np.inf
    P_min = None
    a_opt = None
    Rhat_opt = None
    alpha = 2
    for a in np.linspace(0.01, 0.99, 10):
        if not constraints:
            P, ps = find_reachable_set(a)
            Rhat = 0
        else:
            P, Rhat, ps = find_safe_reachable_set(a, kwargs['c'], kwargs['b'])
        # Compute the semi-axes lengths
        evs, _ = np.linalg.eigh(P)
        a_ = np.sqrt(alpha / evs[0])
        b_ = np.sqrt(alpha / evs[1])
        area = np.pi * a_ * b_
        if area < P_area_min and ps == "optimal":
            P_area_min = area
            P_min = P
            a_opt = a
            Rhat_opt = Rhat
    return P_min, a_opt, Rhat_opt

# Question 1
print("Question 1a")
P_min, a_opt, _ = do_grid_search()
print(f"P={np.round(P_min, 3)} \na={np.round(a_opt, 2)}")
plot_ellipsoid(P_min, 2, r'original $\mathcal{R}$')

print("Question 1b")
c_ = np.array([0.1, 1]).T.reshape((2, -1))
b_ = np.array([[3]])
Yinv_min, a_opt, Rhat_opt = do_grid_search(constraints=True, c=c_, b=b_)
print(f"Yinv={np.round(Yinv_min, 3)} \na={np.round(a_opt, 2)} \nRhat={np.round(Rhat_opt, 3)}")
plot_ellipsoid(Yinv_min, 2, r'$D_1$ constrained $\mathcal{R}$')

print("Question 1c")
c_ = np.array([[0.1, 1],
               [2., -1]]).T.reshape((2, -1))
b_ = np.array([[3], [2*np.sqrt(5)]])
Yinv_min, a_opt, Rhat_opt = do_grid_search(constraints=True, c=c_, b=b_)
print(f"Yinv={np.round(Yinv_min, 3)} \na={np.round(a_opt, 2)} \nRhat={np.round(Rhat_opt, 3)}")
plot_ellipsoid(Yinv_min, 2, r'$D_1$, $D_2$ constrained $\mathcal{R}$')
plot_boundary(c_[:, 0], b_[0], r"$D_1$", [-10, 10], color="orange")
plot_boundary(c_[:, 1], b_[1], r"$D_2$", [-10, 10], color="green")
#plt.title('Reachable configuration space')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.xlim(-6, 6)
plt.ylim(-5, 5)
plt.savefig("./figures4report/a4_question1.png")
plt.show()