import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import control as ct

from helpers import plot_ellipsoid, plot_boundary


def find_reachable_set(F, G, R, a):
    P = cp.Variable(F.shape, symmetric=True)
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


def find_safe_reachable_set(F, G, R, a, c, b):
    Rhat = cp.Variable(R.shape, symmetric=True)
    Y = cp.Variable(F.shape, symmetric=True)

    obj = cp.Minimize(cp.trace(Rhat))
    matrix = cp.bmat([[a * Y, np.zeros((F.shape[0], G.shape[1])), Y @ F.T],
                      [np.zeros((G.shape[1], F.shape[0])), (1 - a) * Rhat, G.T],
                      [F @ Y, G, Y]])
    const = []
    for i in range(len(b)):
        const += [c[:, i].T @ Y @ c[:, i] <= b[i]**2 / R.shape[0]]
    const += [matrix >> 0]
    const += [Rhat == cp.diag(cp.diag(Rhat))]
    const += [Rhat >> R]
    slack = np.eye(Y.shape[0]) * 1e-5
    const += [Y + slack >> 0]
    slack2 = np.eye(Rhat.shape[0]) * 1e-5
    const += [Rhat + slack2 >> 0]

    problem = cp.Problem(obj, const)
    problem.solve(solver=cp.SCS, max_iters=100000)
    #print(f"The optimization problem is {problem.status}.")
    Y_inv = np.linalg.inv(Y.value)
    return Y_inv, Rhat.value, problem.status


def do_grid_search(F, G, R, alpha, a_range=[0.01, 0.99, 10], constraints=False, **kwargs):
    P_area_min = np.inf
    P_min = None
    a_opt = None
    Rhat_opt = None
    for a in np.linspace(a_range[0], a_range[1], int(a_range[2])):
        if not constraints:
            P, ps = find_reachable_set(F, G, R, a)
            Rhat = 0
        else:
            P, Rhat, ps = find_safe_reachable_set(F, G, R, a, kwargs['c'], kwargs['b'])
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


"""Question 1"""
F = np.array([[0.84, 0.23],
              [-0.47, 0.12]])
G = np.array([[0.07, 0.3],
              [0.23, 0.1]])
gamma_1, gamma_2 = 8, 10
R = np.diag([1 / gamma_1, 1 / gamma_2])
n_inputs = 2
print(f"R={R}")

print("Question 1a")
P_min, a_opt, _ = do_grid_search(F, G, R, n_inputs)
print(f"P={np.round(P_min, 3)} \na={np.round(a_opt, 2)}")
plot_ellipsoid(P_min, n_inputs, r'original $\mathcal{R}$')

print("Question 1b")
c_ = np.array([0.1, 1]).T.reshape((2, -1))
b_ = np.array([[3]])
Yinv_min, a_opt, Rhat_opt = do_grid_search(F, G, R, n_inputs, a_range=[0.01, 0.99, 30], constraints=True, c=c_, b=b_)
print(f"Yinv={np.round(Yinv_min, 3)} \na={np.round(a_opt, 2)} \nRhat={np.round(Rhat_opt, 3)}")
plot_ellipsoid(Yinv_min, n_inputs, r'$D_1$ constrained $\mathcal{R}$')

# print("Question 1c")
# c_ = np.array([[0.1, 1],
#                [2., -1]]).T.reshape((2, -1))
# b_ = np.array([[3], [2*np.sqrt(5)]])
# Yinv_min, a_opt, Rhat_opt = do_grid_search(F, G, R, n_inputs, constraints=True, c=c_, b=b_)
# print(f"Yinv={np.round(Yinv_min, 3)} \na={np.round(a_opt, 2)} \nRhat={np.round(Rhat_opt, 3)}")
# plot_ellipsoid(Yinv_min, n_inputs, r'$D_1$, $D_2$ constrained $\mathcal{R}$')
# plot_boundary(c_[:, 0], b_[0], r"$D_1$", [-10, 10], color="orange")
# plot_boundary(c_[:, 1], b_[1], r"$D_2$", [-10, 10], color="green")
# #plt.title('Reachable configuration space')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.xlim(-6, 6)
plt.ylim(-5, 5)
#plt.savefig("./figures4report/a4_question1.png")
plt.show()


# """Question 2"""
dt = 0.5
d_star = 1.
v_star = 60 * 3.6  #
beta = -0.1
kp = 0.2
kd = 0.3
omega1_max = 1.1
omega2_max = 0.9
omega3_max = 1.05
gammas = [1.2, 0.8, 1.1]
F = np.array([[1, 0, -dt, dt, 0],
              [0, 1, 0, -dt, dt],
              [kp, 0, (1 + beta) - kd, kd, 0],
              [-kp, kp, kd, (1 + beta) - 2 * kd, kd],
              [0, -kp, 0, kd, (1 + beta) - kd]])
G = np.hstack([np.zeros((2, 3)).T, dt * np.eye(3)]).T
R = np.diag([1 / gamma for gamma in gammas])
n_inputs = 3
#
# print("Question 2a")
# P_min, a_opt, _ = do_grid_search(F, G, R, n_inputs, a_range=[0.88, 0.89, 1])
# print(f"P={np.round(P_min, 3)} \nR={np.round(R, 3)} \na={np.round(a_opt, 2)}")
# plot_ellipsoid(P_min[:2, :2], n_inputs, r'original $\mathcal{R}$')
#
# print("Question 2b")
# c_ = np.array([[-1, 0, 0, 0, 0],
#                [0, -1, 0, 0, 0]]).T #.reshape((5, -1))
# b_ = np.array([[1.], [1.]])
# Yinv_min, a_opt, Rhat_opt = do_grid_search(F, G, R, n_inputs, a_range=[0.88, 0.89, 1], constraints=True, c=c_, b=b_)
# print(f"Yinv={np.round(Yinv_min, 3)} \na={np.round(a_opt, 2)} \nRhat={np.round(Rhat_opt, 3)}")
# plot_ellipsoid(Yinv_min[3:, 3:], n_inputs, r'$D_1$, $D_2$ constrained $\mathcal{R}$')
# plot_boundary(c_[:2, 0].copy(), b_[0], r"$D_1$", [-10, 10], color="orange")
# plot_boundary(c_[:2, 1].copy(), b_[1], r"$D_2$", [-10, 10], color="green")
#
#
# plt.xlabel(r'$\tilde{d}_1$')
# plt.ylabel(r'$\tilde{d}_2$')
# plt.legend()
# plt.grid(True)
# plt.xlim(-6, 6)
# plt.ylim(-6, 6)
# plt.savefig("./figures4report/a4_question2.png")
# plt.show()
#
# print("Question 2c")
# K1, _, _ = ct.dlqr(F, G, P_min, R)
# Yinv_min = np.round(Yinv_min, 5)
# K2, _, _ = ct.dlqr(F, G, Yinv_min, Rhat_opt)
#
# print(f"K1 LQR gain is: {np.round(K1, 3)}", "\n", f"K2 LQR gain is: {np.round(K2, 3)}")
# x0 = np.array([-6, -6, 16, 16, 16]).reshape((-1, 1))
# T = 40  # [s]
# n_timesteps = int(T / dt)
#
# x1 = np.hstack([x0, np.zeros((5, n_timesteps - 1))])
# u1 = np.zeros((3, n_timesteps - 1))
# x2 = np.hstack([x0, np.zeros((5, n_timesteps - 1))])
# u2 = np.zeros((3, n_timesteps - 1))
#
# def acc_limits(u):
#     u_filtered = np.zeros(3)
#     u_filtered[0] = np.clip(u[0], -1.1, 1.1)
#     u_filtered[1] = np.clip(u[1], -0.9, 0.9)
#     u_filtered[2] = np.clip(u[2], -1.05, 1.05)
#     return u_filtered
#
# def acc_limits_safe(u):
#     u_filtered = np.zeros(3)
#     limit1, limit2, limit3 = 1/Rhat_opt[0, 0], 1/Rhat_opt[1, 1], 1/Rhat_opt[2, 2]
#     u_filtered[0] = np.clip(u[0], -limit1, limit1)
#     u_filtered[1] = np.clip(u[1], -limit2, limit2)
#     u_filtered[2] = np.clip(u[2], -limit3, limit3)
#     return u_filtered
#
# attack = False
# for k in range(n_timesteps - 1):
#     u1[:, k] = - K1 @ x1[:, k]
#     if attack:
#         u1[0, k] = 0.6681
#     u1[:, k] = acc_limits(u1[:, k])
#     x1[:, k + 1] = F @ x1[:, k] + G @ u1[:, k]
#
#     u2[:, k] = - K2 @ x2[:, k]
#     if attack:
#         u2[0, k] = 0.6681
#     u2[:, k] = acc_limits_safe(u2[:, k])
#     x2[:, k + 1] = F @ x2[:, k] + G @ u2[:, k]
#     if k >= 40:
#         attack = True
#
# # plot the results
# plt.figure(2)
# time = np.linspace(0, T, n_timesteps)
# plt.plot(time, x1[0, :], label=r"K1 $\tilde{d_1}$")
# plt.plot(time, x1[1, :], label=r"K1 $\tilde{d_2}$")
# plt.gca().set_prop_cycle(None)
# plt.plot(time, x2[0, :], "--", label=r"K2 $\tilde{d_1}$")
# plt.plot(time, x2[1, :], "--", label=r"K2 $\tilde{d_2}$")
# plt.plot([0, T], [-1, -1], linewidth=2, label="safety border")
# plt.xlabel("time [s]")
# plt.ylabel(r"distance error [m]")
# plt.grid(True)
# plt.legend()
# plt.savefig("./figures4report/a4_question3_state_trajectories.png")
#
# plt.show()
# plt.figure(3)
# plt.plot(time[:-1], u1[0, :], label=r"K1 $\omega_1$")
# plt.plot(time[:-1], u1[1, :], label=r"K1 $\omega_2$")
# plt.plot(time[:-1], u1[2, :], label=r"K1 $\omega_3$")
# plt.gca().set_prop_cycle(None)
# plt.plot(time[:-1], u2[0, :], "--", label=r"K2 $\omega_1$")
# plt.plot(time[:-1], u2[1, :], "--", label=r"K2 $\omega_2$")
# plt.plot(time[:-1], u2[2, :], "--", label=r"K2 $\omega_3$")
# plt.xlabel("time [s]")
# plt.ylabel(r"acceleration [m/$s^2$]")
#
# plt.legend()
# plt.savefig("./figures4report/a4_question3_control_effort.png")
# plt.show()
