import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os


SAVE_FIGS = False

"""centralized problem"""
n_agents = 4
x = cp.Variable(1)
v = np.array([0.1, 0.5, 0.4, 0.2])
cost = cp.quad_form(x - v, np.eye(4))
constraints = [-1. <= x]
constraints += [x <= 1.]
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve()

# Print result.
print("-------------------")
print("Centralized problem")
print(f"The {prob.status} solution is", x.value)
print("-------------------")

"""distributed projected gradient method"""
T = 50  # number of iterations
eta = 1 / n_agents
A = eta * np.ones((n_agents, n_agents))
#A = 0.5 / 3 * np.ones((n_agents, n_agents)) + 1 / 3 * np.eye(n_agents)
#A = 0.3 * np.ones((n_agents, n_agents)) - 0.2 * np.eye(n_agents)
assert np.allclose(np.sum(A, axis=0), np.ones(n_agents)), "Connectivity matrix is not doubly stochastic"
assert np.allclose(np.sum(A, axis=1), np.ones(n_agents)), "Connectivity matrix is not doubly stochastic"


class Agent:
    agents = []

    def __init__(self, id, v):
        self.id = id
        self.x = np.empty((n_agents, T))
        self.x[id, 0] = v
        self.v = v
        Agent.agents.append(self)

    def receive_x(self, k, private=False):
        for i in range(n_agents):
            if i != self.id:
                self.x[i, k] = Agent.agents[i].x[i, k]
                if private:
                    self.x[i, k] += self.laplace_noise(k)

    def update_x(self, k):
        z = np.dot(A[self.id, :], self.x[:, k])
        grad = 2 * (z - self.v)
        gamma = 0.6**(k + 1) * 1  # step size
        self.x[self.id, k + 1] = self.proj(z - gamma * grad)

    @staticmethod
    def proj(p):
        if -1. <= p <= 1.:
            return p
        elif p < -1.:
            return -1.
        else:
            return 1.

    @staticmethod
    def laplace_noise(k):
        C2 = 1 #0.00025 # 0.0001
        epsilon = 0.01
        c = 1.
        q = 0.6
        rho = 0.61   # any value in (q, 1)
        b_k = 2 * C2 * np.sqrt(n_agents) * c * rho**(k + 1) / (epsilon * (rho - q))
        w_k = np.random.laplace(0., b_k)  # scaled laplace noise
        return w_k



v = [0.1, 0.5, 0.4, 0.2, 0.1, 0.5, 0.4, 0.2]
for i in range(n_agents):
    Agent(i, v[i])


def run(private=False):
    for k in range(T - 1):
        for agent in Agent.agents:
            agent.receive_x(k, private=private)
            agent.update_x(k)

    x_final = Agent.agents[0].x[:, -1]
    x_tilde = np.average(x_final)

    return x_tilde

# x_tildes = []
# for i in range(1000):
#     x_tildes.append(run(private=True))
#     print(f"iteration counter {i}")
# plt.hist(x_tildes)
# plt.xlim([-1, 1])
# plt.show()

# plot the results
run(private=True)
plt.figure(1)
for agent in Agent.agents:
    plt.plot(agent.x[agent.id, :], label=("Agent " + str(agent.id + 1)))

plt.legend()
plt.xlabel("iteration")
plt.ylabel("x value")
if SAVE_FIGS:
    if not os.path.exists("figures4report"):
        os.mkdir("figures4report")
    plt.savefig('figures4report/distributed_public_consensus_4.png')
plt.show()

""""""

