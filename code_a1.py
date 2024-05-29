import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

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
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print(f"The problem is {prob.status}")

"""distributed projected gradient method"""
n_agents = 4
T = 50  # number of iterations
eta = 1 / n_agents
A = eta * np.ones((n_agents, n_agents))
#A = 0.5 / 3 * np.ones((n_agents, n_agents)) + 1 / 3 * np.eye(n_agents)
print(np.sum(A, axis=0), np.sum(A, axis=1), np.ones(n_agents), np.array_equal(np.sum(A, axis=0), np.ones(n_agents)))
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

    def receive_x(self, k):
        for i in range(n_agents):
            if i != self.id:
                self.x[i, k] = Agent.agents[i].x[i, k]

    def update_x(self, k):
        z = np.dot(A[self.id, :], self.x[:, k])
        print(z)
        raise
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

v = [0.1, 0.5, 0.4, 0.2, 0.1, 0.5, 0.4, 0.2]
for i in range(n_agents):
    Agent(i, v[i])
plt.figure(1)
for k in range(T - 1):
    for agent in Agent.agents:
        agent.receive_x(k)
        agent.update_x(k)

# plot the results
for agent in Agent.agents:
    plt.plot(agent.x[agent.id, :], label=("Agent " + str(agent.id + 1)))

plt.legend()
plt.xlabel("iteration")
plt.ylabel("x value")
plt.show()

""""""

