import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os


SAVE_FIGS = False
np.random.seed(1)
params = {}
"""centralized problem"""
n_agents = 8
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

        self.w = 0.  # noise at current iterate if private
        Agent.agents.append(self)

    def receive_x(self, k, private=False):
        for i in range(n_agents):
            if i != self.id:
                self.x[i, k] = Agent.agents[i].x[i, k]
                if private:
                    self.x[i, k] += Agent.agents[i].w

    def update_x(self, k, private=False):
        z = np.dot(A[self.id, :], self.x[:, k])
        grad = 2 * (z - self.v)
        gamma = 0.6**(k + 1) * 1  # step size
        self.x[self.id, k + 1] = self.proj(z - gamma * grad)
        if private:
            self.w = self.laplace_noise(k)

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
        C2 = 4/n_agents
        epsilon = params["epsilon"]
        c = 1.
        q = 0.6
        rho = 0.61   # any value in (q, 1)
        b_k = 2 * C2 * np.sqrt(n_agents) * c * rho**(k + 1) / (epsilon * (rho - q))
        w_k = np.random.laplace(0., b_k)  # scaled laplace noise
        return np.clip(w_k, -(-0.002*k+0.1), (-0.002*k+0.1))


v = [0.1, 0.5, 0.4, 0.2, 0.1, 0.5, 0.4, 0.2]
for i in range(n_agents):
    Agent(i, v[i])


def run(private=False):
    for k in range(T - 1):
        for agent in Agent.agents:
            agent.receive_x(k, private=private)
            agent.update_x(k, private=private)

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

print("Question 1")
run(private=False)
plt.figure(1)
for agent in Agent.agents:
    plt.plot(agent.x[agent.id, :], label=("Agent " + str(agent.id + 1)))
plt.plot([0, 50], [0.3, 0.3], "--", label=r"$x^*$")
plt.legend()
plt.xlabel("iteration number")
plt.ylabel("local x value")
plt.tight_layout()
if SAVE_FIGS:
    if not os.path.exists("figures4report"):
        os.mkdir("figures4report")
    plt.savefig(f'figures4report/a1_public_{n_agents}.png')
x_final = np.round(Agent.agents[0].x[0, -1], 2)
print(f"x final is {x_final}, delta is {x_final - 0.3}")
plt.show()

print("\nQuestion 2")
params["epsilon"] = 0.001
run(private=True)
plt.figure(2)
for agent in Agent.agents:
    plt.plot(agent.x[agent.id, :], label=("Agent " + str(agent.id + 1)))
plt.plot([0, 50], [0.3, 0.3], "--", label=r"$x^*$")
plt.legend()
plt.xlabel("iteration number")
plt.ylabel("local x value")
plt.tight_layout()
x_final = np.round(Agent.agents[0].x[0, -1], 2)
print(f"x final is {x_final}, delta is {x_final - 0.3}")
if SAVE_FIGS:
    plt.savefig(f'figures4report/a1_private_{n_agents}_eps_{params["epsilon"]}.png')
else:
    plt.title("Private, epsilon=0.001")
plt.show()

print("\nQuestion 2b")
params["epsilon"] = 0.01
run(private=True)
plt.figure(2)
for agent in Agent.agents:
    plt.plot(agent.x[agent.id, :], label=("Agent " + str(agent.id + 1)))
plt.plot([0, 50], [0.3, 0.3], "--", label=r"$x^*$")
plt.legend()
plt.xlabel("iteration number")
plt.ylabel("local x value")
plt.tight_layout()
x_final = np.round(Agent.agents[0].x[0, -1], 2)
print(f"x final is {x_final}, delta is {x_final - 0.3}")
if SAVE_FIGS:
    plt.savefig(f'figures4report/a1_private_{n_agents}_eps_{params["epsilon"]}.png')
else:
    plt.title("Private, epsilon=0.01")
plt.show()

