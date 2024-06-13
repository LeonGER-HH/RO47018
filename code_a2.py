import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import time
from phe import paillier


class Mediator:
    def __init__(self, x_bar0):
        self.x_bar = [x_bar0]


class Agent:
    agents = []

    def __init__(self, x0, u0=0., q=1):
        self.x = [x0]
        self.u = [u0]
        self.q = q
        Agent.agents.append(self)


iter_max = 18
rho = 1

agent1 = Agent(1)
agent2 = Agent(0.3)
agent3 = Agent(0.1)
mediator = Mediator(0.5)

times = []
start_time = time.perf_counter()
for k in range(iter_max):
    iter_start_time = time.perf_counter()
    for agent in Agent.agents:
        x_update = cp.Variable(1)
        cost = x_update**2 * agent.q + rho / 2 * cp.norm(x_update - mediator.x_bar[-1] + agent.u[-1], 2)**2
        problem = cp.Problem(cp.Minimize(cost))
        problem.solve()
        agent.x.append(x_update.value[0])

    # xbar update
    n = len(Agent.agents)
    x_bar_update = 1 / n * np.sum([agent.x[-1] for agent in Agent.agents])
    mediator.x_bar.append(x_bar_update)

    for agent in Agent.agents:
        u_update = agent.u[-1] + agent.x[-1] - mediator.x_bar[-1]
        agent.u.append(u_update)
    iter_end_time = time.perf_counter()
    times.append(iter_end_time - iter_start_time)
end_time = time.perf_counter()
print(f"The whole iteration took {np.round(end_time - start_time, 2)}s.")

x_ticks = np.arange(0, iter_max + 1)

print("Question 1")
plt.figure(1)
plt.plot(x_ticks, agent1.x, label=r"$x_1$")
plt.plot(x_ticks, agent2.x, label=r"$x_2$")
plt.plot(x_ticks, agent3.x, label=r"$x_3$")
plt.plot(x_ticks, mediator.x_bar, label=r"$\bar{x}$")
plt.xlabel("iteration number")
plt.legend()
plt.show()

plt.figure(2)
plt.plot(np.arange(0, iter_max), times)
plt.xlabel("iteration number")
plt.ylabel("time [s]")
plt.show()

