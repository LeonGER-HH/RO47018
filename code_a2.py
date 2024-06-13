import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import time
from phe import paillier
from helpers import admm_analysis
import copy


class Mediator:
    def __init__(self, x_bar0):
        self.x_bar = [x_bar0]
        self.xi_buffer = {}

        # crypto
        self.pub_k, self.pri_k = paillier.generate_paillier_keypair()
        # for reset
        self._initial_state = copy.deepcopy(self.__dict__)

    def reset(self, agents):
        self.__dict__ = copy.deepcopy(self._initial_state)
        for agent in agents:
            agent.reset()

    def receive_xi(self, agent_id, xi):
        self.xi_buffer[agent_id] = xi

    def calc_xbar(self, encrypted=False):
        n = len(self.xi_buffer)
        x_bar_update = np.sum([self.xi_buffer[agent] for agent in self.xi_buffer.keys()]) / n
        if encrypted:
            x_bar_update = self.pub_k.encrypt(x_bar_update)
        self.x_bar.append(x_bar_update)

    def run_admm(self, agents, encrypted=False, plotting=False):
        iter_max = 18
        times = []
        start_time = time.perf_counter()
        for k in range(iter_max):
            iter_start_time = time.perf_counter()

            for agent in agents:
                xi = agent.update_x(self)
                self.receive_xi(agent.id, xi)

            self.calc_xbar(encrypted=encrypted)

            for agent in agents:
                agent.update_u(self)

            iter_end_time = time.perf_counter()
            times.append(iter_end_time - iter_start_time)
        end_time = time.perf_counter()
        print(f"The whole ADMM took {np.round(end_time - start_time, 2)}s.")
        if plotting:
            admm_analysis(iter_max, agents, self, times)


class Agent:
    agents = []

    def __init__(self, id_, x0, u0=0., q=1):
        self.id = id_
        self.x = [x0]
        self.u = [u0]
        self.q = q
        Agent.agents.append(self)

        # for reset
        self._initial_state = copy.deepcopy(self.__dict__)

    def reset(self):
        self.__dict__ = copy.deepcopy(self._initial_state)

    def update_x(self, mediator):
        rho = 1
        # x_ = cp.Variable(1)
        # # debug start
        # print("debug:")
        # print(1 ** 2 * self.q + rho / 2 * cp.norm(1 - mediator.x_bar[-1] + self.u[-1], 2) ** 2)
        # # debug end
        # cost = x_ ** 2 * self.q + rho / 2 * cp.norm(x_ - mediator.x_bar[-1] + self.u[-1], 2) ** 2
        # problem = cp.Problem(cp.Minimize(cost))
        # problem.solve()
        # self.x.append(x_.value[0])
        # return x_.value[0]
        x_ = rho * (mediator.x_bar[-1] - self.u[-1]) / (2 * self.q + rho)
        if isinstance(x_, paillier.EncryptedNumber):
            x_ = mediator.pri_k.decrypt(x_)
        self.x.append(x_)
        return x_

    def update_u(self, mediator):
        u_ = self.u[-1] + self.x[-1] - mediator.x_bar[-1]
        self.u.append(u_)
        return u_


iter_max = 18
rho = 1

agent1 = Agent(1, 1)
agent2 = Agent(2, 0.3)
agent3 = Agent(3, 0.1)
mediator = Mediator(0.5)

print("\nQuestion 1")
mediator.run_admm(Agent.agents, plotting=True)

print("\nQuestion 2")
mediator.reset(Agent.agents)
mediator.run_admm(Agent.agents, encrypted=True, plotting=True)

print("\nQuestion 3")
mediator.reset(Agent.agents)


