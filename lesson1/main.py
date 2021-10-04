from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from lesson1.utils import IntemOfSaleBehavior, AgentPercepts, AgentBeliefState, Controller


class Environment:
    """
    Ambiente
    """

    def __init__(self, item_of_sale_behavior: IntemOfSaleBehavior):
        self.item_of_sale_behavior = item_of_sale_behavior

    def initial_percepts(self) -> AgentPercepts:
        """
        Percepções iniciais
        :return: retorna as percepções iniciais
        """
        return AgentPercepts(n=self.item_of_sale_behavior.n,
                             price=self.item_of_sale_behavior.price,
                             max_n=self.item_of_sale_behavior.max_n)

    def signal(self, action: dict) -> AgentPercepts:
        """
        Envia um sinal para o ambiente
        :param action: Sinal a ser enviado
        :return: Retorna a percepção atual do ambiente
        """
        usage = np.random.normal(self.item_of_sale_behavior.mu_usage[
                                     self.item_of_sale_behavior.clock % len(self.item_of_sale_behavior.mu_usage)],
                                 self.item_of_sale_behavior.sigma_usage[
                                     self.item_of_sale_behavior.clock % len(self.item_of_sale_behavior.sigma_usage)])
        bought = action['to_buy']

        self.item_of_sale_behavior.n = self.item_of_sale_behavior.n - usage + bought

        if self.item_of_sale_behavior.clock % 7 == 0:
            self.item_of_sale_behavior.price = 1.2

            self.item_of_sale_behavior.on_sale = True if np.random.rand() < 0.5 else False

            if self.item_of_sale_behavior.on_sale:
                self.item_of_sale_behavior.price -= self.item_of_sale_behavior.sigma_price
            else:
                self.item_of_sale_behavior.price = max(
                    np.random.normal(self.item_of_sale_behavior.mu_price, self.item_of_sale_behavior.sigma_price), 0.9)

        self.item_of_sale_behavior.clock += 1
        return AgentPercepts(n=self.item_of_sale_behavior.n,
                             price=self.item_of_sale_behavior.price,
                             max_n=self.item_of_sale_behavior.max_n)


class ThirdController(Controller):
    def command(self, low_level_perceptions: AgentPercepts, hight_levels_command: dict,
                belief_state: AgentBeliefState) -> dict:
        hight_levels_command['to_buy'] = 0
        return hight_levels_command


class SecondController(Controller):
    def command(self, low_level_perceptions: AgentPercepts, hight_levels_command: dict,
                belief_state: AgentBeliefState) -> dict:
        if low_level_perceptions.price <= belief_state.low:
            hight_levels_command['to_buy'] = low_level_perceptions.max_n - low_level_perceptions.n
            return hight_levels_command
        else:
            return ThirdController().perceive(low_level_perceptions, hight_levels_command,
                                      belief_state)


class FirstController(Controller):
    def command(self, low_level_perceptions: AgentPercepts, hight_levels_command: dict,
                belief_state: AgentBeliefState) -> dict:
        if low_level_perceptions.n <= belief_state.low:
            hight_levels_command['to_buy'] = belief_state.min - low_level_perceptions.n
            return hight_levels_command
        else:
            return SecondController().perceive(low_level_perceptions, hight_levels_command,
                                       belief_state)


class Agent:
    """
    Classe agente
    """

    def __init__(self, env: Environment):
        self.env = env
        self.percepts = env.initial_percepts()
        self.spendings = []
        self.clock = 1
        self.belief_state = AgentBeliefState(average_price=self.percepts.price,
                                             cheap=self.percepts.price,
                                             low=0,
                                             min=self.percepts.max_n)
        self.controller = FirstController()
        self.body = self.Body()

    class Body:
        def act(self, command):
            return command

        def recive_stimuli(self, low_level_perceptions: AgentPercepts, hight_levels_command: dict,
                           belive_state: AgentBeliefState, controller: FirstController) -> dict:
            response_command = controller.perceive(low_level_perceptions, hight_levels_command,
                                                   belive_state)
            return self.act(response_command)

    def send_body_stimuli(self):
        """
        Envia o estimulo para o corpo
        """
        response_body_command = self.body.recive_stimuli(self.percepts, {}, self.belief_state, self.controller)
        self.update_agent_belief_state(self.env.signal(response_body_command), response_body_command['to_buy'])

    def update_agent_belief_state(self, percepts: AgentPercepts, to_buy):
        self.spendings.append(to_buy * self.percepts.price)
        self.percepts = percepts
        self.belief_state.average_price = (self.belief_state.average_price * self.clock + self.percepts.price) / (
                self.clock + 1)
        self.belief_state.cheap = self.percepts.price
        self.belief_state.low = 100
        self.belief_state.min = self.percepts.max_n
        self.clock += 1


if __name__ == "__main__":

    env = Environment(IntemOfSaleBehavior(
        n=0,
        price=1.2,
        mu_usage=[10, 100, 150, 300, 125, 50, 15],
        sigma_usage=[2, 10, 10, 20, 10, 10, 2],
        mu_price=1.2,
        sigma_price=0.2,
        on_sale=False,
        max_n=1500)
    )
    ag = Agent(env)

    prices = []
    n = []

    for i in range(1000):
        ag.send_body_stimuli()
        prices.append(env.item_of_sale_behavior.price)
        n.append(env.item_of_sale_behavior.n)

    plt.plot(prices)
    plt.show()

    plt.plot(n)
    plt.show()

    plt.plot(ag.spendings)
    plt.show()
