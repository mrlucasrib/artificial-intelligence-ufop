from dataclasses import dataclass, field
from typing import List


@dataclass
class IntemOfSaleBehavior:
    n: int
    price: float
    mu_usage: List[int]
    sigma_usage: List[int]
    mu_price: float
    sigma_price: float
    on_sale: bool
    max_n: int
    clock: int = field(default=0)


@dataclass
class AgentBeliefState:
    average_price: float
    cheap: float
    low: float
    min: int


@dataclass
class AgentPercepts:
    n: int
    price: float
    max_n: int


class Controller:

    def remerber(self, low_level_perceptions: AgentPercepts, belief_state: AgentBeliefState,
                 hight_levels_command: dict):
        return self.command(low_level_perceptions, hight_levels_command, belief_state)

    def command(self, low_level_perceptions: AgentPercepts, hight_levels_command: dict,
             belive_state: AgentBeliefState) -> dict:
        raise NotImplemented()

    def perceive(self, low_level_perceptions: AgentPercepts, hight_levels_command: dict,
                 belive_state: AgentBeliefState):
        return self.remerber(low_level_perceptions, belive_state, hight_levels_command)
