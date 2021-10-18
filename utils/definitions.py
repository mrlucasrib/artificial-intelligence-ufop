from abc import ABC, abstractmethod


class ABCAgent(ABC):
    @abstractmethod
    def act(self):
        pass


class ABCEnvironment(ABC):
    @abstractmethod
    def signal(self, action):
        pass
