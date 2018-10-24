from abc import ABC, abstractmethod


class NetworkModel(ABC):

    @abstractmethod
    def get_model(self):
        pass
