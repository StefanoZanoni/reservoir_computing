from abc import ABC, abstractmethod


class TrainingMethod(ABC):
    def __init__(self):
        self.weights = None

    @abstractmethod
    def fit(self, x, y):
        pass
