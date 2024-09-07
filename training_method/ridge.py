import torch

from sklearn.linear_model import Ridge
from .tm import TrainingMethod


class RidgeRegression(TrainingMethod):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
        temp_x = x.cpu().numpy()
        temp_y = y.cpu().numpy()
        self.model.fit(temp_x, temp_y)
        return torch.tensor(self.model.coef_, dtype=torch.float32)
