from training_method.tm import TrainingMethod
from sklearn.linear_model import SGDRegressor
import torch


class LSM(TrainingMethod):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:

        # Flatten the input and target data for fitting
        x_flat = x.view(-1, x.shape[-1]).numpy()
        y_flat = y.view(-1).numpy()

        # Initialize the SGDRegressor with LMS settings
        model = SGDRegressor(loss='squared_loss', learning_rate='constant', eta0=self.learning_rate)

        # Fit the model
        model.fit(x_flat, y_flat)

        # Store the fitted weights
        return torch.tensor(model.coef_, dtype=torch.float32)
