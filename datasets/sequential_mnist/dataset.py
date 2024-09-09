import numpy as np
import torch
from torch.utils.data import Dataset


class SequentialMNIST(Dataset):

    def __init__(self, training: bool = True, normalize: bool = True):
        data = np.load('datasets/sequential_mnist/sequential_mnist_dataset/sequential_mnist.npz')

        if training:
            x = data['x_train']
            y = data['y_train']
        else:
            x = data['x_test']
            y = data['y_test']

        if normalize:
            x = x.astype(np.float32)
            x /= 255.0

        self.data = torch.tensor(x, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
