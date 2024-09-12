import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


class SequentialMNIST(Dataset):

    def __init__(self, training: bool = True, normalize: bool = False, permute: bool = False,
                 seed: int = None):
        transform_list = [transforms.ToTensor(), transforms.Lambda(lambda x: x / 255.0)]
        if normalize:
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        transform = transforms.Compose(transform_list)

        self.data = datasets.MNIST(root='./datasets/sequential_mnist/data/',
                                   train=training,
                                   download=True,
                                   transform=transform)
        self.data.data = self.data.data.view(-1, 28 * 28).float()
        self.targets = self.data.targets
        self.data = self.data.data

        if permute:
            if seed is not None:
                np.random.seed(seed)
            permutation = np.random.permutation(28 * 28)
            self.data = self.data[:, permutation]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
