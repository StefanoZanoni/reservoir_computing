import torch
from torch.utils.data import Dataset


class MC(Dataset):
    def __init__(self, k: int, training: bool = True):
        if training:
            sequence_length = 1000 + k
        else:
            sequence_length = 200 + k
        time_series = 2 * torch.rand(sequence_length) - 1
        self.data = time_series[k:]
        self.target = time_series[:-k]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
