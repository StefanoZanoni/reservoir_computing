import torch
from torch.utils.data import Dataset


class MemoryCapacity(Dataset):
    def __init__(self, k: int, training: bool = True, seed: int = None):

        if training:
            sequence_length = 1000 + k
        else:
            sequence_length = 200 + k

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        time_series = 2 * torch.rand(sequence_length, generator=generator, dtype=torch.float32) - 1
        self.data = time_series[k:]
        self.target = time_series[:-k]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data, self.target
