import torch
import numpy as np
from torch.utils.data import Dataset
import random

# calculated from train split

PHOTON_MIN = -2.512557
PHOTON_MAX = 2.2779698

ELECTRON_MIN = -2.512557
ELECTRON_MAX = 2.2734396

PHOTON_MEAN = torch.Tensor([0.08224899, 0.08224899]).view(2, 1, 1)
PHOTON_STD = torch.Tensor([0.14533927, 0.15015556]).view(2, 1, 1)

ELECTRON_MEAN = torch.Tensor([0.08224899, 0.08224899]).view(2, 1, 1)
ELECTRON_STD = torch.Tensor([0.14544101, 0.15080675]).view(2, 1, 1)


class PhotonElectronClassificationDataset(Dataset):
    def __init__(self, split="train"):
        self.photon_data = np.load(f"../data/photon_{split}.npy")
        self.electron_data = np.load(f"../data/electron_{split}.npy")

        self.num_samples = self.photon_data.shape[0]

    def __getitem__(self, index):
        # let photon = 0 and electron = 1

        if index >= self.num_samples:
            index = index % self.num_samples

            x = torch.from_numpy(self.electron_data[index]).permute(2, 0, 1)
            y = 1

            x = (x - ELECTRON_MIN) / (ELECTRON_MAX - ELECTRON_MIN)
            x = (x - ELECTRON_MEAN) / ELECTRON_STD

        else:
            x = torch.from_numpy(self.photon_data[index]).permute(2, 0, 1)
            y = 0

            x = (x - PHOTON_MIN) / (PHOTON_MAX - PHOTON_MIN)
            x = (x - PHOTON_MEAN) / PHOTON_STD

        return x, y

    def __len__(self):
        return self.num_samples * 2


class PhotonElectronContrastiveDataset(Dataset):
    def __init__(self, split="train"):
        self.photon_data = np.load(f"../data/photon_{split}.npy")
        self.electron_data = np.load(f"../data/electron_{split}.npy")

        self.num_samples = self.photon_data.shape[0]

    def __getitem__(self, index):
        should_pick_same_class = random.random() > .5

        if index >= self.num_samples:
            index = index % self.num_samples

            x1 = torch.from_numpy(self.electron_data[index]).permute(2, 0, 1)
            x1 = (x1 - ELECTRON_MIN) / (ELECTRON_MAX - ELECTRON_MIN)
            x1 = (x1 - ELECTRON_MEAN) / ELECTRON_STD

            second_index = random.randint(0, self.num_samples - 1)

            if should_pick_same_class:
                x2 = torch.from_numpy(self.electron_data[second_index]).permute(2, 0, 1)
                x2 = (x2 - ELECTRON_MIN) / (ELECTRON_MAX - ELECTRON_MIN)
                x2 = (x2 - ELECTRON_MEAN) / ELECTRON_STD
            else:
                x2 = torch.from_numpy(self.photon_data[second_index]).permute(2, 0, 1)
                x2 = (x2 - PHOTON_MIN) / (PHOTON_MAX - PHOTON_MIN)
                x2 = (x2 - PHOTON_MEAN) / PHOTON_STD

        else:
            x1 = torch.from_numpy(self.photon_data[index]).permute(2, 0, 1)
            x1 = (x1 - PHOTON_MIN) / (PHOTON_MAX - PHOTON_MIN)
            x1 = (x1 - PHOTON_MEAN) / PHOTON_STD

            second_index = random.randint(0, self.num_samples - 1)

            if should_pick_same_class:
                x2 = torch.from_numpy(self.photon_data[second_index]).permute(2, 0, 1)
                x2 = (x2 - PHOTON_MIN) / (PHOTON_MAX - PHOTON_MIN)
                x2 = (x2 - PHOTON_MEAN) / PHOTON_STD

            else:
                x2 = torch.from_numpy(self.electron_data[second_index]).permute(2, 0, 1)
                x2 = (x2 - ELECTRON_MIN) / (ELECTRON_MAX - ELECTRON_MIN)
                x2 = (x2 - ELECTRON_MEAN) / ELECTRON_STD

        return x1, x2, int(should_pick_same_class)

    def __len__(self):
        return self.num_samples * 2
    

if __name__ == "__main__":
    contrastive_dataset = PhotonElectronClassificationDataset(split="test")

    print(contrastive_dataset[0])