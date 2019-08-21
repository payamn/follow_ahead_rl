import torch
from torch.utils.data import Dataset
import torchvision

import os
import pickle
import numpy as np

class Follow_Ahead_Dataset(Dataset):
    """ datset of lidar data and future position of person"""

    def __init__(self, dataset_dir):
        """
        Args:
            dataset_dir (string): Path to the dataset file with annotations.
        """
        self.dataset_dir = dataset_dir
        self.files = []
        for r, d, f in os.walk(self.dataset_dir):
            for file in f:
                if '.pkl' in file:
                    self.files.append(os.path.join(r, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(self.files[idx], 'rb')as f:
            data = pickle.load(f)
            image = data[0][0]
            image = np.transpose(image, (2, 0, 1))
            image = np.concatenate((image, image, image))

        return torch.from_numpy(np.asarray(image)), torch.from_numpy(np.asarray(data[0][1])), torch.from_numpy(np.asarray(data[1]))