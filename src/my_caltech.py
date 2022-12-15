import torch
from torchvision.datasets import VisionDataset

from PIL import Image
import numpy as np
import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class myCaltech(VisionDataset):

    def __init__(self, root, dataset, transform=None, target_transform=None):
        super(myCaltech, self).__init__(root, transform=transform, target_transform=target_transform)

        # (self.data, self.targets) = dataset
        self.data = dataset[0]
        self.targets = dataset[1]
        self.g_idx = dataset[2]

        self.num_sample = len(self.data)
        self.num_class = max(self.targets) + 1

        self.idxs = torch.tensor(range(self.num_sample))


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        img, target, idx_array = self.data[index], int(self.targets[index]), self.idxs[index]

        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return idx_array, img, target

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        return len(self.data)