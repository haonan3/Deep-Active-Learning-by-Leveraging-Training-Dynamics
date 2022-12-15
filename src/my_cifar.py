from __future__ import print_function

import torch
from PIL import Image
import os
import os.path
import numpy as np
import sys

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle



class CIFAR10(VisionDataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root=None, transform=None, target_transform=None, dataset=None):
        assert root is not None or dataset is not None

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        # (self.data, self.targets) = dataset
        self.data = dataset[0]
        self.targets = dataset[1]
        self.g_idx = dataset[2]

        self.idxs = torch.tensor(range(self.data.shape[0]))

        self.num_sample = self.data.shape[0]
        self.num_class = max(self.targets) + 1


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, idx_array = self.data[index], self.targets[index], self.idxs[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return idx_array, img, target

    def __len__(self):
        return len(self.data)