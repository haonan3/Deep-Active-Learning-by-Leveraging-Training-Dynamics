import pickle
import random
import sys
import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from torchvision.transforms import transforms

from src.my_caltech import myCaltech
from src.my_cifar import CIFAR10 as myCIFAR10
from src.my_mnist import myMNIST
from src.my_svhn import mySVHN
from src.utils import read_label_file, read_image_file, enhance, set_seed

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def cf10_load_data(init_label_num, init_label_perC=None, part=False, class_ratio_list=None, train_imbalance=0):
    assert init_label_num is not None or init_label_perC is not None
    assert init_label_num is None or init_label_perC is None

    train_data, train_targets = [], []

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

    for file_name, checksum in train_list:
        file_path = os.path.join(parent_path+'/data/cifar-10-batches-py', file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')

            train_data.append(entry['data'])

            if 'labels' in entry:
                train_targets.extend(entry['labels'])
            else:
                train_targets.extend(entry['fine_labels'])


    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32)
    raw_dim = train_data[0].reshape(-1,).shape[0]
    train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC
    train_targets = np.array(train_targets)
    class_num = max(train_targets) + 1

    if part:
        train_data = train_data[:5000]
        train_targets = train_targets[:5000]


    test_data, test_targets = [], []
    for file_name, checksum in test_list:
        file_path = os.path.join(parent_path+'/data/cifar-10-batches-py', file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')

            test_data.append(entry['data'])

            if 'labels' in entry:
                test_targets.extend(entry['labels'])
            else:
                test_targets.extend(entry['fine_labels'])

    test_data = np.vstack(test_data).reshape(-1, 3, 32, 32)
    test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC
    test_targets = np.array(test_targets)

    train_num = train_data.shape[0]

    if init_label_num is not None:
        if not train_imbalance: # the balanced setting
            split_idx = int(init_label_num)
            idx_range = list(range(train_num))
            random.shuffle(idx_range)
            idx_train = np.array(idx_range[:split_idx])
            idx_pool = np.array(idx_range[split_idx:])
        else: # sample training data according to the class ratio
            idx_array = np.array(list(range(train_num)))
            idx_train = []
            for class_i, ratio_i in enumerate(class_ratio_list):
                idx_array_i = idx_array[train_targets == class_i]
                sampled_idx_i = np.random.choice(idx_array_i, int(ratio_i * init_label_num)+1, replace=False)
                idx_train.append(sampled_idx_i)
            idx_train = np.concatenate(idx_train, axis=0)
            idx_pool = np.array(list(set(idx_array.tolist()) - set(idx_train.tolist())))

    else:
        assert not train_imbalance # image per class, must be balance
        total_idx_list = []
        for i in range(class_num):
            class_ilabel_idxs = list(np.where(train_targets == i)[0])
            random.shuffle(class_ilabel_idxs)
            total_idx_list.extend(class_ilabel_idxs[:init_label_perC])
        idx_train = np.array(total_idx_list)
        idx_pool = np.array(list(set(list(range(train_targets.shape[0]))) - set(total_idx_list)))


    return raw_dim, class_num, (train_data[idx_train], train_targets[idx_train], idx_train), \
           (test_data, test_targets, np.array(list(range(test_targets.shape[0])))),\
           (train_data[idx_pool], train_targets[idx_pool], idx_pool)


def mnist_load_data(init_label_num=None, init_label_perC=None):
    assert init_label_num is not None or init_label_perC is not None
    assert init_label_num is None or init_label_perC is None

    raw_folder = parent_path+'/data/MNIST/raw'
    train=True
    image_file = f"{'train' if train else 't10k'}-images-idx3-ubyte"
    train_data = read_image_file(os.path.join(raw_folder, image_file))

    label_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte"
    train_targets = read_label_file(os.path.join(raw_folder, label_file))

    train=False
    image_file = f"{'train' if train else 't10k'}-images-idx3-ubyte"
    test_data = read_image_file(os.path.join(raw_folder, image_file))

    label_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte"
    test_targets = read_label_file(os.path.join(raw_folder, label_file))


    class_num = max(train_targets) + 1
    #
    train_num = train_data.shape[0]
    if init_label_num is not None:
        split_idx = int(init_label_num)
        idx_range = list(range(train_num))
        random.shuffle(idx_range)
        idx_train = np.array(idx_range[:split_idx])
        idx_pool = np.array(idx_range[split_idx:])
    else:
        total_idx_list = []
        for i in range(class_num):
            class_ilabel_idxs = list(np.where(train_targets == i)[0])
            random.shuffle(class_ilabel_idxs)
            total_idx_list.extend(class_ilabel_idxs[:init_label_perC])
        idx_train = np.array(total_idx_list)
        idx_pool = np.array(list(set(list(range(train_targets.shape[0]))) - set(total_idx_list)))

    raw_dim = train_data[0].reshape(-1,).shape[0]
    #

    return raw_dim, class_num, (train_data[idx_train].numpy(), train_targets[idx_train].numpy(), idx_train), \
           (test_data.numpy(), test_targets.numpy(), np.array(list(range(test_targets.shape[0])))  ),\
           (train_data[idx_pool].numpy(), train_targets[idx_pool].numpy(), idx_pool)


def svhn_load_data(init_label_num=None, init_label_perC=None):
    import scipy.io as sio
    # dataset = SVHN(root=parent_path+'/data/', download=True)

    train_file = parent_path+'/data/SVHN/train_32x32.mat'
    test_file = parent_path + '/data/SVHN/test_32x32.mat'

    loaded_mat_train = sio.loadmat(train_file)
    data_train = loaded_mat_train['X']
    labels_train = loaded_mat_train['y'].astype(np.int64).squeeze()
    np.place(labels_train, labels_train == 10, 0)
    data_train = np.transpose(data_train, (3, 2, 0, 1))

    loaded_mat_test = sio.loadmat(test_file)
    data_test = loaded_mat_test['X']
    labels_test = loaded_mat_test['y'].astype(np.int64).squeeze()
    np.place(labels_test, labels_test == 10, 0)
    data_test = np.transpose(data_test, (3, 2, 0, 1))

    raw_dim = data_train[0].reshape(-1,).shape[0]
    class_num = max(labels_train) + 1

    train_num = data_train.shape[0]

    if init_label_num is not None:
        split_idx = int(init_label_num)
        idx_range = list(range(train_num))
        random.shuffle(idx_range)
        idx_train = np.array(idx_range[:split_idx])
        idx_pool = np.array(idx_range[split_idx:])
    else:
        total_idx_list = []
        for i in range(class_num):
            class_ilabel_idxs = list(np.where(labels_train == i)[0])
            random.shuffle(class_ilabel_idxs)
            total_idx_list.extend(class_ilabel_idxs[:init_label_perC])
        idx_train = np.array(total_idx_list)
        idx_pool = np.array(list(set(list(range(labels_train.shape[0]))) - set(total_idx_list)))


    return raw_dim, class_num, (data_train[idx_train], labels_train[idx_train], idx_train), \
           (data_test, labels_test,  np.array(list(range(labels_test.shape[0])))   ),\
           (data_train[idx_pool], labels_train[idx_pool], idx_pool)


def caltech101_load_data(init_label_num, init_label_perC):
    root = parent_path + '/data/caltech101'
    categories = sorted(os.listdir(os.path.join(root, "101_ObjectCategories")))
    categories.remove("BACKGROUND_Google")  # this is not a real class


    index_all = []
    y = []
    order_index = []
    caltech101_indx_dict = {}
    for (i, c) in enumerate(categories):
        n = len(os.listdir(os.path.join(root, "101_ObjectCategories", c)))
        index_all.extend(range(1, n + 1))
        y.extend(n * [i])
        order_index.extend( list(len(order_index) + np.array(list(range(n)))) )
        caltech101_indx_dict[i] = order_index[-n:]

    y = np.array(y)

    image_array = []
    for idx in order_index:
        img = Image.open(os.path.join(root,
                                      "101_ObjectCategories",
                                      categories[y[idx]],
                                      "image_{:04d}.jpg".format(index_all[idx]))).convert('RGB')

        image_array.append(img)

    class_num = 101
    train_idx, test_idx, pool_idx = [], [], []
    if init_label_num is not None:
        init_label_num = int(init_label_num)
        random.shuffle(order_index)
        train_idx = order_index[:init_label_num]
        test_num = int(0.2*len(index_all))
        test_idx = order_index[init_label_num:init_label_num+test_num]
        pool_idx = order_index[init_label_num+test_num:]
    elif init_label_perC is not None:
        agg_idx = 0
        for k,v in caltech101_indx_dict.items():
            num_in_class = len(v)
            test_num_in_class = 10 # int(0.2*num_in_class)
            temp_list = list(range(agg_idx, agg_idx+num_in_class))
            random.shuffle(temp_list)
            train_idx.extend(temp_list[:init_label_perC])
            test_idx.extend(temp_list[init_label_perC:init_label_perC+test_num_in_class])
            pool_idx.extend(temp_list[init_label_perC+test_num_in_class:])
            agg_idx += num_in_class

    return None, class_num, ([image_array[i] for i in train_idx], y[train_idx], np.array(train_idx)), \
           ([image_array[i] for i in test_idx], y[test_idx], np.array(test_idx)),\
           ([image_array[i] for i in pool_idx], y[pool_idx], np.array(pool_idx))


def load_data(args):
    assert args.train_imbalance + args.test_imbalance + args.pool_imbalance > 0
    set_seed(args.seed)
    if args.dataset_str == 'cf10':
        raw_dim, class_num, train_data, test_data, pool_data = cf10_load_data(init_label_num=args.init_label_num,
                                                        init_label_perC=args.init_label_perC)
        args.input_dim = raw_dim
        args.class_num = class_num
        return train_data, test_data, pool_data
    elif args.dataset_str == 'imbalanced_cf10':
        class_ratio_list = [0.2967, 0.4941, 0.7035, 0.4236, 0.1752, 0.7764, 0.38670932, 0.1355, 0.8960, 0.8871]
        def join_two_data(data_a, data_b):
            return (np.concatenate((data_a[0], data_b[0]), axis=0),
                    np.concatenate((data_a[1], data_b[1]), axis=0),
                    np.concatenate((data_a[2], data_b[2]), axis=0))

        raw_dim, class_num, train_data, test_data, pool_data = cf10_load_data(init_label_num=args.init_label_num,
                                                        init_label_perC=args.init_label_perC,
                                        class_ratio_list=class_ratio_list, train_imbalance=args.train_imbalance)
        print("process for the imbalanced dataset.")
        if args.test_imbalance:
            train_pool_num = len(set(pool_data[2].tolist()).union(set(train_data[2].tolist())))
            test_data = (test_data[0], test_data[1], test_data[2] + train_pool_num)
            jointed_data = join_two_data(pool_data, test_data)
            jointed_data_idx = np.array(range(len(jointed_data[1])))
            assert class_num == len(class_ratio_list)
            joint_train_idx = []
            joint_test_idx = []
            if args.pool_imbalance:
                for i in range(class_num):
                    ratio_i = class_ratio_list[i]
                    class_i_idx_array = jointed_data_idx[jointed_data[1] == i]
                    sampled_idx_i = np.random.choice(class_i_idx_array,
                                    int(ratio_i*len(class_i_idx_array)), replace=False)
                    random.shuffle(sampled_idx_i)
                    test_train_split = int(0.2*len(sampled_idx_i))
                    sampled_idx_i_test = sampled_idx_i[:test_train_split]
                    sampled_idx_i_train = sampled_idx_i[test_train_split:]
                    joint_train_idx.append(sampled_idx_i_train)
                    joint_test_idx.append(sampled_idx_i_test)
                joint_train_idx = np.concatenate(joint_train_idx, axis=0)
                joint_test_idx = np.concatenate(joint_test_idx, axis=0)
                random.shuffle(joint_train_idx)
                random.shuffle(joint_test_idx)
            else:
                del joint_train_idx
                joint_train_idx = np.array(list(set(jointed_data_idx.tolist()) - set(joint_test_idx.tolist())))
            pool_data = (
            jointed_data[0][joint_train_idx], jointed_data[1][joint_train_idx], jointed_data[2][joint_train_idx])
            test_data = (jointed_data[0][joint_test_idx], jointed_data[1][joint_test_idx], jointed_data[2][joint_test_idx])
        else: # balance test
            if args.pool_imbalance:
                sampled_pool_list=[]
                pool_data_idx = np.array(range(len(pool_data[1])))
                for i in range(class_num):
                    ratio_i = class_ratio_list[i]
                    class_i_idx_array = pool_data_idx[pool_data[1] == i]
                    sampled_idx_i = np.random.choice(class_i_idx_array,
                                                     int(ratio_i * len(class_i_idx_array)), replace=False)
                    sampled_pool_list.append(sampled_idx_i)
                sampled_pool_idx = np.concatenate(sampled_pool_list, axis=0)
                pool_data = (pool_data[0][sampled_pool_idx], pool_data[1][sampled_pool_idx],
                    pool_data[2][sampled_pool_idx])
            else: # balance pool
                exit(101)
                pass
        args.input_dim = raw_dim
        args.class_num = class_num
        return train_data, test_data, pool_data

    elif args.dataset_str == 'mnist':
        raw_dim, class_num, train_data, test_data, pool_data = mnist_load_data(init_label_num=args.init_label_num, init_label_perC=args.init_label_perC)
        args.input_dim = raw_dim
        args.class_num = class_num
        return train_data, test_data, pool_data

    elif args.dataset_str == 'svhn':
        raw_dim, class_num, train_data, test_data, pool_data = svhn_load_data(init_label_num=args.init_label_num, init_label_perC=args.init_label_perC)
        args.input_dim = raw_dim
        args.class_num = class_num
        return train_data, test_data, pool_data

    elif args.dataset_str == 'caltech101':
        raw_dim, class_num, train_data, test_data, pool_data = caltech101_load_data(init_label_num=args.init_label_num, init_label_perC=args.init_label_perC)
        args.input_dim = raw_dim
        args.class_num = class_num
        return train_data, test_data, pool_data

    exit(100)



def create_dataloder(args, data, shuffle, certain_batch_size=None):
    bz = args.batch_size if certain_batch_size is None else certain_batch_size

    if 'cf10' in args.dataset_str:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        dataset = myCIFAR10(root='', transform=transform, dataset=data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bz, shuffle=shuffle, num_workers=4)
    elif args.dataset_str == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])
        dataset = myMNIST(root='', transform=transform, dataset=data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bz, shuffle=shuffle, num_workers=4)
    elif args.dataset_str == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
        dataset =mySVHN(root='', dataset=data, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bz, shuffle=shuffle, num_workers=4)
    elif args.dataset_str == 'caltech101':
        transform = transforms.Compose([
            transforms.Scale(384, Image.LANCZOS),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(enhance),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        dataset = myCaltech(root='', dataset=data, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bz, shuffle=shuffle, num_workers=4)

    return loader
