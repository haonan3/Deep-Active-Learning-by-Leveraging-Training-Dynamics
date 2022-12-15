# Deep Active Learning by Leveraging Training Dynamics

This repository is the PyTorch implementation of dynamicAL (NeurIPS 2022).

[OpenReview](https://openreview.net/pdf?id=aJ5xc1QB7EX)

:camera: If you make use of the code/experiment, please cite our paper (Bibtex below).

```
@inproceedings{
wang2022deep,
title={Deep Active Learning by Leveraging Training Dynamics},
author={Haonan Wang and Wei Huang and Ziwei Wu and Hanghang Tong and Andrew J Margenot and Jingrui He},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=aJ5xc1QB7EX}
}
```

## :mailbox_with_no_mail: Contact
Contact: Haonan Wang (haonan3@illinois.edu)

Don't hesitate to send us an e-mail if you have any question.

We're also open to any collaboration!


## :wrench: Installation
Install required Python packages
```console
$  pip install requirements.txt
```


Manually download datasets and put them under the `data` folder as the following, or leave the dataset preparation to pytorch.
```console
$ mkdir data
$ cd data
$ mkdir SVHN
$ cd SVHN
$ wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
$ wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
$ wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
$ cd ..
$ wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
$ wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
$ tar -xf cifar-10-python.tar.gz
$ tar -xf 101_ObjectCategories.tar.gz
```

## :hammer: Test run
We provide a demo to run the code, please check `run.sh`.
```console
$  sh run.sh
```

After running, you can check the result through
```console
$  tensorboard --logdir=runs
```

Default parameters are not the best performing-hyper-parameters. Hyper-parameters need to be specified through the commandline arguments.