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

We provide a demo to run the code, please check `run.sh`.
```console
$  sh run.sh
```

After running, you can check the result through
```console
$  tensorboard --logdir=runs
```