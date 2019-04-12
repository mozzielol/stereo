import tensorflow as tf
import sonnet as snt
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from copy import deepcopy
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import np_utils
import os


def mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def permute(task, seed):
    np.random.seed(seed)
    perm = np.random.permutation(task.train._images.shape[1])
    permuted = deepcopy(task)
    permuted.train._images = permuted.train._images[:, perm]
    permuted.test._images = permuted.test._images[:, perm]
    permuted.validation._images = permuted.validation._images[:, perm]
    return permuted




def construct_permute_mnist(num_tasks=2,  split='train', permute_all=False, subsample=1):
    """Create permuted MNIST tasks.
        Args:
                num_tasks: Number of tasks
                split: whether to use train or testing data
                permute_all: When set true also the first task is permuted otherwise it's standard MNIST
                subsample: subsample by so much
        Returns:
            List of (X, y) tuples representing each dataset
    """
    # Load MNIST data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_train, y_train = X_train[::subsample], y_train[::subsample]
    X_test, y_test = X_test[::subsample], y_test[::subsample]

    permutations = []
    # Generate random permutations
    for i in range(num_tasks):
        idx = np.arange(X_train.shape[1],dtype=int)
        if permute_all or i>0:
            np.random.shuffle(idx)
        permutations.append(idx)

    both_datasets = []
    for (X, y) in ((X_train, y_train), (X_test, y_test)):
        datasets = []
        for perm in permutations:
            data = X[:,perm], np_utils.to_categorical(y, nb_classes)
            datasets.append(data)
        both_datasets.append(datasets)
    return both_datasets


def construct_split_cifar10(task_labels,  split='train'):
    """Split CIFAR10 dataset by labels.
        Args:
            task_labels: list of list of labels, one for each dataset
            split: whether to use train or testing data
        Returns:
            List of (X, y) tuples representing each dataset
    """
    # Load CIFAR10 data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # X_train = X_train.reshape(-1, 3, 32, 32)
    # X_test = X_test.reshape(-1, 32**2)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    no = X_train.max()
    X_train /= no
    X_test /= no

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    return split_dataset_by_labels(X, y, task_labels, nb_classes)


def construct_split_mnist(task_labels,  split='train', multihead=False):
    """Split MNIST dataset by labels.
        Args:
                task_labels: list of list of labels, one for each dataset
                split: whether to use train or testing data
        Returns:
            List of (X, y) tuples representing each dataset
    """
    # Load MNIST data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    return split_dataset_by_labels(X, y, task_labels, nb_classes, multihead)


def split_dataset_by_labels(X, y, task_labels, nb_classes=None, multihead=False):
    """Split dataset by labels.
    Args:
        X: data
        y: labels
        task_labels: list of list of labels, one for each dataset
        nb_classes: number of classes (used to convert to one-hot)
    Returns:
        List of (X, y) tuples representing each dataset
    """
    if nb_classes is None:
        nb_classes = len(np.unique(y))
    datasets = []
    for labels in task_labels:
        idx = np.in1d(y, labels)
        if multihead:
            label_map = np.arange(nb_classes)
            label_map[labels] = np.arange(len(labels))
            data = X[idx], np_utils.to_categorical(label_map[y[idx]], len(labels))
        else:
            data = X[idx], np_utils.to_categorical(y[idx], nb_classes)
        datasets.append(data)
    return datasets

def load_mnist(split='train'):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test
    nb_classes = 10
    y = np_utils.to_categorical(y, nb_classes)
    return X, y


















