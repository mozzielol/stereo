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
from six.moves import urllib

def download(download_link, file_name, expected_bytes):
    """ Download the pretrained VGG-19 model if it's not already downloaded """
    if os.path.exists(file_name):
        print("VGG-19 pre-trained model is ready")
        return
    print("Downloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded VGG-19 pre-trained model', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')

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

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    X,y = X_train,y_train
    train_set = []
    for perm in permutations:
        data = X[:,perm], np_utils.to_categorical(y, nb_classes)
        train_set.append(data)

    X,y = X_test,y_test
    test_set = []
    for perm in permutations:
        data = X[:,perm], np_utils.to_categorical(y, nb_classes)
        test_set.append(data)
    return train_set,test_set


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


def construct_split_mnist(task_labels,  split='train', flatten=True,multihead=False):
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
    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = np.reshape(np.array(X_train),newshape=[-1,28,28,1])
        X_test = np.reshape(np.array(X_test),newshape=[-1,28,28,1])
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

def load_mnist(split='train',flatten=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if flatten:
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


def create_mixture(distributions,alpha_list = [0.2,0.8],sample_size=2000):
    alpha_list = np.array(alpha_list)
    alpha_list /= alpha_list.sum()
    num_distr = len(distributions)
    data = np.zeros((sample_size, num_distr))
    for idx, distr in enumerate(distributions):
        data[:, idx] = np.random.normal(size=(sample_size,), **distr["kwargs"])
    random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=alpha_list)
    sample = data[np.arange(sample_size), random_idx]
    sample = np.array(np.asmatrix(sample).T)
    return sample

"""
import numpy as np
import matplotlib.pyplot as plt

distributions = [
    {"type": np.random.normal, "kwargs": {"loc": -3, "scale": 2}},
    {"type": np.random.uniform, "kwargs": {"low": 4, "high": 6}},
    {"type": np.random.normal, "kwargs": {"loc": 2, "scale": 1}},
]
coefficients = np.array([0.5, 0.2, 0.3])
coefficients /= coefficients.sum()      # in case these did not add up to 1
sample_size = 100000

num_distr = len(distributions)
data = np.zeros((sample_size, num_distr))
for idx, distr in enumerate(distributions):
    data[:, idx] = distr["type"](size=(sample_size,), **distr["kwargs"])
random_idx = np.random.choice(np.arange(num_distr), size=(sample_size,), p=coefficients)
sample = data[np.arange(sample_size), random_idx]
plt.hist(sample, bins=100, density=True)
plt.show()
"""
















