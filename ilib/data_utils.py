
from six.moves import cPickle as pickle
import os
import numpy as np

def load_pickle(f):
    return load_pickle.load(f, encoding='bytes')

def load_dataset_batch(fname):
    with open(fname, 'rb') as f:
        data = load_pickle(f)
        X = data['data']
        Y = data['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y 

def load_dataset(root):
    X_train = []
    Y_train = []
    for _ in range(1, 6):
        fname = os.path.join(root, 'data_batch_%d' % (_, ))
        X, Y = load_dataset_batch(fname)
        X_train.append(X)
        Y_train.append(Y)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    del X, Y
    X_test, Y_test = load_dataset_batch(os.path.join(root, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

def get_dataset(n_train=49000, n_valid=1000, n_test=1000, sub_mean=True):
    root_dir = '/content/cifar-10-batches-py'
    X_train, Y_train, X_test, Y_test = load_dataset(root_dir)

    # validation set
    mask = list(range(n_train, n_train + n_valid))
    X_valid = X_train[mask]
    Y_vaild = Y_train[mask]

    # training set
    mask = list(range(n_train))
    X_train = X_train[mask]
    Y_train = Y_train[mask]

    # test set
    mask = list(range(n_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]

    # normalize the data
    if sub_mean:
        mean_img = np.mean(X_train, axis=0)
        X_train -= mean_img
        X_valid -= mean_img
        X_test -= mean_img

    # transpose the data-> 
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_valid = X_valid.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # package data into a dictionary
    return {
      'X_train': X_train, 'y_train': Y_train,
      'X_valid': X_valid, 'Y_valid': Y_valid,
      'X_test': X_test, 'y_test': y_test,
    }
