
from six.moves import cPickle as pickle
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
        fname = os.path.join(root, 'data_batch%d' % (_, ))
        X, Y = load_dataset_batch(fname)
        X_train.append(X)
        Y_train.append(Y)
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    del X, Y
    X_test, Y_test = load_dataset_batch(os.path.join(root, 'test_batch'))
    return X_train, Y_train, X_test, Y_test
