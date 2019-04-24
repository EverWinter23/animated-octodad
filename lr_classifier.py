import numpy as np
from svm import *
from softmax import *

class LrClassifier:
    def __init__(self):
        self.W = None
    
    def train(self, X, Y, lr=1e-3, rg=1e-5, n_iters=100,
              batch_size=200, verbose=False):
        n_train, dim = X.shape
        n_classes = np.max(Y) + 1

        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, n_classes)

        # stochastic gradient descent to optimize W
        loss_hist = list()
        for _ in range(n_iters):
            idxs = np.arange(n_train)
            idxs = np.random.choice(idxs, batch_size, replace=True)
            X_batch = X[idxs]
            Y_batch = Y[idxs].reshape(batch_size, )

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, Y_batch, rg)
            loss_hist.append(loss)

            # update weights using the gradient and the learning rate
            self.W += lr * grad

            # every 100 lines output loss
            if verbose and _ % 100 == 0:
                print("iteration {}/{}: loss={}".format(_, n_iters, loss))
        
        return loss_hist

    def predict(self, X):
        Y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        Y_pred = np.argmax(scores, axis=1)
        
        return Y_pred
    
    def loss(self, X_batch, Y_batch, rg):
        pass

class LinearSVM(LrClassifier):
    def loss(self, X_batch, Y_batch, rg):
        return svm_loss(self.W, X_batch, Y_batch, rg)

class Softmax(LrClassifier):
    def loss(Self, X_batch, Y_batch, rg):
        return softmax_loss(self.W, X_batch, Y_batch, rg)

