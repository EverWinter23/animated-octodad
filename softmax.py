import numpy as np

def softmax_loss(W, X, Y, rg):
    loss = 0.0
    n_train = X.shape[0]
    n_class = W.shape[1]
    dW = np.zeros(W.shape)
    
    f = X.dot(W)
    f -= np.max(f, axis=1, keepdims=True)
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f) / sum_f

    loss = np.sum(-np.log(p[np.arange(n_train), y]))

    idx = np.zeros(p.shape)
    idx[np.arange(n_train), y] = 1
    dW = X.T.dot(p - idx)

    loss /= n_train
    loss += 0.5 * rg * np.sum(W * W)
    
    dW /= n_train
    dW += rg * W

    return loss, dW