import numpy as np

def svm_loss(W, X, Y, rg):
    loss = 0.0
    n_train = X.shape[0]
    dW = np.zeros(W.shape)
    
    s_c = X.dot(W)
    s_y = s_c[np.arange(n_train), Y].reshape(n_train, 1)
    margins = np.maximum(0, s_c - s_y + 1)
    margins[np.arange(n_train), Y] = 0
    loss = np.sum(np.sum(margins, axis=1)) / n_train

    # regularization
    loss += 0.5 * rg * np.sum(W * W)

    # calculating the gradient
    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
    non_zero_counts = np.sum(X_mask, axis=1)
    X_mask[np.arange(n_train), Y] -= non_zero_counts
    dW = X.T.dot(X_mask)

    # average out weights
    dW /= n_train
    dW += rg * W

    return loss, dW


