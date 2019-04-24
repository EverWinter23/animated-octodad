import numpy as np

class KNN:
    def __init__(self):
        pass
    
    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X, k=1):
        dists = self.compute_l2_dists(X)
        return self.predict_labels(dists, k=k)

    def compute_l2_dists(self, X):
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]

        dists = np.zeros((n_test, n_train))
        dists = np.sqrt(-2 * np.dot(X, self.X_train.T)
                        + np.sum(X**2, axis=1, keepdims=True)
                        + np.sum(self.X_train**2, axis=1))
        
        return dists

    def predict_labels(self,  dists, k=1):
        n_test = dists.shape[0]
        Y_pred = np.zeros(n_test)

        for i in range(n_test):
            k_nearest_idxs = np.argsort(dists[i])[:k]
            closest_y = self.Y_train[k_nearest_idxs]
            Y_pred[i] = np.argmax(np.bincount(closest_y))
        
        return Y_pred
        

    
