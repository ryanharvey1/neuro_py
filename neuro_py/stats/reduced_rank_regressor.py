"""
Reduced rank regression class.
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Optimal linear 'bottlenecking' or 'multitask learning'.
"""
import numpy as np
from scipy import sparse
from sklearn.metrics import r2_score


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = np.random.randn(num, dimX)
    W = np.random.randn(dimX, rrank) @ np.random.randn(rrank, dimY)
    Y = X @ W + np.random.randn(num, dimY) * noise
    return X, Y


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """

    def __init__(self, X, Y, rank, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0
        self.rank = rank

        CXX = X.T @ X + reg * sparse.eye(np.size(X, 1))
        CXY = X.T @ Y
        _U, _S, V = np.linalg.svd(CXY.T @ (np.linalg.pinv(CXX) @ CXY))
        self.W = V[0:rank, :].T
        self.A = (np.linalg.pinv(CXX) @ (CXY @ self.W)).T

    def __str__(self):
        return "Reduced Rank Regressor (rank = {})".format(self.rank)

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return X @ (self.A.T @ self.W.T)

    def score(self, X, Y):
        """Score the model."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))

        y_pred = self.predict(X)
        return r2_score(Y, y_pred)
