import numpy as np
import scipy

from lazy_loader import attach as _attach
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

__all__ = (
    "ideal_data",
    "ReducedRankRegressor",
    "MultivariateRegressor",
    "ReducedRankRegressor",
)
__getattr__, __dir__, __all__ = _attach(f"{__name__}", submodules=__all__)
del _attach


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = np.random.randn(num, dimX)
    W = np.random.randn(dimX, rrank) @ np.random.randn(rrank, dimY)
    Y = X @ W + np.random.randn(num, dimY) * noise
    return X, Y

"""
Reduced rank regression class.
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Optimal linear 'bottlenecking' or 'multitask learning'.
"""
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

"""
Multivariate linear regression
Requires scipy to be installed.

Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com

Just simple linear regression with regularization - nothing new here
"""
class MultivariateRegressor(object):
    """
    Multivariate Linear Regressor.
    - X is an n-by-d matrix of features.
    - Y is an n-by-D matrix of targets.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, X, Y, reg=None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0

        W1 = np.linalg.pinv(np.dot(X.T, X) + reg * sparse.eye(np.size(X, 1)))
        W2 = np.dot(X, W1)
        self.W = np.dot(Y.T, W2)

    def __str__(self):
        return 'Multivariate Linear Regression'

    def predict(self, X):
        """Return the predicted Y for input X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.array(np.dot(X, self.W.T))
    
    def score(self, X, Y):
        """Return the coefficient of determination R^2 of the prediction."""
        y_pred = self.predict(X)
        return r2_score(Y, y_pred)

"""
kernel Reduced Rank Ridge Regression by Mukherjee
    DOI:10.1002/sam.10138

Code by Michele Svanera (2017-June)

"""
class ReducedRankRegressor(BaseEstimator):
    """
    kernel Reduced Rank Ridge Regression
    - X is an n-by-P matrix of features (n-time points).
    - Y is an n-by-Q matrix of targets (n-time points).
    - rank is a rank constraint.
    - reg is a regularization parameter.
    """

    def __init__(self, rank=10, reg=1, P_rr=None, Q_fr=None, trainX=None):
        self.rank = rank
        self.reg = reg
        self.P_rr = P_rr
        self.Q_fr = Q_fr
        self.trainX = trainX

    def __str__(self):
        return "kernel Reduced Rank Ridge Regression by Mukherjee (rank = {})".format(
            self.rank
        )

    def fit(self, X, Y):
        # use try/except blog with exceptions!
        self.rank = int(self.rank)

        K_X = scipy.dot(X, X.T)
        tmp_1 = self.reg * scipy.identity(K_X.shape[0]) + K_X
        Q_fr = np.linalg.solve(tmp_1, Y)
        P_fr = scipy.linalg.eig(scipy.dot(Y.T, scipy.dot(K_X, Q_fr)))[1].real
        P_rr = scipy.dot(P_fr[:, 0 : self.rank], P_fr[:, 0 : self.rank].T)

        self.Q_fr = Q_fr
        self.P_rr = P_rr
        self.trainX = X

    def predict(self, testX):
        # use try/except blog with exceptions!

        K_Xx = scipy.dot(testX, self.trainX.T)
        Yhat = scipy.dot(K_Xx, scipy.dot(self.Q_fr, self.P_rr))

        return Yhat

    def rrr_scorer(self, Yhat, Ytest):
        diag_corr = (np.diag(np.corrcoef(Ytest, Yhat))).mean()
        return diag_corr


## Optional
    def get_params(self, deep=True):
        return {"rank": self.rank, "reg": self.reg}
#
#    def set_params(self, **parameters):
#        for parameter, value in parameters.items():
#            self.setattr(parameter, value)
#        return self

    def mse(self, X, y_true):
        """
        Score the model on test data.
        """
        Yhat = self.predict(X).real
        MSE = (np.power((y_true - Yhat), 2) / np.prod(y_true.shape)).mean()
        return MSE

    def score(self, X, Y):
        """Score the model."""

        y_pred = self.predict(X)
        return r2_score(Y, y_pred)