"""
kernel Reduced Rank Ridge Regression by Mukherjee
    DOI:10.1002/sam.10138

Code by Michele Svanera (2017-June)

"""

import scipy
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = scipy.randn(num, dimX)
    W = scipy.randn(dimX, rrank) @ scipy.randn(rrank, dimY)
    Y = X @ W + scipy.randn(num, dimY) * noise
    return X, Y


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