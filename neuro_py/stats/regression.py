from typing import Optional, Tuple

import numpy as np
import scipy
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score


def ideal_data(
    num: int, dimX: int, dimY: int, rrank: int, noise: float = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate low-rank data.

    Parameters
    ----------
    num : int
        Number of samples.
    dimX : int
        Dimensionality of the input data.
    dimY : int
        Dimensionality of the output data.
    rrank : int
        Rank of the low-rank structure.
    noise : float, optional
        Standard deviation of the noise added to the output data (default is 1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - X : np.ndarray
            The generated input data of shape (num, dimX).
        - Y : np.ndarray
            The generated output data of shape (num, dimY).

    """
    X = np.random.randn(num, dimX)
    W = np.random.randn(dimX, rrank) @ np.random.randn(rrank, dimY)
    Y = X @ W + np.random.randn(num, dimY) * noise
    return X, Y


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning').

    Parameters
    ----------
    X : np.ndarray
        An n-by-d matrix of features.
    Y : np.ndarray
        An n-by-D matrix of targets.
    rank : int
        A rank constraint.
    reg : Optional[float], optional
        A regularization parameter (default is None).

    References
    ----
    Implemented by Chris Rayner (2015).
    dchrisrayner AT gmail DOT com
    """

    def __init__(
        self, X: np.ndarray, Y: np.ndarray, rank: int, reg: Optional[float] = None
    ):
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

    def __str__(self) -> str:
        return "Reduced Rank Regressor (rank = {})".format(self.rank)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return X @ (self.A.T @ self.W.T)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Score the model."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))

        y_pred = self.predict(X)
        return r2_score(Y, y_pred)


class MultivariateRegressor(object):
    """
    Multivariate Linear Regressor.

    Parameters
    ----------
    X : np.ndarray
        An n-by-d matrix of features.
    Y : np.ndarray
        An n-by-D matrix of targets.
    reg : Optional[float], optional
        A regularization parameter (default is None).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, reg: Optional[float] = None):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if reg is None:
            reg = 0

        W1 = np.linalg.pinv(np.dot(X.T, X) + reg * sparse.eye(np.size(X, 1)))
        W2 = np.dot(X, W1)
        self.W = np.dot(Y.T, W2)

    def __str__(self) -> str:
        return "Multivariate Linear Regression"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted Y for input X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.array(np.dot(X, self.W.T))

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        y_pred = self.predict(X)
        return r2_score(Y, y_pred)


class kernelReducedRankRegressor(BaseEstimator):
    """
    Kernel Reduced Rank Ridge Regression.

    Parameters
    ----------
    rank : int, optional
        The rank constraint (default is 10).
    reg : float, optional
        The regularization parameter (default is 1).
    P_rr : Optional[np.ndarray], optional
        The P matrix for reduced rank (default is None).
    Q_fr : Optional[np.ndarray], optional
        The Q matrix for fitted values (default is None).
    trainX : Optional[np.ndarray], optional
        The training features (default is None).

    References
    ----------
    Mukherjee, S. (DOI:10.1002/sam.10138)
    Code by Michele Svanera (2017-June).
    """

    def __init__(
        self,
        rank: int = 10,
        reg: float = 1,
        P_rr: Optional[np.ndarray] = None,
        Q_fr: Optional[np.ndarray] = None,
        trainX: Optional[np.ndarray] = None,
    ):
        self.rank = rank
        self.reg = reg
        self.P_rr = P_rr
        self.Q_fr = Q_fr
        self.trainX = trainX

    def __str__(self) -> str:
        return "kernel Reduced Rank Ridge Regression by Mukherjee (rank = {})".format(
            self.rank
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
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

    def predict(self, testX: np.ndarray) -> np.ndarray:
        # use try/except blog with exceptions!

        K_Xx = scipy.dot(testX, self.trainX.T)
        Yhat = scipy.dot(K_Xx, scipy.dot(self.Q_fr, self.P_rr))

        return Yhat

    def rrr_scorer(self, Yhat: np.ndarray, Ytest: np.ndarray) -> float:
        diag_corr = (np.diag(np.corrcoef(Ytest, Yhat))).mean()
        return diag_corr

    ## Optional
    def get_params(self, deep: bool = True) -> dict:
        return {"rank": self.rank, "reg": self.reg}

    #
    #    def set_params(self, **parameters):
    #        for parameter, value in parameters.items():
    #            self.setattr(parameter, value)
    #        return self

    def mse(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Score the model on test data.

        Parameters
        ----------
        X : np.ndarray
            The test data features.
        y_true : np.ndarray
            The true target values.

        Returns
        -------
        float
            The mean squared error of the predictions.
        """
        Yhat = self.predict(X).real
        MSE = (np.power((y_true - Yhat), 2) / np.prod(y_true.shape)).mean()
        return MSE

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Score the model."""

        y_pred = self.predict(X)
        return r2_score(Y, y_pred)
