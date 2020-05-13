#!/usr/bin/env python

# Alternating Direction Method of Multipliers for the Lasso regression

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ADMM(BaseEstimator, RegressorMixin):
    """Alternating Direction Method of Multipliers for the Lasso regression."""

    def __init__(self, lambda_=1.0, rho=1.0, max_iter=1000):
        """Initialize the instance with the following parameters:

        lambda_: Lagrange multiplier for the model
        rho: small constant parameter used throughout the model
        threshold: threshold for soft-thresholding function
        max_iter: max num of iteration to train the model
        coefs: weight vector acquired after training the model
        """
        self.lambda_ = lambda_
        self.rho = rho
        self.threshold = lambda_ / rho
        self.max_iter = max_iter
        self.coefs = None

    def __str__(self):
        return "ADMM"

    def __repr__(self):
        return "ADMM"

    def __soft_threshold(self, beta):
        t = self.threshold
        indexes_pos = beta >= t
        indexes_neg = beta <= t
        indexes_zero = abs(beta) <= t
        mu = np.zeros(beta.shape)
        mu[indexes_pos] = beta[indexes_pos] - t
        mu[indexes_neg] = beta[indexes_neg] + t
        mu[indexes_zero] = 0.0
        return mu

    def __inverse(self, X, p):
        return np.linalg.inv(np.dot(X.T, X) + self.rho * np.identity(p))

    def fit(self, X, y):
        N = X.shape[0]  # num of samples in the dataset
        p = X.shape[1]  # num of dimensions of each feature
        matrix_inved = self.__inverse(X, p)
        beta = np.dot(X.T, y) / N
        theta = beta.copy()
        mu = np.zeros(len(beta))
        for _ in range(self.max_iter):
            beta = np.dot(matrix_inved,
                          np.dot(X.T, y) + self.rho * theta - mu)
            theta = self.__soft_threshold(beta + mu / self.rho)
            mu += self.rho * (beta - theta)
        self.coefs = beta
        return self

    def predict(self, X):
        y = np.dot(X, self.coefs)
        return y
