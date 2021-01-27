import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from LRCT.LRCT import LRCTree
from LRCT.Exceptions import NotFitError

class MultiClassLRCT(BaseEstimator, ClassifierMixin):
    """Multi-Class Linear Regression Classification Tree

    LRCT which performes one-versus-rest classification
    """

    def __init__(
            self,
            max_depth = None,
            min_samples_split = 2,
            min_samples_leaf = 1,
            n_independent = 1,
            highest_degree = 1,
            fit_intercepts = True,
            method = 'ols',
            n_bins = 10,
            **kwargs
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_independent = n_independent
        self.highest_degree = highest_degree
        self.fit_intercepts = fit_intercepts
        self.method = method
        self.n_bins = n_bins
        self.kwargs = kwargs

        self.trees = []
        self._is_fit = False

    def fit(self, x, y):
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise TypeError('x must be numpy array or DataFrame')
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError('y must be ndarray or Series')
        if len(y.shape) != 1:
            raise ValueError('y must have a single dimension')

        if x.shape[0] != y.shape[0]:
            raise ValueError('Number of records does not match number of samples')

        if isinstance(y, pd.Series):
            y = y.values

        sorted_to_predict = np.sort(np.unique(y))
        for val in sorted_to_predict:
            altered_y = (y == val).astype(int)
            self.trees.append(
                LRCTree(self.max_depth,
                        self.min_samples_split,
                        self.min_samples_leaf,
                        self.n_independent,
                        self.highest_degree,
                        self.fit_intercepts,
                        self.method,
                        self.n_bins,
                ).fit(x, altered_y)
            )
        self._is_fit = True
        return self

    def _predict_single_instance(self, instance, proba = False):
        probs = np.array([t._predict_single_instance(instance, proba = True)[1] for t in self.trees])
        probs = probs/probs.sum()
        if proba:
            return probs
        else:
            if (probs == probs.max()).sum() == 1:
                return probs.argmax()
            else:
                return np.random.choice([i for i in range(probs.shape[0]) if probs[i] == probs.max()])

    def predict(self, x):
        if not self._is_fit:
            raise NotFitError
        
        return np.apply_along_axis(lambda row : self._predict_single_instance(row, proba = False), 1, x)

    def predict_proba(self, x):
        if not self._is_fit:
            raise NotFitError

        return np.apply_along_axis(lambda row : self._predict_single_instance(row, proba = True), 1, x)

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)
