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
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_independent=1,
            highest_degree=1,
            fit_intercepts=True,
            method='ols',
            n_bins=10,
            **kwargs
    ):
        """
        Parameters
        ----------
        max_depth : int or None (default None)
            The maximum depth the Tree ecan train to. If None, does not use
            depth as a stopping criteria
        min_samples_split : int (default 2)
            The minimum number of samples that have to be at a Node to make
            a split
        min_samples_leaf : int (default 1)
            The minimum number of samples that have to be at a leaf Node
        n_independent : int (default 1)
            The number of independent variables to us per multivariate split.
            This number can be set to 0, which will result in CART
        highest_degree : int (default 1)
            The highest degree to which to raise the independent variables
            in surface function learning
        fit_intercepts : bool (default True)
            Whether to fit intercepts in linear regression models
        method : str (default 'ols')
            One of 'ols', 'ridge', or 'lasso', the type of regression model to
            fit
        n_bins : int (default 10)
            The number of bins to use per independent variable in training
        **kwargs : additional keyword arguments
            Additional keyword arguments to pass to the linear regression models
        """

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
        """Fit the Tree

        Parameters
        ----------
        x : 2d array-like
           The independent data to learn from
        y : 1d numpy array or pandas Series
           The target to learn

        Returns
        -------
        tree : MultiClassLRCT
           The fit tree (self)
        """

        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise TypeError('x must be numpy array or DataFrame')
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError('y must be ndarray or Series')
        if len(y.shape) != 1:
            raise ValueError('y must have a single dimension')

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                'Number of records does not match number of samples')

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

    def _predict_single_instance(self, instance, proba=False):
        probs = np.array([t._predict_single_instance(
            instance, proba=True)[1] for t in self.trees])
        probs = probs/probs.sum()
        if proba:
            return probs
        else:
            if (probs == probs.max()).sum() == 1:
                return probs.argmax()
            else:
                return np.random.choice([i for i in range(probs.shape[0]) if probs[i] == probs.max()])

    def predict(self, x):
        """Predict classes for a set of values

        Parameters
        ----------
        x : pandas DataFrame
            Data to predict from

        Returns
        -------
        preds : numpy array
            Numpy array of predictions
        """
        if not self._is_fit:
            raise NotFitError

        return np.apply_along_axis(lambda row: self._predict_single_instance(row, proba=False), 1, x)

    def predict_proba(self, x):
        """Predict class probabilities for a set of values

        Parameters
        ----------
        x : 2d array-like
            Data to predict from

        Returns
        -------
        preds : numpy array
            Numpy array of predicted probabilities. Column indices correspond to same classes
        """
        if not self._is_fit:
            raise NotFitError

        return np.apply_along_axis(lambda row: self._predict_single_instance(row, proba=True), 1, x)

    def fit_predict(self, x, y):
        """Fit the model and predict on X

        Parameters
        ----------
        x : pandas DataFrame
            DataFrame to train on and predict from
        y : pandas Series or 1d numpy array
            Labels for x to learn from

        Returns
        -------
        preds : 1d numpy array
            Predictions on x after fitting
        """
        self.fit(x, y)
        return self.predict(x)
