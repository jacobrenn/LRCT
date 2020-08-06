import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from LRCT.Exceptions import NotFitError
from LRCT.Node import Node
from LRCT.splitting_functions import find_best_lrct_split, find_best_split


class LRCTree(BaseEstimator, ClassifierMixin):
    """Linear Regression Classification Tree

    LRCT serves as an improved classification tree capable of making multivariate
    linear and nonlinear splits in its training.  It does this by approximating the
    optimal surface function across multiple variables by applying binning and linear
    regression.
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
            The maximum depth the Tree can train to. If None, does not use
            depth as a stopping criteria
        min_samples_split : int (default 2)
            The minimum number of samples that have to be at a Node to make
            a split
        min_samples_leaf : int (default 1)
            The minimum number of samples that have to be at a leaf Node
        n_independent : int (default 1)
            The number of independent variables to use per multivariate split.
            This value can be set to 0, which will result in CART
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

        self._nodes = {}

    @property
    def depth(self):
        if self._nodes == {}:
            return 0
        else:
            return max([n.depth for n in self.nodes])

    @property
    def max_depth(self):
        return self._max_depth if self._max_depth else np.inf

    @max_depth.setter
    def max_depth(self, value):
        self._max_depth = value

    @property
    def min_samples_split(self):
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, value):
        self._min_samples_split = value

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

    @min_samples_leaf.setter
    def min_samples_leaf(self, value):
        self._min_samples_leaf = value

    @property
    def n_independent(self):
        return self._n_independent

    @n_independent.setter
    def n_independent(self, value):
        self._n_independent = value

    @property
    def highest_degree(self):
        return self._highest_degree

    @highest_degree.setter
    def highest_degree(self, value):
        self._highest_degree = value

    @property
    def fit_intercepts(self):
        return self._fit_intercepts

    @fit_intercepts.setter
    def fit_intercepts(self, value):
        self._fit_intercepts = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def n_bins(self):
        return self._n_bins

    @n_bins.setter
    def n_bins(self, value):
        self._n_bins = value

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value):
        self._kwargs = value

    @property
    def nodes(self):
        try:
            return [n for n in self._nodes.values()]
        except:
            raise NotFitError()

    def _add_nodes(self, nodes):
        """Add Nodes to the Tree

        Parameters
        ----------
        Nodes : Node or list of Nodes
            Nodes to add to the Tree

        Notes
        -----
        - Raises ValueError if Node is passed with identifier that already exists in Tree
        """
        is_node = isinstance(nodes, Node)
        acceptable_list = isinstance(nodes, list) and all([isinstance(n, Node) for n in nodes])

        if not (is_node or acceptable_list):
            raise ValueError('adding nodes requires Node objects or list of Node objects')

        # figure out the existing IDs
        existing_ids = [id for id in self._nodes.keys()]

        # value checking for existing IDs
        if is_node and nodes.identifier in existing_ids:
            raise ValueError('Node with that ID already exists')
        if acceptable_list and any([n.identifier in existing_ids for n in nodes]):
            raise ValueError('Trying to set Node with existing ID')

        # do the actual adding of Node(s)
        if is_node:
            self._nodes[nodes.identifier] = nodes
        elif acceptable_list:
            for node in nodes:
                self._nodes[node.identifier] = node

    def describe(self):
        """Print a description of the Tree"""
        if self._nodes == {}:
            print('Empty Tree')
        else:
            print('\n'.join([f'{"-" * n.depth}{n}' for n in self._nodes.values()]))

    def _split_node(self, node_id, x_data, y_data):
        """Split a Node as in training

        Parameters
        ----------
        node_id : int
            The identifier of the Node to split
        x_data : 2d array-like
            The data to train from
        y_data : array-like
            The target values

        Notes
        -----
        - If a split is found, alters the Tree by adding Nodes and altering the split of the Node
        being split

        Returns
        -------
        split_info : tuple
            Tuple of the form (less_node_id, greater_node_id, less_x, greater_x, less_y, greater_y)
        None : NoneType
            If no split is found
        """
        # if x_data does not have more than min_samples_split rows or y_data has only one unique value, do nothing
        if x_data.shape[0] < self.min_samples_split or np.unique(y_data).shape[0] == 1:
            return None

        # make a copy of X and work with that
        x_copy = np.array(x_data).copy()

        # get the node, the prospective parent id, and the prospective parent depth
        node = self._nodes[node_id]
        parent_id = node.identifier
        parent_depth = node.depth

        # do not continue if already at max depth
        if parent_depth == self.max_depth:
            return None

        highest_id = max(self._nodes.keys())

        # find the split information
        best_traditional_split = find_best_split(x_copy, y_data)
        best_lrct_split = find_best_lrct_split(
            x_copy,
            y_data,
            self.n_independent,
            self.highest_degree,
            self.fit_intercepts,
            self.method,
            self.n_bins,
            **self.kwargs
        )

        if best_traditional_split['split_gini'] <= best_lrct_split['split_gini']:
            split = best_traditional_split
        else:
            split = best_lrct_split

        if split.get('col_idx'):
            less_idx = x_copy[:, split.get('col_idx')] <= split.get('split_value')
            greater_idx = x_copy[:, split.get('col_idx')] > split.get('split_value')
        else:
            ind = split['indices']
            new_col = np.zeros(x_copy.shape[0])
            for i in range(split['coefs'].shape[0]):
                column_idx = ind[i % (len(ind) - 1)]
                power = (i // (len(ind) - 1)) + 1
                new_col += split['coefs'][i]*x_copy[:, column_idx]**power
            new_col -= x_copy[:, ind[-1]]
            less_idx = new_col <= split.get('split_value')
            greater_idx = new_col > split.get('split_value')

        if (less_idx.sum() < self.min_samples_leaf) or (greater_idx.sum() < self.min_samples_leaf):
            return None

        less_node = Node(
            highest_id + 1,
            parent_id,
            parent_depth + 1
        )
        greater_node = Node(
            highest_id + 2,
            parent_id,
            parent_depth + 1
        )
            
        self._add_nodes([less_node, greater_node])
        self._nodes[parent_id].split = split
        return highest_id + 1, highest_id + 2, x_copy[less_idx], x_copy[greater_idx], y_data[less_idx], y_data[greater_idx]

    def fit(self, x, y):
        """Fit the Tree

        Parameters
        ----------
        x : 1d array-like
            The independent data to learn from
        y : 1d numpy array or pandas Series
            The target to learn

        Returns
        -------
        tree : LRCTree
            The fit tree (self)
        """

        # typechecking
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            raise TypeError('x must be ndarray or pandas DataFrame')
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError('y must be ndarray or Series')
        if len(y.shape) != 1:
            raise ValueError('y must have a single dimension')

        # check the shapes of each variable
        if x.shape[0] != y.shape[0]:
            raise ValueError('Number of records does not match number of samples')

        # if y is pandas Series, use only values (numpy array)
        if isinstance(y, pd.Series):
            y = y.values

        # instantiate the Nodes
        self._nodes = {}

        # add the parent Node
        self._add_nodes(Node())

        # keep a record of all Node data for record keeping -- will be lost at the end however
        node_data = {
            0: {
                'x': x,
                'y': y
            }
        }

        # fitting logic is inside while loop
        while self.depth < self.max_depth:

            # record current depth Nodes and how many Nodes exist to determine stopping criteria later
            current_depth_nodes = [n for n in self.nodes if n.depth == self.depth]
            num_nodes = len(self.nodes)

            # try to split all of the current depth Nodes
            for n in current_depth_nodes:
                split_results = self._split_node(n.identifier, node_data[n.identifier]['x'],
                                                 node_data[n.identifier]['y'])

                # get the split information if there is an actual split to get
                # remember -- if split_results is not None, the Node was actually split and new Nodes were added to the
                # Tree
                if split_results is not None:
                    # get the data for each side
                    less_id = split_results[0]
                    greater_id = split_results[1]
                    x_less = split_results[2]
                    x_greater = split_results[3]
                    y_less = split_results[4]
                    y_greater = split_results[5]

                    # create the Nodes
                    node_data[less_id] = {
                        'x': x_less,
                        'y': y_less
                    }
                    node_data[greater_id] = {
                        'x': x_greater,
                        'y': y_greater
                    }

            # if no new Nodes, then we are done
            new_num_nodes = len(self.nodes)
            if new_num_nodes == num_nodes:
                break

        # set _is_fit to True
        self._is_fit = True

        # _values_to_predict helps with predicting probabilities
        self.values_to_predict_ = np.unique(y)

        # _node_distributions helps with predicting
        self.node_distributions_ = {
            n.identifier: np.array([(node_data[n.identifier]['y'] == i).sum() for i in self.values_to_predict_])
            for n in self.nodes if n.split is np.nan
        }

        # return self for consistency with scikit-learn
        return self

    def _predict_single_instance(self, instance, proba = False):
        """Predict a single new instance

        Parameters
        ----------
        instance : 1d numpy array
            Object which has the desired implementation of the .keys() method
        proba : bool (default False)
            Whether to return probabilities

        Returns
        -------
        prediction : int or float
            The class predicted or probabilities predicted
        """

        # start at the root Node
        current_node = self._nodes[0]

        while isinstance(current_node.split, dict):
            child_node_ids = [n.identifier for n in self.nodes if n.parent_id == current_node.identifier]
            if current_node.split.get('col_idx'):
                if instance[current_node.split['col_idx']] <= current_node.split['split_value']:
                    new_node_id = min(child_node_ids)
                else:
                    new_node_id = max(child_node_ids)
            else:
                ind = current_node.split['indices']
                new_val = 0
                for i in range(current_node.split['coefs'].shape[0]):
                    column_index = ind[i % (len(ind) - 1)]
                    power = (i // (len(ind) - 1)) + 1
                    new_val += current_node.split['coefs'][i]*instance[column_index]**power
                new_val -= instance[ind[-1]]
                if new_val <= current_node.split['split_value']:
                    new_node_id = min(child_node_ids)
                else:
                    new_node_id = max(child_node_ids)
            
            current_node = self._nodes[new_node_id]
        
        # after we're out of the while loop, get the current distribution to see what we need to predict
        current_distribution = self.node_distributions_[current_node.identifier]

        # get the probabilities
        probabilities = current_distribution / current_distribution.sum()
        
        # return probabilities if asked for
        if proba:
            return probabilities
        else:
            # check if there's a clear winner
            if (probabilities == probabilities.max()).sum() == 1:
                return probabilities.argmax()
            
            # if there's no clear winner, return a random guess from the most common ones
            highest_indices = [i for i in range(probabilities.shape[0]) if probabilities[i] == probabilities.max()]
            return np.random.choice(highest_indices)
            
    def _predict_single_proba(self, instance):
        """Predict class probabilities for a single instance

        Parameters
        ----------
        instance : pd.Series or dict
            Object which has the desired implementation of the .keys() method

        Returns
        -------
        probas : numpy array
            Array of shape (n_classes,), where n_classes is the number of classes trained on
        """

        # implementation of this function is similar to the implementation of _predict_single_instance
        # except return the node distributions normalized to sum to 1
        current_node = self._nodes[0]

        while isinstance(current_node.split, tuple):
            child_node_ids = [n.identifier for n in self.nodes if n.parent_id == current_node.identifier]
            split_col, split_value = current_node.split
            if split_col not in instance.keys():
                rest, last_col = split_col.split(' - ')[0], split_col.split(' - ')[1]
                new_coefs = [item.split('*')[0] for item in rest.split(' + ')]
                new_cols = [item.split('*')[1].split('^')[0] for item in rest.split(' + ')]
                new_col_components = []
                for i in range(len(new_coefs)):
                    if '^' in new_cols[i]:
                        col, exp = new_cols[i].split('^')[0], new_cols[i].split('^')[1]
                        new_col_components.append(f'{new_coefs[i]}*instance["{col}"]**{exp}')
                    else:
                        new_col_components.append(f'{new_coefs[i]}*instance["{new_cols[i]}"]')
                new_col_str = ' + '.join(new_col_components)
                new_col_str += f' - instance["{last_col}"]'
                val_to_check = eval(new_col_str)
            else:
                val_to_check = instance[split_col]

            if val_to_check <= split_value:
                new_node_id = min(child_node_ids)
            else:
                new_node_id = max(child_node_ids)
            current_node = self._nodes[new_node_id]

        return self.node_distributions_[current_node.identifier] / self.node_distributions_[
            current_node.identifier].sum()

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

        # this function just applies the _predict_single_instance method to each of the rows
        return np.apply_along_axis(lambda row : self._predict_single_instance(row, proba = False), 1, x)

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

        # apply predicting with probabilities
        return np.apply_along_axis(lambda row : self._predict_single_instance(row, proba = True), 1, x)

    def score(self, x, y):
        """Score the model's performance on new labeled data using accuracy score as a measure

        Parameters
        ----------
        x : pandas DataFrame
            DataFrame to predict from
        y : pandas Series or 1d numpy array
            Labels for data

        Returns
        -------
        score : float
            Accuracy score of the model on its predictions of x
        """
        preds = self.predict(x)
        return accuracy_score(y, preds)

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
