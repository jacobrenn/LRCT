from LRCT.splitting_functions import find_best_split, find_best_lrct_split
from LRCT.Node import Node
from LRCT.Exceptions import NotFitError

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

import warnings

class LRCTree(BaseEstimator, ClassifierMixin):
    '''Linear Regression Classification Tree

    LRCT serves as an improved classification tree capable of making multivariate
    linear and nonlinear splits in its training.  It does this by approximating the
    optimal surface function across multiple variables by applying binning and linear
    regression.
    '''
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
        '''
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
        '''
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
        nn = value is not None
        if not isinstance(value, int) and nn:
            raise TypeError('max_depth must be int')
        if nn and value <= 0:
            raise ValueError('max_depth must be greater than 0')
        self._max_depth = value
    
    @property
    def min_samples_split(self):
        return self._min_samples_split
    @min_samples_split.setter
    def min_samples_split(self, value):
        if not isinstance(value, int):
            raise TypeError('min_samples_split must be int')
        if value < 2:
            raise ValueError('Minimum value for min_samples_split must be 2')
        self._min_samples_split = value

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf
    @min_samples_leaf.setter
    def min_samples_leaf(self, value):
        if not isinstance(value, int):
            raise TypeError('min_samples_leaf must be int')
        if value <= 0:
            raise ValueError('min_samples_leaf must be greater than 0')
        self._min_samples_leaf = value
    
    @property
    def n_independent(self):
        return self._n_independent
    @n_independent.setter
    def n_independent(self, value):
        if not isinstance(value, int):
            raise TypeError('n_independent must be int')
        if value < 0:
            raise ValueError('n_independent must be nonnegative')
        if value == 0:
            warnings.warn('Setting n_independent to 0 will result in CART tree')
        self._n_independent = value

    @property
    def highest_degree(self):
        return self._highest_degree
    @highest_degree.setter
    def highest_degree(self, value):
        if not isinstance(value, int):
            raise TypeError('LRCT currently supports integer-valued highest degrees')
        self._highest_degree = value

    @property
    def fit_intercepts(self):
        return self._fit_intercepts
    @fit_intercepts.setter
    def fit_intercepts(self, value):
        if not isinstance(value, bool):
            raise TypeError('fit_intercepts must be boolean')
        self._fit_intercepts = value

    @property
    def method(self):
        return self._method
    @method.setter
    def method(self, value):
        allowable_methods = ['ols', 'ridge', 'lasso']
        if value not in allowable_methods:
            raise ValueError(f'method must be one of {allowable_methods}')
        self._method = value

    @property
    def n_bins(self):
        return self._n_bins
    @n_bins.setter
    def n_bins(self, value):
        if not isinstance(value, int):
            raise TypeError('n_bins must be int')
        if value <= 1:
            raise ValueError('n_bins must be greater than or equal to 0')
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
        '''Add Nodes to the Tree

        Parameters
        ----------
        Nodes : Node or list of Nodes
            Nodes to add to the Tree

        Notes
        -----
        - Raises ValueError if Node is passed with identifier that already exists in Tree
        '''
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
        '''Print a description of the Tree'''
        if self._nodes == {}:
            print('Empty Tree')
        else:
            print('\n'.join([f'{"-"*n.depth}{n}' for n in self._nodes.values()]))

    def _split_node(self, node_id, x_data, y_data):
        '''Split a Node as in training
        
        Parameters
        ----------
        node_id : int
            The identifier of the Node to split
        x_data : pandas DataFrame
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
        '''
        #if x_data does not have more than min_samples_split rows or y_data has only one unique value, do nothing
        if x_data.shape[0] < self.min_samples_split or np.unique(y_data).shape[0] == 1:
            return None

        #make a copy of X and work with that 
        x_copy = x_data.copy()

        #get the node, the prospective parent id, and the prospective parent depth
        node = self._nodes[node_id]
        parent_id = node.identifier
        parent_depth = node.depth
        
        #do not continue if already at max depth
        if parent_depth == self.max_depth:
            return None
        
        highest_id = max(self._nodes.keys())

        #find the split information
        split_info = find_best_lrct_split(
            x_copy,
            y_data,
            self.n_independent,
            self.highest_degree,
            self.fit_intercepts,
            self.method,
            self.n_bins,
            **self.kwargs
        )

        if split_info is np.nan:
            return None
        else:
            split_col, split_value = split_info

        #if split_col is not one of the original columns, must be LRCT column -- parse and create
        if split_col not in x_copy.columns:
            rest, last_col = split_col.split(' - ')[0], split_col.split(' - ')[1]
            new_coefs = [item.split('*')[0] for item in rest.split(' + ')]
            new_cols = [item.split('*')[1].split('^')[0] for item in rest.split(' + ')]
            new_col_components = []
            for i in range(len(new_coefs)):
                if '^' in new_cols[i]:
                    col, exp = new_cols[i].split('^')[0], new_cols[i].split('^')[1]
                    new_col_components.append(f'{new_coefs[i]}*x_copy["{col}"]**{exp}')
                else:
                    new_col_components.append(f'{new_coefs[i]}*x_copy["{new_cols[i]}"]')
            new_col_str = ' + '.join(new_col_components)
            new_col_str += f' - x_copy["{last_col}"]'
            new_col = eval(new_col_str)
            split_col_values = new_col
        else:
            split_col_values = x_copy[split_col]

        #create indices for both sides of the split
        less_idx = split_col_values <= split_value
        greater_idx = split_col_values > split_value

        #create the new Nodes
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
        
        #check for stopping conditions
        if (less_idx.sum() < self.min_samples_leaf) or (greater_idx.sum() < self.min_samples_leaf):
            return None
        
        #if we've gotten here, we're good to go -- add the Nodes and return pertinent info
        self._add_nodes([less_node, greater_node])
        self._nodes[parent_id].split = split_info
        return highest_id + 1, highest_id + 2, x_copy[less_idx], x_copy[greater_idx], y_data[less_idx], y_data[greater_idx]

    def fit(self, x, y):
        '''Fit the Tree

        Parameters
        ----------
        x : pandas DataFrame
            The independent data to learn from
        y : 1d numpy array or pandas Series
            The target to learn

        Returns
        -------
        tree : LRCTree
            The fit tree (self)
        '''
        
        # typechecking
        if not isinstance(x, pd.DataFrame):
            raise TypeError('x must be pandas DataFrame')
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
                0 : {
                    'x' : x,
                    'y' : y
                }
            }

        # fitting logic is inside while loop
        while self.depth < self.max_depth:

            # record current depth Nodes and how many Nodes exist to determine stopping criteria later
            current_depth_nodes = [n for n in self.nodes if n.depth == self.depth]
            num_nodes = len(self.nodes)

            # try to split all of the current depth Nodes
            for n in current_depth_nodes:
                split_results = self._split_node(n.identifier, node_data[n.identifier]['x'], node_data[n.identifier]['y'])

                # get the split information if there is an actual split to get
                # remember -- if split_results is not None, the Node was actually split and new Nodes were added to the Tree
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
                        'x' : x_less,
                        'y' : y_less
                    }
                    node_data[greater_id] = {
                        'x' : x_greater,
                        'y' : y_greater
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
            n.identifier : np.array([(node_data[n.identifier]['y'] == i).sum() for i in self.values_to_predict_])
            for n in self.nodes if n.split is np.nan
        }

        # return self for consistency with scikit-learn
        return self

    def _predict_single_instance(self, instance):
        '''Predict a single new instance

        Parameters
        ----------
        instance : pd.Series or dict
            Object which has the desired implementation of the .keys() method

        Returns
        -------
        prediction : int
            The class predicted
        '''

        # start at the root Node
        current_node = self._nodes[0]

        # continue until at a leaf Node (split is np.nan)
        while not (current_node.split is np.nan):

            # figure out the child IDs
            child_node_ids = [n.identifier for n in self.nodes if n.parent_id == current_node.identifier]
            # get the split column and the split value
            split_col, split_value = current_node.split

            # if the column is not in the default keys, it must be a multivariate split
            # if so, construct the column that you want
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

            # the else case is easy
            else:
                val_to_check = instance[split_col]

            # we always know that the Node corresponding to split_col < split_val is made first
            # hence, it has the lower ID and the Node corresponding to split_col > split_val has the
            # higher ID
            if val_to_check <= split_value:
                new_node_id = min(child_node_ids)
            else:
                new_node_id = max(child_node_ids)
            current_node = self._nodes[new_node_id]

        # after we're out of the while loop, get the current distribution to see what we need to predict
        current_distribution = self.node_distributions_[current_node.identifier]

        # if we have a clear winner (highest vote), predict that
        if current_distribution.min() != current_distribution.max():
            return current_distribution.argmax()

        # else, predict a random value
        else:
            return np.random.choice(self.values_to_predict_.shape[0])

    def _predict_single_proba(self, instance):
        '''Predict class probabilities for a single instance

        Parameters
        ----------
        instance : pd.Series or dict
            Object which has the desired implementation of the .keys() method

        Returns
        -------
        probas : numpy array
            Array of shape (n_classes,), where n_classes is the number of classes trained on
        '''

        # implementation of this function is similar to the implementation of _predict_single_instance
        # except return the node distributions normalized to sum to 1
        current_node = self._nodes[0]
        while not (current_node.split is np.nan):
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
        
        return self.node_distributions_[current_node.identifier] / self.node_distributions_[current_node.identifier].sum()

    def predict(self, x):
        '''Predict classes for a set of values
        
        Parameters
        ----------
        x : pandas DataFrame
            DataFrame to predict from

        Returns
        -------
        preds : numpy array
            Numpy array of predictions
        '''
        if not self._is_fit:
            raise NotFitError

        # this function just applies the _predict_single_instance method to each of the rows
        return x.apply(lambda row : self._predict_single_instance(row), axis = 1).values

    def predict_proba(self, x):
        '''Predict class probabilities for a set of values

        Parameters
        ----------
        x : pandas DataFrame
            DataFrame to predict from
        
        Returns
        -------
        preds : numpy array
            Numpy array of predicted probabilities. Column indices correspond to same classes
        '''
        if not self._is_fit:
            raise NotFitError

        # this function just applies the _predictsingle_proba method
        probs = x.apply(lambda row : self._predict_single_proba(row), axis = 1).values
        return np.array([p.tolist() for p in probs])

    def score(self, x, y):
        '''Score the model's performance on new labeled data using accuracy score as a measure

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
        '''
        preds = self.predict(x)
        return accuracy_score(y, preds)

    def fit_predict(self, x, y):
        '''Fit the model and predict on X

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
        '''
        self.fit(x, y)
        return self.predict(x)
