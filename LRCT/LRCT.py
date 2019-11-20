from LRCT.splitting_functions import find_best_split, find_best_lrct_split
from LRCT.Node import Node
from LRCT.Exceptions import AlreadyFitError, NotFitError
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings

class LRCTree:
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
        self._is_fit = False

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
        return [n for n in self._nodes.values()]
    
    def _add_nodes(self, nodes):
        
        is_node = isinstance(nodes, Node)
        acceptable_list = isinstance(nodes, list) and all([isinstance(n, Node) for n in nodes])
        
        if not (is_node or acceptable_list):
            raise ValueError('adding nodes requires Node objects or list of Node objects')

        existing_ids = [id for id in self._nodes.keys()]
        
        if is_node and nodes.identifier in existing_ids:
            raise ValueError('Node with that ID already exists')
        if acceptable_list and any([n.identifier in existing_ids for n in nodes]):
            raise ValueError('Trying to set Node with existing ID')
        
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

        x_copy = x_data.copy()
        
        node = self._nodes[node_id]
        parent_id = node.identifier
        parent_depth = node.depth
        
        #do not continue if already at max depth
        if parent_depth == self.max_depth:
            return None
        
        highest_id = max(self._nodes.keys())

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

        if self._is_fit:
            raise AlreadyFitError
        if not isinstance(x, pd.DataFrame):
            raise TypeError('x must be pandas DataFrame')
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError('y must be ndarray or Series')
        if len(y.shape) != 1:
            raise ValueError('y must have a single dimension')
        if self._nodes != {}:
            raise ValueError('Tree already has nodes.  Must fit tree with no nodes')

        # add the parent Node
        self._add_nodes(Node())

        node_data = {
                0 : {
                    'x' : x,
                    'y' : y
                }
            }

        while self.depth < self.max_depth:
            current_depth_nodes = [n for n in self.nodes if n.depth == self.depth]
            num_nodes = len(self.nodes)
            
            for n in current_depth_nodes:
                split_results = self._split_node(n.identifier, node_data[n.identifier]['x'], node_data[n.identifier]['y'])
                if split_results is not None:
                    less_id = split_results[0]
                    greater_id = split_results[1]
                    x_less = split_results[2]
                    x_greater = split_results[3]
                    y_less = split_results[4]
                    y_greater = split_results[5]

                    node_data[less_id] = {
                        'x' : x_less,
                        'y' : y_less
                    }

                    node_data[greater_id] = {
                        'x' : x_greater,
                        'y' : y_greater
                    }
            new_num_nodes = len(self.nodes)
            if new_num_nodes == num_nodes:
                break
        
        self._is_fit = True
        self._values_to_predict = np.unique(y)
        self._node_distributions = {
            n.identifier : np.array([(node_data[n.identifier]['y'] == i).sum() for i in self._values_to_predict])
            for n in self.nodes if n.split is np.nan
        }
        return self

    def _predict_single_instance(self, instance):
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
        
        current_distribution = self._node_distributions[current_node.identifier]
        if current_distribution.min() != current_distribution.max():
            return current_distribution.argmax()
        else:
            return np.random.choice(self._values_to_predict.shape[0])

    def _predict_single_proba(self, instance):
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
        
        return self._node_distributions[current_node.identifier] / self._node_distributions[current_node.identifier].sum()

    def predict(self, x):
        if not self._is_fit:
            raise NotFitError
        
        return x.apply(lambda row : self._predict_single_instance(row), axis = 1).values

    def predict_proba(self, x):
        if not self._is_fit:
            raise NotFitError

        probs = x.apply(lambda row : self._predict_single_proba(row), axis = 1).values
        return np.array([p.tolist() for p in probs])

    def score(self, x, y):
        preds = self.predict(x)
        return accuracy_score(y, preds)

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)