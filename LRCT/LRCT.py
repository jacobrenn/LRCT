from splitting_functions import find_best_split, find_best_lrct_split
from Node import Node
import numpy as np
import pandas as pd
import warnings
from splitting_functions import find_best_lrct_split

class LRCTree:

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

        self._nodes = {}

    @property
    def max_depth(self):
        return self._max_depth
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
        if self._nodes == {}:
            print('Empty Tree')
        else:
            print('\n'.join([f'{"-"*n.depth}{n}' for n in self._nodes.values()]))

    def _split_node(self, node_id, x_data, y_data):
        
        #if x_data does not have more than min_samples_split rows or y_data has only one unique value, do nothing
        if x_data.shape[0] < self.min_samples_leaf or np.unique(y_data).shape[0] == 1:
            return

        x_copy = x_data.copy()
        
        node = self._nodes[node_id]
        parent_id = node.identifier
        parent_depth = node.depth
        
        #do not continue if already at max depth
        if parent_depth == self.max_depth:
            return

        
        highest_id = max(self._nodes.keys())

        split_col, split_value = find_best_lrct_split(x_copy, y_data)
        if split_col not in x_copy.columns:
            rest, last_col = split_col.split(' - ')[0], split_col.split(' - ')[1]
            new_coefs = [item.split('*')[0] for item in rest.split(' + ')]
            new_cols = [item.split('*')[1].split('^')[0] for item in rest.split(' + ')]
            new_col_components = []
            for i in range(len(new_coefs)):
                if '^' in new_cols[i]:
                    col, exp = new_cols[i].split('^')[0], new_cols[i].split('^')[1]
                    new_col_components.append(f'{new_coefs[i]}*x_values["{col}"]**{exp}')
                else:
                    new_col_components.append(f'{new_coefs[i]}*x_values["{new_cols[i]}"]')
            new_col_str = ' + '.join(new_col_components)
            new_col_str += f' - x_values["{last_col}"]'
            new_col = eval(new_col_str)
            split_col_values = new_col
        else:
            split_col_values = x_copy[split_col]

        less_idx = split_col_values <= split_value
        greater_idx = split_col_values > split_value

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
        
        # TODO: do some checking for min_samples_leaf and return the right stuff