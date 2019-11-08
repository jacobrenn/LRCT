from splitting_functions import find_best_split, find_best_lrct_split
from Node import Node
import numpy as np
import pandas as pd
import warnings

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

    def _find_node_split(self, node_id, x_data, y_data):
        pass