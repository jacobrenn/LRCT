import numpy as np

class Node:
    '''A Node object to be used inside a Tree object as a splitting point'''
    def __init__(
        self,
        identifier = 0,
        parent_id = None,
        depth = 0,
        split = np.nan,
    ):
        '''
        Parameters
        ----------
        identifier : int (default 0)
            The ID of the Node
        parent_id : int or None (default None)
            The ID of the parent Node
        depth : int (default 0)
            The depth the Node resides at in the Tree
        split : tuple of length 2 or np.nan (default np.nan)
            The spilt made at the Node
        '''
        self.identifier = identifier
        self.parent_id = parent_id
        self.depth = depth
        self.split = split

    def __str__(self):
        return f'ID: {self.identifier}, Parent ID: {self.parent_id}, Depth: {self.depth}, Split: {self.split}'
    def __repr__(self):
        return self.__str__()

    @property
    def identifier(self):
        return self._identifier
    @identifier.setter
    def identifier(self, value):
        if not isinstance(value, int):
            raise TypeError('identifier must be integer')
        if value < 0:
            raise ValueError('identifier must be greater than 0')
        self._identifier = value
    
    @property
    def parent_id(self):
        return self._parent_id
    @parent_id.setter
    def parent_id(self, value):
        nn = value is not None
        if not isinstance(value, int) and nn:
            raise TypeError('parent_id must be None or integer')
        if nn and value < 0:
            raise ValueError('parent_id must be greater than 0')
        self._parent_id = value

    @property
    def depth(self):
        return self._depth
    @depth.setter
    def depth(self, value):
        if not isinstance(value, int):
            raise TypeError('depth must be integer valued')
        if value < 0:
            raise ValueError('depth must be nonnegative')
        self._depth = value

    @property
    def split(self):
        return self._split
    @split.setter
    def split(self, value):
        nn = value is not np.nan
        if not isinstance(value, tuple) and nn:
            raise TypeError('split must be tuple or np.nan')
        if nn and len(value) != 2:
            raise ValueError('split must be tuple of length 2 or np.nan')
        self._split = value

    def describe(self):
        print(self.__repr__())
