import numpy as np
import pandas as pd
from itertools import product, combinations
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def gini_impurity(values, weighted = False):
    '''Calculate the Gini impurity measure of a set of values

    Parameters
    ----------
    values : 1d numpy array
        The values to calculate score from
    weighted : bool (default False)
        Whether to return the weighted Gini impurity

    Returns
    -------
    impurity : float
        The impurity score
    '''
    # REMOVING TYPE CHECKING TO SPEED UP
    # type checking
    #if not isinstance(values, np.ndarray):
    #    raise TypeError('values must be numpy array')
    #if len(values.shape) != 1:
    #    raise ValueError('values have too many dimensions')

    # so these numbers don't have to be calculated multiple times
    total_num = values.shape[0]
    unique = np.unique(values)

    # short cut
    if unique.shape in [0, 1]:
        return 0
    
    # calculate the score as needed
    if weighted:
        return (1 - np.array([((values == value).sum()/total_num)**2 for value in unique]).sum()) * total_num
    return 1 - np.array([((values == value).sum()/total_num)**2 for value in unique]).sum()

def _get_split_candidates(col):
    '''Return all candidates for splitting from a column, i.e. the midpoints 
    between all unique points in the column

    Parameters
    ----------
    col : 1d numpy array
        The column to find candidates for

    Notes
    -----
    - Returns np.nan if exactly one unique value in the column

    Returns
    -------
    candidates : numpy array
        The splitting candidates
    '''
    unique_sorted = np.sort(np.unique(col))
    if unique_sorted.shape[0] == 1:
        return np.nan
    return np.array(list(zip(unique_sorted, unique_sorted[1:]))).mean(axis = 1)

def _evaluate_split_candidate(x_col, y_values, candidate):
    '''Calculate the results of splitting a column on a single value

    Parameters
    ----------
    x_col : 1d numpy array
        The column in X
    y_values : 1d numpy array
        The target values
    candidate : numeric
        The value to split x_col on

    Returns
    -------
    impurity : float
        The resulting weighted impurity from making the requested split
    '''
    # get the index
    idxer = x_col <= candidate

    # calculated the weighted Gini impurity for each 'side'
    gini_less = gini_impurity(y_values[idxer], weighted = True)
    gini_greater = gini_impurity(y_values[~idxer], weighted = True)
    return (gini_less + gini_greater) / y_values.shape[0]

def _column_best_split(x_col, y_values):
    '''Calculate the best split for a column

    Parameters
    ----------
    x_col : 1d numpy array
        The column of X
    y_values : 1d numpy array
        The target values

    Notes
    -----
    - If no split candidates are possible, returns array([np.inf, np.inf])

    Returns
    -------
    split_results : numpy array, shape [2,0]
        The split results, in order of value, resulting_impurity
    '''
    candidates = _get_split_candidates(x_col)

    if candidates is np.nan:
        return np.array([np.inf, np.inf])
    
    candidate_ginis = (_evaluate_split_candidate(x_col, y_values, cand) for cand in candidates)
    split_results = np.array(list(zip(candidates, candidate_ginis)))
    return split_results[split_results[:, 1] == split_results[:, 1].min()][0, :]

def find_best_split(x_values, y_values):
    '''Find the best split for a set of values given a target

    Parameters
    ----------
    x_values : 1d or 2d numpy array
        The X
    y_values : 1d numpy array
        The target values

    Notes
    -----
    - If no split is found, returns np.nan

    Returns
    -------
    split_results : tuple
        Tuple of the form (column_index, split_value) for the best split
    '''
    if np.unique(y_values).shape[0] == 1:
        return np.nan

    if len(x_values.shape) == 1 or x_values.shape[1] == 1:
        return (0, _column_best_split(x_values, y_values)[0])
    
    split_results = np.apply_along_axis(
        _column_best_split,
        0,
        x_values,
        y_values = y_values
    )

    if split_results[1, :].min() == np.inf:
        return np.nan

    col_num = split_results[1, :].argmin()
    return (col_num, split_results[0, col_num])

def _single_col_bin(col, n_bins = 10):
    '''Bin a single column'''
    return np.linspace(
        col.min(),
        col.max(),
        n_bins + 1
    )

def get_bin_coordinates(x_values, col_indices = None, bins_per_var = 10):
    '''Get the bin coordinates for a set of values

    Parameters
    ----------
    x_values : np.array
        The values to bin
    col_indices : index-like or None (default None)
        The column indices to use for finding bin values
    bins_per_var : int (default 10)
        The number of bins per variable to create

    Returns
    -------
    bins : numpy array
        Array of the bin values, shape (n_bins, n_cols)
    '''
    if len(x_values.shape) == 1:
        return _single_col_bin(x_values, bins_per_var)

    if col_indices is None:
        col_indices = np.arange(x_values.shape[1])
    
    subset = x_values[:, col_indices]
    return np.apply_along_axis(_single_col_bin, 0, subset, n_bins = bins_per_var)

def get_surface_coords(x_values, y_values, bin_col_indices, target_col_index, bins_per_var = 10):
    '''Get the approximate bin surface coordinates for the surface function

    Parameters
    ----------
    x_values : numpy array
        The values to bin
    y_values : numpy array
        The target values
    bin_col_indices : index-like
        Columns to use for binning
    target_col_index : int
        The column to use in for splitting
    bins_per_var : int (default 10)
        The number of bins to use per variable

    Returns
    -------
    surface_coords : np.array
        The surface coordinates
    '''

    #raw_bin_coords is an array of coordinates
    raw_bin_coords = get_bin_coordinates(x_values, bin_col_indices, bins_per_var)

    #all_indices is an iterable of indices referencing raw_bin_coords
        #Ex: [[0, 0, 0], [0, 0, 1], ...]
    all_indices = product(range(bins_per_var), repeat = len(bin_col_indices))

    #coord_array is the array to return
    coord_array = []

    #iterate over all of the indices: e.g. ind = [0, 0, 1]
    for ind in all_indices:
        
        #sub_array will be added to the coord_array
        sub_array = []
        
        #create copies to be sure of overwriting things
        x_subset = x_values.copy()
        y_subset = y_values.copy()

        #bin_col_indices[i] is the column number
        #ind[i] is the row index on 
        for i in range(len(bin_col_indices)):
            
            #col_ind is the *row* index for the specific column -- i.e. things that are in the bin for that col
            col_ind = (x_subset[:, bin_col_indices[i]] >= raw_bin_coords[ind[i], i]) & (x_subset[:, bin_col_indices[i]] <= raw_bin_coords[ind[i] + 1, i])
            #append the midpoint of the bin in this variable to sub_array
            sub_array.append(np.mean([raw_bin_coords[ind[i], i], raw_bin_coords[ind[i] + 1, i]]))

            #create the appropriate subset
            x_subset = x_subset[col_ind]
            y_subset = y_subset[col_ind]
        
        #get the coordinate in the dimension of target_col
        try:
            split_info = find_best_split(x_subset[:, target_col_index], y_subset)
            if split_info is np.nan:
                split = np.inf
            else:
                split = split_info[1]
        except IndexError:
            split = np.inf
        sub_array.append(split)
        coord_array.append(sub_array)
    coord_array = np.array(coord_array)
    return coord_array[coord_array[:, -1] != np.inf]

def get_surface_coef(surface_coords, highest_degree = 1, fit_intercept = True, method = 'ols', **kwargs):
    '''Creates the estimate to the surface function

    Parameters
    ----------
    surface_coords : numpy array
        Coordinates from the surface, format expected from the get_surface_coords function
    column_names : iterable
        Names of the columns for the variables
    highest_degree : int (default 1)
        Highest degree to use for any single one of the variables
    fit_intercept : bool (default True)
        Whether to fit the intercepts in the regression training
    method : str (default 'ols')
        One of 'ols', 'ridge', and 'lasso' -- the method of linear regression to use
    **kwargs : additional arguments
        Additional arguments to pass to the linear regression model, if desired

    Returns
    -------
    surface_funciton : str
        A string corresponding to the surface function that is fit
    '''

    # checking for the different possibilities for method
    if method == 'ols':
        model = LinearRegression(fit_intercept = fit_intercept, **kwargs)
    elif method == 'ridge':
        model = Ridge(fit_intercept = fit_intercept, **kwargs)
    elif method == 'lasso':
        model = Lasso(fit_intercept = fit_intercept, **kwargs)
    else:
        raise ValueError(f'Accepted values for `method` are `ols`, `ridge`, and `lasso`, got {method}')

    ### THIS PART CAN BE IMPROVED SIGNIFICANTLY -- I SUSPECT THIS IS WHERE WE GET OUR SLOW PERFORMANCE PROBLEMS

    # separate the prediction and the predictor columns
    predictor_columns = surface_coords[:, :-1]
    prediction_column = surface_coords[:, -1]

    # create the additional columns (for higher degrees)
    # order will be matrices of increasing degree appended together
    if highest_degree > 1:
        for degree in range(2, highest_degree + 1):
            for col_idx in range(predictor_columns.shape[1]):
                predictor_columns = np.concatenate([predictor_columns, (predictor_columns[:, col_idx]**degree).reshape(-1, 1)], axis = 1)

    # fit the regression model
    model.fit(predictor_columns, prediction_column.reshape(-1, 1))
    return model.coef_

def find_best_lrct_split(x_values, y_values, num_independent = 1, highest_degree = 1, fit_intercept = True, method = 'ols', n_bins = 10, **kwargs):
    '''Find the best split on data using LRCT methods

    Parameters
    ----------
    x_values : pandas DataFrame
        A DataFrame of values to use to predict from
    y_values : 1d numpy array
        The target values to predict
    num_independent : int (default 1)
        Number of variables to use as independent variables in LRCT surface learning
    highest_degree : int (default 1)
        The highest degree to take independent variables to
    fit_intercept : bool (default True)
        Whether to fit intercepts in the linear regressions that are trained
    method : str (default 'ols')
        One of 'ols', 'ridge', and 'lasso' -- the method of linear regression to use
    n_bins : int (default 10)
        The number of bins to use per independent variable in determining surface coordinates
    **kwargs : additional arguments
        Additional arguments to pass to the linear regression model, if desired

    Notes
    -----
    - Returns np.nan if no best split can be found

    Returns
    -------
    split_info : tuple
        A tuple of the form (column name, split_value)
    '''

    # separate columns from the values
    cols = x_values.columns.tolist()
    x_numpy = x_values.values
    x_copy = x_values.copy()

    # get the combinations of columns that we will use
    all_indices = combinations(range(len(cols)), num_independent + 1)

    # get all of the surface coordinates fo each of the combinations of columns to determine the coordinates from
    for ind in all_indices:
        surface_coords = get_surface_coords(
            x_numpy,
            y_values,
            bin_col_indices = ind[:-1],
            target_col_index = ind[-1],
            bins_per_var = n_bins
        )
        try:
            surface_function = create_surface_function(
                surface_coords = surface_coords,
                column_names = [cols[i] for i in ind],
                highest_degree = highest_degree,
                fit_intercept = fit_intercept,
                method = method,
                **kwargs
            )
            # parse the surface function
            rest, last_col = surface_function.split(' - ')[0], surface_function.split(' - ')[1]
            new_coefs = [item.split('*')[0] for item in rest.split(' + ')]
            new_cols = [item.split('*')[1].split('^')[0] for item in rest.split(' + ')]
            new_col_components = []
            for i in range(len(new_coefs)):
                if '^' in new_cols[i]:
                    col, exp = new_cols[i].split('^')[0], new_cols[i].split('^')[1]
                    new_col_components.append(f'{new_coefs[i]}*x_values["{col}"]**{exp}')
                else:
                    new_col_components.append(f'{new_coefs[i]}*x_values["{new_cols[i]}"]')

            # rewrite the surface function so that it works as `eval`
            new_col_str = ' + '.join(new_col_components)
            new_col_str += f' - x_values["{last_col}"]'

            # create a new function using the surface function
            x_copy[surface_function] = eval(new_col_str)
        except ValueError:
            pass

    # now just find the best split using traditional methods
    split_info = find_best_split(x_copy, y_values)

    # returns split info or nan if the split info gives nothing
    if split_info is np.nan:
        return split_info
    else:
        return x_copy.columns[split_info[0]], split_info[1]
