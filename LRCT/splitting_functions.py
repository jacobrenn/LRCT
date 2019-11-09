import numpy as np
import pandas as pd
from itertools import product, combinations
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def gini_impurity(values, weighted = False):

    if not isinstance(values, np.ndarray):
        raise TypeError('values must be numpy array')
    if len(values.shape) != 1:
        raise ValueError('values have too many dimensions')

    total_num = values.shape[0]
    unique = np.unique(values)

    if unique.shape in [0, 1]:
        return 0
    
    if weighted:
        return (1 - np.array([((values == value).sum()/total_num)**2 for value in unique]).sum()) * total_num
    return 1 - np.array([((values == value).sum()/total_num)**2 for value in unique]).sum()

def _get_split_candidates(col):
    if col.shape[0] == 1:
        return np.nan
    unique_sorted = np.sort(np.unique(col))
    if unique_sorted.shape[0] == 1:
        return np.nan
    return np.array(list(zip(unique_sorted, unique_sorted[1:]))).mean(axis = 1)

def _split_candidate_results(x_col, y_values, candidate):
    idxer = x_col <= candidate

    gini_less = gini_impurity(y_values[idxer], weighted = True)
    gini_greater = gini_impurity(y_values[~idxer], weighted = True)
    return (gini_less + gini_greater) / y_values.shape[0]

def _column_best_split(x_col, y_values):

    candidates = _get_split_candidates(x_col)

    if candidates is np.nan:
        return np.array([np.inf, np.inf])
    
    candidate_ginis = (_split_candidate_results(x_col, y_values, cand) for cand in candidates)
    split_results = np.array(list(zip(candidates, candidate_ginis)))
    return split_results[split_results[:, 1] == split_results[:, 1].min()][0, :]

def find_best_split(x_values, y_values):

    if len(x_values.shape) == 1 or x_values.shape[1] == 1:
        return (0, _column_best_split(x_values, y_values)[0])
    
    split_results = np.apply_along_axis(
        _column_best_split,
        0,
        x_values,
        y_values = y_values
    )

    if split_results[1, :].min() is np.inf:
        return np.nan

    col_num = split_results[1, :].argmin()
    return (col_num, split_results[0, col_num])

def _single_col_bin(col, n_bins = 10):
    m, M = col.min(), col.max()
    return np.linspace(m, M, n_bins + 1)    

def get_bin_coordinates(x_values, col_indices = None, bins_per_var = 10):

    if len(x_values.shape) == 1:
        return _single_col_bin(x_values, bins_per_var)

    if col_indices is None:
        return np.apply_along_axis(_single_col_bin, 0, x_values, n_bins = bins_per_var)
    
    subset = x_values[:, col_indices]
    return np.apply_along_axis(_single_col_bin, 0, subset, n_bins = bins_per_var)

def get_surface_coords(x_values, y_values, bin_col_indices, target_col_index, bins_per_var = 10):
    
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
            split = find_best_split(x_subset[:, target_col_index], y_subset)[1]
        except IndexError:
            split = np.inf
        sub_array.append(split)
        coord_array.append(sub_array)
    coord_array = np.array(coord_array)
    return coord_array[coord_array[:, -1] != np.inf]

def create_surface_function(surface_coords, column_names, highest_degree = 1, fit_intercept = True, method = 'ols', **kwargs):
    if method == 'ols':
        model = LinearRegression(fit_intercept = fit_intercept, **kwargs)
    elif method == 'ridge':
        model = Ridge(fit_intercept = fit_intercept, **kwargs)
    elif method == 'lasso':
        model = LinearRegression(fit_intercept = fit_intercept, **kwargs)
    else:
        raise ValueError(f'Accepted values for `mathod` are `ols`, `ridge`, and `lasso`, got {method}')

    surface_coords = pd.DataFrame(surface_coords, columns = column_names)
    if highest_degree > 1:
        for col in surface_coords.columns[:-1]:
            for degree in range(2, highest_degree + 1):
                surface_coords[f'{col}**{degree}'] = surface_coords[col]**degree

    to_predict = surface_coords[column_names[-1]]
    predict_from = surface_coords[[col for col in surface_coords.columns if col != column_names[-1]]]

    model.fit(predict_from.values, to_predict.values)
    coefs = model.coef_
    
    ret_function = ' + '.join([f'{coefs[i]}*{predict_from.columns[i]}' for i in range(coefs.shape[0])]) 
    ret_function += f' - {column_names[-1]}'
    return ret_function.replace('**','^')

def find_best_lrct_split(x_values, y_values, num_independent = 1, highest_degree = 1, fit_intercept = True, method = 'ols', n_bins = 10, **kwargs):
    cols = x_values.columns.tolist()
    x_numpy = x_values.values
    x_copy = x_values.copy()
    all_indices = combinations(range(len(cols)), num_independent + 1)
    for ind in all_indices:
        surface_coords = get_surface_coords(
            x_numpy,
            y_values,
            bin_col_indices = ind[:-1],
            target_col_index = ind[-1],
            bins_per_var = n_bins
        )
        surface_function = create_surface_function(
            surface_coords = surface_coords,
            column_names = [cols[i] for i in ind],
            highest_degree = highest_degree,
            fit_intercept = fit_intercept,
            method = method,
            **kwargs
        )

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
        new_col_str = ' + '.join(new_col_components)
        new_col_str += f' - x_values["{last_col}"]'
        x_copy[surface_function] = eval(new_col_str)
    col_num, split_value = find_best_split(x_copy, y_values)
    return x_copy.columns[col_num], split_value